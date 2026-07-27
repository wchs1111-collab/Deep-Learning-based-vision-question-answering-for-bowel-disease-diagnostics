import io
import torch
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent  # xinqifan code/
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = str(BASE_DIR / "qwen2.5-3b-instruct-trl-sft-kvasir-vqa")

# 与训练时相同的系统提示词
SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in interpreting visual data from medical images. "
    "Your task is to analyze the provided gastrointestinal medical image and respond to queries with concise answers, "
    "usually a single word, number, or short phrase. "
    "Focus on delivering accurate, succinct answers based on the visual information. "
    "Avoid additional explanation unless absolutely necessary."
)

model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型，关闭时释放资源。"""
    global model, processor
    print(f"正在加载模型 {MODEL_ID} （4-bit 量化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model.load_adapter(ADAPTER_PATH)
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)
    print("模型加载完成，服务已就绪。")
    yield
    del model, processor


app = FastAPI(lifespan=lifespan)

# 允许前端跨域访问（开发阶段可以先用 "*"，上线后建议改成具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 512) -> str:
    """使用微调后的 Qwen2.5-VL 模型对图像进行视觉问答推理。"""
    sample = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]
    text_input = processor.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(sample)
    device = next(model.parameters()).device
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from backend!"}


@app.post("/api/vqa")
async def visual_question_answering(
    image: UploadFile = File(..., description="胃肠道医学图像文件"),
    question: str = Form(..., description="关于图像的问题"),
):
    """
    视觉问答接口：上传医学图像 + 提问，返回模型答案。

    - **image**: 图像文件（支持 JPEG/PNG 等常见格式）
    - **question**: 关于图像的自然语言问题
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图像文件")
    answer = generate_answer(pil_image, question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)