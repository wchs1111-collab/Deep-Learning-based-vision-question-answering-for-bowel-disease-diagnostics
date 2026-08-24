import io
import shutil
import sqlite3
import uuid
import torch
from datetime import datetime
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

# 数据库 & 图片存储路径（与 main.py 同目录）
_HERE = Path(__file__).resolve().parent
DB_PATH = str(_HERE / "vqa_history.db")
UPLOADS_DIR = _HERE / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

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



def init_db() -> None:
  
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vqa_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at     TEXT    NOT NULL,
            image_filename TEXT    NOT NULL,
            image_path     TEXT    NOT NULL,
            question       TEXT    NOT NULL,
            answer         TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def db_save(image_filename: str, image_path: str, question: str, answer: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO vqa_history "
        "(created_at, image_filename, image_path, question, answer) VALUES (?,?,?,?,?)",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_filename,
            image_path,
            question,
            answer,
        ),
    )
    record_id = cur.lastrowid
    conn.commit()
    conn.close()
    return record_id


def db_list(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM vqa_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_delete(record_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM vqa_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型，关闭时释放资源。"""
    global model, processor
    init_db()
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 512) -> str:
    """使用微调后的 Qwen2.5-VL 模型对图像进行视觉问答推理。"""

    # ── Step 1: 构造多轮对话格式 ──────────────────────────────────────────────
    # 按照 Qwen2.5-VL 的对话协议组织输入：
    #   - system 角色：注入与训练时相同的系统提示词，约束模型输出简洁的医学答案
    #   - user 角色：同时包含图像对象和文字问题
    sample = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},   # PIL Image 对象
                {"type": "text", "text": question},   # 用户提问
            ],
        },
    ]

    # ── Step 2: 将对话模板转换为模型可接受的文本字符串 ────────────────────────
    # apply_chat_template 把对话列表渲染成带有特殊 token 的提示字符串
    # tokenize=False 表示只生成字符串，不直接分词（后续由 processor 统一处理）
    # add_generation_prompt=True 在末尾追加触发模型生成回答的起始 token
    text_input = processor.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ── Step 3: 提取图像特征输入 ───────────────────────────────────────────────
    # process_vision_info 从对话结构中解析出图像数据（像素值等），
    # 返回值第二项为视频帧（此处不需要，用 _ 忽略）
    image_inputs, _ = process_vision_info(sample)

    # ── Step 4: 将文本 + 图像一起编码为模型输入张量，并转移到 GPU ─────────────
    # next(model.parameters()).device 自动获取模型所在的设备（CPU / CUDA）
    # processor 同时处理文本 token 和图像 patch，输出 PyTorch 张量
    device = next(model.parameters()).device
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # ── Step 5: 自回归生成回答 token 序列 ─────────────────────────────────────
    # max_new_tokens 限制最多生成 512 个新 token，防止输出过长
    # 贪婪解码：每步都取概率最高的 token，保证同一输入的结果可复现
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # ── Step 6: 裁剪掉输入部分，只保留新生成的 token ──────────────────────────
    # generated_ids 包含完整序列（输入 + 输出），
    # 通过切片 out_ids[len(in_ids):] 去掉输入 token，仅留下模型生成的内容
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # ── Step 7: 将 token ID 解码为可读文本并返回 ──────────────────────────────
    # skip_special_tokens=True 自动过滤 <|im_end|> 等特殊 token
    # [0] 取 batch 中第一条（本服务每次只处理单张图片）
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

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无效的图像文件")

    
    original_name = image.filename or "upload.jpg"
    safe_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    save_path = UPLOADS_DIR / safe_name
    save_path.write_bytes(contents)

    answer = generate_answer(pil_image, question)
    record_id = db_save(original_name, str(save_path), question, answer)
    return {"answer": answer, "id": record_id}


@app.get("/api/history")
def get_history(limit: int = 50):

    return {"history": db_list(limit)}


@app.delete("/api/history/{record_id}")
def delete_history(record_id: int):
  
    db_delete(record_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)