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

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent  # xinqifan code/
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = str(
    BASE_DIR / "qwen2.5-3b-instruct-trl-sft-kvasir-vqa"
    
)

# Database and image storage paths (in the same directory as main.py)
_HERE = Path(__file__).resolve().parent
DB_PATH = str(_HERE / "vqa_history.db")
UPLOADS_DIR = _HERE / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Use the same system prompt as during training
SYSTEM_MESSAGE = (
    "You are a Vision Language Model specialized in interpreting visual data from medical images.\n"
    "Your task is to analyze the provided gastrointestinal medical image and respond to queries with concise answers, usually a single word, number, or short phrase.\n"
    "Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."
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
    """Load the model at startup and release resources at shutdown."""
    global model, processor
    init_db()
    print(f"Loading model {MODEL_ID} (4-bit quantization)...")
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
    print("Model loaded. The service is ready.")
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
    """Run visual question answering on an image with the fine-tuned Qwen2.5-VL model."""

    # ── Step 1: Build the multi-turn conversation structure ───────────────────
    # Organize the input according to the Qwen2.5-VL conversation protocol:
    #   - system role: inject the training system prompt to elicit concise medical answers
    #   - user role: include both the image object and the text question
    sample = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},   # PIL Image object
                {"type": "text", "text": question},   # User question
            ],
        },
    ]

    # ── Step 2: Convert the conversation template into model-compatible text ──
    # apply_chat_template renders the conversation list as a prompt with special tokens.
    # tokenize=False returns only a string; the processor performs tokenization later.
    # add_generation_prompt=True appends the token that prompts the model to answer.
    text_input = processor.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ── Step 3: Extract the visual inputs ──────────────────────────────────────
    # process_vision_info extracts image data, such as pixel values, from the conversation.
    # The second return value contains video frames and is unused here.
    image_inputs, _ = process_vision_info(sample)

    # ── Step 4: Encode text and images as model tensors and move them to the device ──
    # next(model.parameters()).device identifies the model device (CPU or CUDA).
    # The processor handles text tokens and image patches, returning PyTorch tensors.
    device = next(model.parameters()).device
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # ── Step 5: Autoregressively generate the answer token sequence ────────────
    # max_new_tokens limits output length to at most 512 new tokens.
    # Greedy decoding selects the most probable token for reproducible results.
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # ── Step 6: Remove the input tokens and retain only generated tokens ────────
    # generated_ids contains the full input and output sequence.
    # Slice from len(in_ids) to retain only model-generated content.
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # ── Step 7: Decode token IDs into readable text and return it ──────────────
    # skip_special_tokens=True removes special tokens such as <|im_end|>.
    # [0] selects the first batch item because each request contains one image.
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
    image: UploadFile = File(..., description="Gastrointestinal medical image file"),
    question: str = Form(..., description="Question about the image"),
):

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="The model is not loaded yet. Please try again later.")
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    
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