# Deep-Learning-based-vision-question-answering-for-bowel-disease-diagnostics

## Project Overview

This project implements a deep-learning-based medical visual question answering system for bowel disease diagnostic support. It uses Qwen2.5-VL-3B-Instruct as the base model and applies supervised fine-tuning on the Kvasir-VQA dataset. LoRA-based parameter-efficient fine-tuning is used to investigate how different rank values and attention-module configurations affect model performance. Through the web interface, users can upload gastrointestinal endoscopy images and ask image-related questions, after which the model generates concise medical answers. The FastAPI backend provides endpoints for image upload, visual question answering, and history retrieval and deletion. It uses 4-bit quantization to reduce inference memory requirements and SQLite to store question-answer records. The frontend is built with Vue, TypeScript, and Vite, providing image previews, question submission, result display, and history management. The repository also contains a complete training and evaluation notebook, multiple checkpoints, and adapter weights for reproducing experiments, comparing configurations, and deploying local inference. This system is intended solely for research and educational purposes and must not replace professional medical diagnosis.

## Usage

### 1. Requirements

- Python 3.10 or 3.11
- Node.js 22.18 or later
- A CUDA-compatible NVIDIA GPU with sufficient VRAM
- An internet connection on the first run to download `Qwen/Qwen2.5-VL-3B-Instruct`

### 2. Start the Backend

Open PowerShell in the project root, then create and activate an isolated Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision transformers peft accelerate bitsandbytes qwen-vl-utils fastapi "uvicorn[standard]" python-multipart pillow
```

Ensure that the fine-tuned adapter is located at `xinqifan code/qwen2.5-3b-instruct-trl-sft-kvasir-vqa`, then start the service:

```powershell
Set-Location ".\xinqifan code\backweb"
python main.py
```

After the model has loaded, the backend is available at `http://localhost:8080`, and the interactive API documentation is available at `http://localhost:8080/docs`.

### 3. Start the Frontend

Open another PowerShell terminal in the project root and run:

```powershell
Set-Location ".\xinqifan code\Front project"
npm install
npm run dev
```

Open the local URL displayed in the terminal, usually `http://localhost:5173`. Upload a gastrointestinal endoscopy image, enter a related question, and submit it to view the model's answer. Previous question-answer records can be viewed or deleted from the history page.

If the backend is hosted at a different address, set the environment variable before starting the frontend. For example:

```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8080"
npm run dev
```

### 4. Run the Training and Evaluation Notebook

Open `xinqifan code/Vision_Language_Models_Qwen2_5_VL_3B_Kvasir_VQA.ipynb` in Jupyter and run the cells in order to explore data processing, model fine-tuning, metric evaluation, and inference service examples. Before training, adjust the dataset path, model output path, batch size, and other parameters to match the local environment.