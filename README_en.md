# IBF Comic Translator Local

[🇷🇺 Русский](README.md) | [🇨🇳 中文](README_zh.md)

A local comic translator running directly in your browser.

The project consists of two parts:
- `extension/` — Chrome extension that adds an `IBF` button to images on the page, sends the image to a local API, and replaces it with the translated version
- `server/` — FastAPI server that performs OCR, translation, and text rendering over the original image

**Tech stack:**
- OCR: `PaddleOCR 3.5` (supports `engine='transformers'` — no PaddlePaddle required)
- Translation: `tencent/Hy-MT2-1.8B` (GGUF via llama-cpp-python)
- Runtime: `Linux x86_64` or `WSL2`

---

## Requirements

- Linux x86_64 or WSL2
- Python 3.12
- Google Chrome or Chromium
- 4 GB RAM minimum (8 GB+ recommended)

For GPU mode additionally:
- NVIDIA GPU ([CUDA Toolkit 13.3](https://developer.nvidia.com/cuda-downloads))
- Latest [NVIDIA driver](https://www.nvidia.com/en-us/drivers/) installed

Check your environment before installation:

```bash
python3 --version   # requires 3.12 (use pyenv to install a specific version)
nvidia-smi          # GPU mode only
```

---

## Quick start with Docker

### CPU mode
```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local/server/
bash download-model.sh
docker compose --profile cpu up -d --build
```

### GPU mode
```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local/server/
bash download-model.sh
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.') docker compose --profile gpu up -d --build
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local
```

### 2. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

### 3. Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Download the translation model

```bash
bash download-model.sh
```

The script downloads `Hy-MT2-1.8B-Q8_0.gguf` (~1.9 GB) to `server/models/`.

For a lighter version:
```bash
bash download-model.sh Hy-MT2-1.8B-Q4_K_M.gguf   # ~1.1 GB
bash download-model.sh Hy-MT2-1.8B-Q2_K.gguf      # ~0.7 GB
```

### 5. Python dependencies

From the `server/` directory:

```bash
cd server
```

**CPU:**
```bash
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
```

**GPU:**
```bash
bash install-linux-gpu.sh
```

---

## Running the server

```bash
cd server
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

On first launch, the server downloads PaddleOCR models (~200 MB). This takes 1–2 minutes.

Verify the server started:
```bash
curl http://127.0.0.1:8000/health
# {"ok": true}
```

Expected log output:
```
[STARTUP] Translator warm-up complete
[STARTUP] PaddleOCR en ready
```

---

## Installing the Chrome extension

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top right corner)
3. Click **Load unpacked**
4. Select the `extension/` folder

After loading:
- Open the extension popup
- Make sure the extension is enabled
- Select OCR language and translation language as needed

> For WSL2: the server is accessible from Chrome on Windows at `http://127.0.0.1:8000`

---

## How it works

1. The extension captures an image from the browser
2. Sends it to `POST /translate-upload`
3. Server: OCR → block grouping → translation → inpaint → render
4. The extension replaces the image on the page

---

## Extension settings

Available in the popup:
- Enable / disable the extension
- `replace` — replace the original image with the translated one
- `overlay` — show the translation on top of the original
- OCR language (en, ja, ko, zh, etc.)
- Translation language

---

## Troubleshooting

**Check torch:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Check PaddleOCR:**
```bash
python -c "from app.ocr import get_paddleocr_engine; ocr = get_paddleocr_engine('en'); print('OK')"
```

**Check llama-cpp-python with GPU:**
```bash
python -c "from llama_cpp import Llama; print('OK')"
```
With GPU build, the model loading logs should show `ggml_cuda_init: found N CUDA devices`.

**OCR not finding text:**
- Check `debug_dir` in the server's meta response
- Open `grouped_blocks.png` in the debug folder
- Check `crop_*.png` for each block

---

## Optional: PatchMatch inpaint

PatchMatch improves text erasing quality on complex backgrounds.

```bash
sudo apt install -y build-essential git pkg-config libopencv-dev
cd server
bash install-patchmatch.sh
export COMIC_TRANSLATOR_INPAINT_BACKEND=patchmatch
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

If PatchMatch is unavailable, the server automatically falls back to TELEA.

---

## Project structure

```
extension/          Chrome extension
server/
  app/              Main code (OCR, translation, rendering)
  models/           GGUF models (downloaded via download-model.sh)
  results/          Translation results
README.md
```

---

## Limitations

- Translation speed depends on GPU. On CPU, 1.8B GGUF gives ~15–25 tok/s (~2–8s per page)
- Complex manga pages may produce noisy OCR blocks
- Some sites block direct image access
- CPU mode works but is noticeably slower than GPU

---

## License

MIT
