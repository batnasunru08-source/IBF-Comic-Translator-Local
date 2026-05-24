# IBF Comic Translator Local

直接在浏览器中运行的本地漫画翻译工具。

项目由两部分组成：
- `extension/` — Chrome 扩展，在页面图片上添加 `IBF` 按钮，将图片发送到本地 API，然后替换为翻译后的版本
- `server/` — FastAPI 服务器，执行 OCR、翻译和文字渲染

**技术栈：**
- OCR: `PaddleOCR 3.5`（支持 `engine='transformers'` — 无需 PaddlePaddle）
- 翻译: `tencent/Hy-MT2-1.8B`（GGUF 格式，通过 llama-cpp-python）
- 运行环境: `Linux x86_64` 或 `WSL2`

---

## 系统要求

- Linux x86_64 或 WSL2
- Python 3.12
- Google Chrome 或 Chromium
- 最少 4 GB 内存（推荐 8 GB+）

GPU 模式额外要求：
- NVIDIA GPU（支持 CUDA 11.8+ 即可）
- 已安装 NVIDIA 驱动 + `nvidia-smi`

安装前检查环境：

```bash
python3 --version   # 需要 3.12
nvidia-smi          # 仅 GPU 模式需要
```

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local
```

### 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

### 3. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 安装 Python 依赖

在 `server/` 目录下：

```bash
cd server
```

**GPU 模式：**
```bash
bash install-linux-gpu.sh
```

**CPU 模式：**
```bash
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
```

### 5. 安装 llama-cpp-python

此包需要单独安装 — 编译取决于你的硬件。

**GPU (CUDA)：**

首先查看你的 GPU 架构：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# 示例：NVIDIA GeForce RTX 3090, 8.6  →  86
#       NVIDIA GeForce RTX 4090, 8.9  →  89
#       NVIDIA GeForce RTX 2080, 7.5  →  75
```

然后指定架构安装：
```bash
# 将 86 替换为你上面的架构编号
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install llama-cpp-python --no-cache-dir
```

**CPU 模式：**
```bash
pip install llama-cpp-python
```

### 6. 下载翻译模型

```bash
bash download-model.sh
```

脚本会将 `Hy-MT2-1.8B-Q8_0.gguf`（约 1.9 GB）下载到 `server/models/` 目录。

如需更轻量的版本：
```bash
bash download-model.sh Hy-MT2-1.8B-Q4_K_M.gguf   # ~1.1 GB
bash download-model.sh Hy-MT2-1.8B-Q2_K.gguf      # ~0.7 GB
```

---

## 启动服务器

```bash
cd server
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

首次启动时，服务器会下载 PaddleOCR 模型（约 200 MB），需要 1–2 分钟。

验证服务器是否启动成功：
```bash
curl http://127.0.0.1:8000/health
# {"ok": true}
```

日志中应显示：
```
[STARTUP] Translator warm-up complete
[STARTUP] PaddleOCR en ready
```

---

## 安装 Chrome 扩展

1. 打开 `chrome://extensions/`
2. 启用**开发者模式**（右上角）
3. 点击**加载已解压的扩展程序**
4. 选择 `extension/` 文件夹

加载后：
- 打开扩展弹出窗口
- 确保扩展已启用
- 根据需要选择 OCR 语言和翻译语言

> WSL2 用户：Windows 上的 Chrome 可通过 `http://127.0.0.1:8000` 访问服务器

---

## 工作原理

1. 扩展从浏览器捕获图片
2. 发送到 `POST /translate-upload`
3. 服务器：OCR → 文本块分组 → 翻译 → 修复 → 渲染
4. 扩展将翻译后的图片替换到页面上

---

## 扩展设置

弹出窗口中可用：
- 启用 / 禁用扩展
- `replace` — 用翻译后的图片替换原图
- `overlay` — 在原文上方显示翻译
- OCR 语言（en、ja、ko、zh 等）
- 翻译语言

---

## 故障排除

**检查 torch：**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**检查 PaddleOCR：**
```bash
python -c "from app.ocr import get_paddleocr_engine; ocr = get_paddleocr_engine('en'); print('OK')"
```

**检查 llama-cpp-python GPU：**
```bash
python -c "from llama_cpp import Llama; print('OK')"
```
GPU 模式下，模型加载日志中应显示 `ggml_cuda_init: found N CUDA devices`。

**OCR 未找到文字：**
- 查看服务器 meta 响应中的 `debug_dir`
- 打开 debug 文件夹中的 `grouped_blocks.png`
- 检查每个块的 `crop_*.png`

---

## 可选：PatchMatch 修复

PatchMatch 可提升复杂背景上的文字擦除质量。

```bash
sudo apt install -y build-essential git pkg-config libopencv-dev
cd server
bash install-patchmatch.sh
export COMIC_TRANSLATOR_INPAINT_BACKEND=patchmatch
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

如果 PatchMatch 不可用，服务器会自动回退到 TELEA。

---

## 项目结构

```
extension/          Chrome 扩展
server/
  app/              核心代码（OCR、翻译、渲染）
  models/           GGUF 模型（通过 download-model.sh 下载）
  results/          翻译结果
  requirements.txt
  download-model.sh
  install-linux-gpu.sh
README.md
```

---

## 限制

- 翻译速度取决于 GPU。CPU 上 1.8B GGUF 约 15–25 tok/s（每页约 2–8 秒）
- 复杂的漫画页面可能产生噪声 OCR 块
- 部分网站会阻止直接访问图片
- CPU 模式可用，但明显慢于 GPU

---

## 许可证

MIT
