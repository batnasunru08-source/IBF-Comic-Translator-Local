# IBF Comic Translator Local

[en English](README_en.md) | [🇨🇳 中文](README_zh.md)

Локальный переводчик комиксов прямо из браузера.

Проект состоит из двух частей:
- `extension/` — Chrome-расширение, добавляет кнопку `IBF` на изображения на странице, отправляет изображение на локальный API и подставляет переведённую версию обратно
- `server/` — FastAPI-сервер, который делает OCR, перевод и рендер текста поверх исходной картинки

**Стек:**
- OCR: `PaddleOCR 3.5` (поддерживает `engine='transformers'` — без PaddlePaddle)
- Translation: `tencent/Hy-MT2-1.8B` (GGUF через llama-cpp-python)
- Runtime: `Linux x86_64` или `WSL2`

---

## Требования

- Linux x86_64 или WSL2
- Python 3.12
- Google Chrome или Chromium
- 4 GB RAM минимум (8 GB+ рекомендуется)

Для GPU-режима дополнительно:
- NVIDIA GPU (любая с CUDA 11.8+)
- Установленный NVIDIA драйвер + `nvidia-smi`

Проверьте базовое окружение перед установкой:

```bash
python3 --version   # нужен 3.12
nvidia-smi          # только для GPU-режима
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/batnasunru08-source/IBF-Comic-Translator-Local.git
cd IBF-Comic-Translator-Local
```

### 2. Системные пакеты

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

### 3. Виртуальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Python-зависимости

Из папки `server/`:

```bash
cd server
```

**GPU:**
```bash
bash install-linux-gpu.sh
```

**CPU:**
```bash
pip install -U pip setuptools wheel
pip install -r requirements-cpu.txt
```

### 5. llama-cpp-python

Пакет нужно установить отдельно — сборка зависит от железа.

**GPU (CUDA):**

Сначала узнайте архитектуру GPU:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Пример: NVIDIA GeForce RTX 3090, 8.6  →  86
#          NVIDIA GeForce RTX 4090, 8.9  →  89
#          NVIDIA GeForce RTX 2080, 7.5  →  75
```

Затем установите с явным указанием архитектуры:
```bash
# Замените 86 на вашу архитектуру из вывода выше
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
    pip install llama-cpp-python --no-cache-dir
```

**CPU:**
```bash
pip install llama-cpp-python
```

### 6. Скачать модель перевода

```bash
bash download-model.sh
```

Скрипт скачает `Hy-MT2-1.8B-Q8_0.gguf` (~1.9 GB) в папку `server/models/`.

Если нужна более лёгкая версия:
```bash
bash download-model.sh Hy-MT2-1.8B-Q4_K_M.gguf   # ~1.1 GB
bash download-model.sh Hy-MT2-1.8B-Q2_K.gguf      # ~0.7 GB
```

---

## Запуск сервера

```bash
cd server
source ../.venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

При первом запуске сервер скачает модели PaddleOCR (~200 MB). Это займёт 1–2 минуты.

Проверьте что сервер запустился:
```bash
curl http://127.0.0.1:8000/health
# {"ok": true}
```

В логах должно быть:
```
[STARTUP] Translator warm-up complete
[STARTUP] PaddleOCR en ready
```

---

## Установка расширения Chrome

1. Откройте `chrome://extensions/`
2. Включите **Developer mode** (верхний правый угол)
3. Нажмите **Load unpacked**
4. Выберите папку `extension/`

После загрузки:
- Откройте popup расширения
- Убедитесь что расширение включено
- При необходимости выберите язык OCR и язык перевода

> Для WSL2: сервер доступен из Chrome на Windows по `http://127.0.0.1:8000`

---

## Как это работает

1. Расширение захватывает изображение из браузера
2. Отправляет на `POST /translate-upload`
3. Сервер: OCR → группировка блоков → перевод → inpaint → рендер
4. Расширение подставляет переведённую версию на страницу

---

## Настройки расширения

В popup доступны:
- Включить / выключить расширение
- `replace` — заменить исходное изображение переведённым
- `overlay` — показать перевод поверх оригинала
- Язык OCR (en, ja, ko, zh и др.)
- Язык перевода

---

## Диагностика

**Проверка torch:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Проверка PaddleOCR:**
```bash
python -c "from app.ocr import get_paddleocr_engine; ocr = get_paddleocr_engine('en'); print('OK')"
```

**Проверка llama-cpp-python с GPU:**
```bash
python -c "from llama_cpp import Llama; print('OK')"
```
При GPU-сборке в логах при загрузке модели должно быть `ggml_cuda_init: found N CUDA devices`.

**OCR не находит текст:**
- Смотрите `debug_dir` в meta-ответе сервера
- Откройте `grouped_blocks.png` в папке debug
- Проверьте `crop_*.png` для каждого блока

---

## Опционально: PatchMatch inpaint

PatchMatch улучшает качество стирания текста на сложных фонах.

```bash
sudo apt install -y build-essential git pkg-config libopencv-dev
cd server
bash install-patchmatch.sh
export COMIC_TRANSLATOR_INPAINT_BACKEND=patchmatch
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Если PatchMatch недоступен — сервер автоматически переключается на TELEA.

---

## Структура проекта

```
extension/          Chrome-расширение
server/
  app/              Основной код (OCR, перевод, рендер)
  models/           GGUF-модели (скачиваются через download-model.sh)
  results/          Результаты переводов
README.md
```

---

## Ограничения

- Скорость перевода зависит от GPU. На CPU 1.8B GGUF даёт ~15–25 tok/s (~2–8с на страницу)
- Сложные страницы манги могут давать шумные OCR-блоки
- Некоторые сайты блокируют прямой доступ к изображениям
- CPU-режим работает, но заметно медленнее GPU

---

## License

MIT
