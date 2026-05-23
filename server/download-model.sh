#!/bin/bash
# Скачивает GGUF модель Hy-MT2-1.8B
# Варианты:
#   Q8_0  — лучшее качество  (~1.9 GB)  ← по умолчанию
#   Q4_K_M — баланс           (~1.1 GB)
#   Q2_K  — самый лёгкий     (~0.7 GB)

set -e
MODEL_FILE="${1:-Hy-MT2-1.8B-Q8_0.gguf}"
MODELS_DIR="$(dirname "$0")/models"

mkdir -p "$MODELS_DIR"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "[MODEL] Already downloaded: $MODELS_DIR/$MODEL_FILE"
    exit 0
fi

echo "[MODEL] Downloading $MODEL_FILE to $MODELS_DIR..."

# Пробуем huggingface-cli (предпочтительно)
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download tencent/Hy-MT2-1.8B-GGUF \
        --include "$MODEL_FILE" \
        --local-dir "$MODELS_DIR"
else
    # Fallback: wget
    echo "[MODEL] huggingface-cli not found, trying wget..."
    wget -c \
        "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/$MODEL_FILE" \
        -O "$MODELS_DIR/$MODEL_FILE"
fi

echo "[MODEL] Done: $MODELS_DIR/$MODEL_FILE ($(du -sh "$MODELS_DIR/$MODEL_FILE" | cut -f1))"
