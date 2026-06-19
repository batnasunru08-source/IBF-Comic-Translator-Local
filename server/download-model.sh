#!/bin/bash
# Скачивает GGUF модель Hy-MT2.
#
# Использование:
#   bash download-model.sh                          # 1.8B Q8_0   (default, ~1.9 GB)
#   bash download-model.sh 7b                       # 7B   Q8_0   (~8.0 GB)
#   bash download-model.sh 1.8b q4_k_m              # 1.8B Q4_K_M (~1.1 GB)
#   bash download-model.sh 7b   q4_k_m              # 7B   Q4_K_M (~4.6 GB)
#   bash download-model.sh Hy-MT2-1.8B-Q4_K_M.gguf  # обратная совместимость: явный файл
#
# Доступные кванты:
#   1.8B: Q8_0 (1.9 GB), Q4_K_M (1.1 GB), Q2_K (0.7 GB)
#   7B:   Q8_0 (8.0 GB), Q6_K (6.2 GB), Q4_K_M (4.6 GB)
#
# Запуск скачанной модели:
#   python -m uvicorn main:app --host 0.0.0.0 --port 8000                       # 1.8B
#   MODEL_FILE=HY-MT2-7B-Q8_0.gguf python -m uvicorn main:app --host ... --port ...  # 7B

set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

first_arg="${1:-}"

if [[ "$first_arg" == *.gguf ]]; then
    MODEL_FILE="$first_arg"
    case "$MODEL_FILE" in
        HY-MT2-7B-*)   HF_REPO="tencent/Hy-MT2-7B-GGUF"   ;;
        Hy-MT2-1.8B-*) HF_REPO="tencent/Hy-MT2-1.8B-GGUF" ;;
        *)             HF_REPO="tencent/Hy-MT2-1.8B-GGUF" ;;
    esac
else
    VARIANT="${first_arg:-1.8b}"
    case "$VARIANT" in
        1.8b)
            HF_REPO="tencent/Hy-MT2-1.8B-GGUF"
            PREFIX="Hy-MT2-1.8B"
            DEFAULT_QUANT="Q8_0"
            ;;
        7b)
            HF_REPO="tencent/Hy-MT2-7B-GGUF"
            PREFIX="HY-MT2-7B"
            DEFAULT_QUANT="Q8_0"
            ;;
        *)
            echo "[MODEL] Unknown variant: $VARIANT (use 1.8b or 7b)" >&2
            exit 1
            ;;
    esac

    QUANT="${2:-$DEFAULT_QUANT}"
    MODEL_FILE="${PREFIX}-${QUANT}.gguf"
fi

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "[MODEL] Already downloaded: $MODELS_DIR/$MODEL_FILE"
    exit 0
fi

echo "[MODEL] Downloading $MODEL_FILE from $HF_REPO to $MODELS_DIR..."

if command -v hf &> /dev/null; then
    hf download "$HF_REPO" \
        --include "$MODEL_FILE" \
        --local-dir "$MODELS_DIR"
elif command -v huggingface-cli &> /dev/null; then
    echo "[MODEL] Note: huggingface-cli is deprecated, prefer 'hf download'"
    huggingface-cli download "$HF_REPO" \
        --include "$MODEL_FILE" \
        --local-dir "$MODELS_DIR"
else
    echo "[MODEL] hf / huggingface-cli not found, trying wget..."
    wget -c \
        "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE" \
        -O "$MODELS_DIR/$MODEL_FILE"
fi

echo "[MODEL] Done: $MODELS_DIR/$MODEL_FILE ($(du -sh "$MODELS_DIR/$MODEL_FILE" | cut -f1))"
