from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterable
import warnings

import cv2
import numpy as np
from PIL import Image

from .models import TextBlock
from .utils import looks_like_meaningful_text

warnings.filterwarnings(
    "ignore",
    message=r"No ccache found\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    module=r"requests(\..*)?"
)

PADDLE_LANG_MAP = {
    "ko": "korean",
    "ja": "japan",
    "ch_sim": "ch",
    "ch_tra": "chinese_cht",
}


def _normalize_paddle_lang(source_lang: str) -> str:
    normalized = (source_lang or "en").strip().lower()
    return PADDLE_LANG_MAP.get(normalized, normalized)


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"[OCR] CUDA check failed: {exc}")
        return False


def _pick_device() -> str:
    """Возвращает 'gpu:0' если CUDA доступна, иначе 'cpu'."""
    return "gpu:0" if _torch_cuda_available() else "cpu"


def _pick_engine() -> str:
    """Всегда используем Transformers backend.

    PaddleOCR 3.5.0 поддерживает два inference backend:
      - 'paddle'       — PaddlePaddle (старый, тяжёлый)
      - 'transformers' — HuggingFace Transformers (новый, без PaddlePaddle)

    Фиксируем transformers — не зависит от наличия PaddlePaddle.
    """
    return "transformers"


@lru_cache(maxsize=16)
def get_paddleocr_engine(source_lang: str):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR не установлен. Установи: pip install 'paddleocr>=3.5.0' paddlex"
        ) from exc

    # Подавляем лишние логи paddlex
    try:
        from paddlex.utils.logging import setup_logging as _setup
        _setup("WARNING")
    except Exception:
        pass

    paddle_lang = _normalize_paddle_lang(source_lang)
    device = _pick_device()
    engine = _pick_engine()

    print(
        f"[OCR] PaddleOCR source_lang={source_lang} paddle_lang={paddle_lang} "
        f"device={device} engine={engine}"
    )

    # При использовании Transformers backend CPU — убираем device='gpu:0',
    # так как routing на GPU делает сам PyTorch через .to(device)
    init_kwargs: dict[str, Any] = {
        "lang": paddle_lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "device": device,
        "engine": engine,
        "precision": "fp16" if device == "gpu:0" else "fp32",
    }

    # mkldnn конфликтует с Transformers backend
    if device == "cpu" and engine == "paddle":
        init_kwargs["enable_mkldnn"] = False

    # Пробуем с выбранным device, при ошибке — CPU fallback
    try:
        return PaddleOCR(**init_kwargs)
    except Exception as exc:
        print(f"[OCR] Init failed with device={device}: {exc}")
        if device != "cpu":
            print("[OCR] Falling back to CPU")
            init_kwargs["device"] = "cpu"
            if engine == "paddle":
                init_kwargs["enable_mkldnn"] = False
            return PaddleOCR(**init_kwargs)
        raise


def _bbox_to_points(bbox: Iterable[Iterable[Any]]) -> list[tuple[int, int]]:
    return [
        (int(round(float(p[0]))), int(round(float(p[1]))))
        for p in bbox
    ]


def _is_paddle_line(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    bbox, rec = item
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    if not isinstance(rec, (list, tuple)) or len(rec) < 2:
        return False
    return True


def _iter_paddle_lines(result: Any) -> Iterable[tuple[Any, Any]]:
    """Универсальный парсер результатов PaddleOCR 3.x.

    Поддерживает оба формата ответа:
    - dict с ключами rec_texts/rec_scores/rec_polys (новый API 3.x)
    - list[list[[bbox, [text, conf]]]]               (старый API)
    """
    if result is None:
        return

    # Новый dict-формат (paddleocr >= 3.0, predict())
    if hasattr(result, "get"):
        texts  = result.get("rec_texts")
        scores = result.get("rec_scores")
        polys  = result.get("rec_polys") or result.get("dt_polys")

        if texts is not None and scores is not None and polys is not None:
            for bbox, text, conf in zip(polys, texts, scores):
                yield bbox, (text, conf)
            return

    # Старый list-формат или одна строка
    if _is_paddle_line(result):
        yield result[0], result[1]
        return

    if isinstance(result, (list, tuple)):
        for item in result:
            yield from _iter_paddle_lines(item)


def _prepare_image(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def recognize_blocks(
    image: Image.Image,
    debug_dir: Path | None = None,
    source_ocr_lang: str = "en",
) -> list[TextBlock]:
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    ocr = get_paddleocr_engine(source_ocr_lang)
    np_image = _prepare_image(image)

    # PaddleOCR 3.x использует predict(), старые версии — ocr()
    if hasattr(ocr, "predict"):
        results = ocr.predict(np_image)
    else:
        results = ocr.ocr(np_image, cls=True)

    blocks: list[TextBlock] = []

    for idx, (bbox, rec) in enumerate(_iter_paddle_lines(results), start=1):
        text = str(rec[0] or "").strip()
        try:
            conf = float(rec[1])
        except (TypeError, ValueError):
            conf = 0.0

        print(f"[OCR] paddle_{idx:03d}: lang={source_ocr_lang} conf={conf:.3f} text={text!r}")

        if conf < 0.15:
            continue
        if not looks_like_meaningful_text(text):
            continue

        box = _bbox_to_points(bbox)

        if debug_dir:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(image.width, max(xs)), min(image.height, max(ys))
            if x2 > x1 and y2 > y1:
                image.crop((x1, y1, x2, y2)).save(debug_dir / f"crop_{idx:03d}.png")

        blocks.append(TextBlock(box=box, source_text=text))

    blocks.sort(key=lambda b: (b.bounds[1], b.bounds[0]))
    return blocks
