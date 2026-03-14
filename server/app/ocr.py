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
    module=r"requests(\..*)?",
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
    except Exception as exc:
        print(f"[OCR] torch import failed while checking CUDA availability: {exc}")
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"[OCR] torch CUDA check failed: {exc}")
        return False


def _can_use_paddle_gpu() -> bool:
    if not _torch_cuda_available():
        return False

    try:
        import paddle
    except ModuleNotFoundError:
        return False

    return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0


@lru_cache(maxsize=16)
def get_paddleocr_engine(source_lang: str):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR
        from paddlex.utils.logging import setup_logging as setup_paddlex_logging
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR не установлен. Установи зависимости командами: "
            "python -m pip install paddleocr и подходящую сборку paddlepaddle "
            "или paddlepaddle-gpu для своей платформы."
        ) from exc

    setup_paddlex_logging("WARNING")

    paddle_lang = _normalize_paddle_lang(source_lang)
    prefer_gpu = _can_use_paddle_gpu()
    print(
        f"[OCR] PaddleOCR source_lang={source_lang} paddle_lang={paddle_lang} "
        f"paddle_gpu_available={prefer_gpu}"
    )

    init_error: Exception | None = None
    devices = ["gpu:0", "cpu"] if prefer_gpu else ["cpu"]
    for device in devices:
        try:
            init_kwargs: dict[str, Any] = {
                "lang": paddle_lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "device": device,
            }
            if device == "cpu":
                init_kwargs["enable_mkldnn"] = False

            return PaddleOCR(
                **init_kwargs,
            )
        except Exception as exc:
            init_error = exc
            print(f"[OCR] PaddleOCR init failed with device={device}: {exc}")

    raise RuntimeError(f"Не удалось инициализировать PaddleOCR: {init_error}") from init_error


def _bbox_to_points(bbox: Iterable[Iterable[Any]]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in bbox:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        points.append((x, y))
    return points


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
    if result is None:
        return

    if hasattr(result, "get"):
        texts = result.get("rec_texts")
        scores = result.get("rec_scores")
        polys = result.get("rec_polys")
        if polys is None or len(polys) == 0:
            polys = result.get("dt_polys")

        if texts is not None and scores is not None and polys is not None:
            for bbox, text, conf in zip(polys, texts, scores):
                yield bbox, (text, conf)
            return

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
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(image.width, max(xs))
            y2 = min(image.height, max(ys))
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crop.save(debug_dir / f"crop_{idx:03d}.png")

        blocks.append(
            TextBlock(
                box=box,
                source_text=text,
            )
        )

    blocks.sort(key=lambda b: (b.bounds[1], b.bounds[0]))
    return blocks
