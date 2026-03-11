from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import easyocr

from .models import TextBlock
from .utils import looks_like_meaningful_text


@lru_cache(maxsize=16)
def get_easyocr_reader(source_lang: str) -> easyocr.Reader:
    gpu_arg = "cuda:0" if torch.cuda.is_available() else False
    print(f"[OCR] EasyOCR lang={source_lang}, torch.cuda.is_available() = {torch.cuda.is_available()}")

    lang_list = [source_lang]
    if source_lang != "en":
        lang_list.append("en")

    return easyocr.Reader(lang_list, gpu=gpu_arg, verbose=False)


@lru_cache(maxsize=1)
def get_manga_ocr():
    from manga_ocr import MangaOcr

    force_cpu = not torch.cuda.is_available()
    return MangaOcr(force_cpu=force_cpu)


def _bbox_to_points(bbox) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in bbox:
        x = int(round(point[0]))
        y = int(round(point[1]))
        points.append((x, y))
    return points


def recognize_blocks(
    image: Image.Image,
    debug_dir: Path | None = None,
    source_ocr_lang: str = "en",
) -> list[TextBlock]:
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if source_ocr_lang == "ja":
        try:
            ocr = get_manga_ocr()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Для OCR японского текста не установлен пакет manga-ocr. "
                "Установи его командой: python -m pip install manga-ocr"
            ) from exc

        np_image = np.array(image)
        h, w = np_image.shape[:2]

        text = (ocr(image) or "").strip()
        print(f"[OCR] mangaocr_full: {text!r}")

        if not looks_like_meaningful_text(text):
            return []

        if debug_dir:
            image.save(debug_dir / "crop_001.png")

        return [
            TextBlock(
                box=[(0, 0), (w, 0), (w, h), (0, h)],
                source_text=text,
            )
        ]

    reader = get_easyocr_reader(source_ocr_lang)
    np_image = np.array(image)
    results = reader.readtext(np_image, detail=1, paragraph=False)

    blocks: list[TextBlock] = []

    for idx, result in enumerate(results, start=1):
        bbox, text, conf = result
        text = (text or "").strip()

        print(f"[OCR] easyocr_{idx:03d}: lang={source_ocr_lang} conf={conf:.3f} text={text!r}")

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