from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import TextBlock
from .utils import clamp, first_existing_path

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def build_inpaint_mask(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for block in blocks:
        x1, y1, x2, y2 = block.bounds
        pad_x = max(8, (x2 - x1) // 16)
        pad_y = max(8, (y2 - y1) // 16)
        cv2.rectangle(
            mask,
            (max(0, x1 - pad_x), max(0, y1 - pad_y)),
            (min(image.shape[1] - 1, x2 + pad_x), min(image.shape[0] - 1, y2 + pad_y)),
            255,
            thickness=-1,
        )
    return mask


def inpaint_text(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    mask = build_inpaint_mask(image, blocks)
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = first_existing_path(FONT_CANDIDATES)
    if not font_path:
        return ImageFont.load_default()
    return ImageFont.truetype(font_path, size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> str:
    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.multiline_textbbox((0, 0), trial, font=font, spacing=4)
        trial_width = bbox[2] - bbox[0]
        if trial_width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_height: int):
    for size in range(min(42, max_height), 11, -2):
        font = _load_font(size)
        wrapped = _wrap_text(draw, text, max_width, font)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4, align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font, wrapped, width, height
    font = _load_font(12)
    wrapped = _wrap_text(draw, text, max_width, font)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4, align="center")
    return font, wrapped, bbox[2] - bbox[0], bbox[3] - bbox[1]


def render_translations(cleaned_image: np.ndarray, blocks: Iterable[TextBlock]) -> Image.Image:
    image = Image.fromarray(cleaned_image)
    draw = ImageDraw.Draw(image)

    gray = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2GRAY)

    for block in blocks:
        if not block.translated_text:
            continue

        x1, y1, x2, y2 = block.bounds
        pad_x = max(6, (x2 - x1) // 12)
        pad_y = max(6, (y2 - y1) // 12)
        bx1 = clamp(x1 + pad_x // 2, 0, image.width - 1)
        by1 = clamp(y1 + pad_y // 2, 0, image.height - 1)
        bx2 = clamp(x2 - pad_x // 2, bx1 + 1, image.width)
        by2 = clamp(y2 - pad_y // 2, by1 + 1, image.height)

        roi = gray[by1:by2, bx1:bx2]
        mean_luma = int(np.mean(roi)) if roi.size else 255
        fill = (0, 0, 0) if mean_luma >= 140 else (255, 255, 255)
        stroke_fill = (255, 255, 255) if fill == (0, 0, 0) else (0, 0, 0)

        font, wrapped, text_w, text_h = _fit_text(
            draw,
            block.translated_text,
            max_width=max(20, bx2 - bx1),
            max_height=max(20, by2 - by1),
        )
        tx = bx1 + ((bx2 - bx1) - text_w) / 2
        ty = by1 + ((by2 - by1) - text_h) / 2

        draw.multiline_text(
            (tx, ty),
            wrapped,
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=stroke_fill,
            spacing=4,
            align="center",
        )

    return image