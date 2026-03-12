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
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = first_existing_path(FONT_CANDIDATES)
    if not font_path:
        return ImageFont.load_default()
    return ImageFont.truetype(font_path, size=size)


def _filter_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape[:2]
    crop_area = h * w
    filtered = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < max(8, crop_area // 3000):
            continue
        if area > crop_area * 0.35:
            continue
        if bw < 2 or bh < 2:
            continue

        aspect = bw / max(1, bh)
        if aspect > 25 or aspect < 0.03:
            continue

        filtered[labels == label] = 255

    return filtered


def _build_local_text_mask(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Тёмный текст на светлом фоне
    _, mask_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Светлый текст на тёмном фоне
    _, mask_light = cv2.threshold(255 - blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(mask_dark, mask_light)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = _filter_components(mask)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    density = float(np.count_nonzero(mask)) / float(mask.size)
    if density < 0.003 or density > 0.45:
        return np.zeros_like(mask)

    return mask


def build_inpaint_mask(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        x1, y1, x2, y2 = block.bounds
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = max(6, bw // 12)
        pad_y = max(6, bh // 12)

        rx1 = clamp(x1 - pad_x, 0, w - 1)
        ry1 = clamp(y1 - pad_y, 0, h - 1)
        rx2 = clamp(x2 + pad_x, rx1 + 1, w)
        ry2 = clamp(y2 + pad_y, ry1 + 1, h)

        crop = image[ry1:ry2, rx1:rx2]
        local_mask = _build_local_text_mask(crop)

        if np.count_nonzero(local_mask) == 0:
            # fallback: мягкий прямоугольник по области текста
            fallback = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
            inner_x1 = max(0, x1 - rx1)
            inner_y1 = max(0, y1 - ry1)
            inner_x2 = min(rx2 - rx1, x2 - rx1)
            inner_y2 = min(ry2 - ry1, y2 - ry1)

            cv2.rectangle(
                fallback,
                (inner_x1, inner_y1),
                (max(inner_x1 + 1, inner_x2), max(inner_y1 + 1, inner_y2)),
                255,
                thickness=-1,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            local_mask = cv2.dilate(fallback, kernel, iterations=1)

        full_mask[ry1:ry2, rx1:rx2] = cv2.bitwise_or(full_mask[ry1:ry2, rx1:rx2], local_mask)

    return full_mask


def inpaint_text(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    mask = build_inpaint_mask(image, blocks)
    if np.count_nonzero(mask) == 0:
        return image.copy()

    # Для текста обычно TELEA выглядит лучше и быстрее
    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def _split_long_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not token:
        return [""]

    parts: list[str] = []
    current = ""

    for ch in token:
        trial = current + ch
        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]
        if current and width > max_width:
            parts.append(current)
            current = ch
        else:
            current = trial

    if current:
        parts.append(current)

    return parts


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if " " not in text:
        lines = _split_long_token(draw, text, font, max_width)
        return "\n".join(lines)

    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        if not current:
            trial = word
        else:
            trial = f"{current} {word}"

        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = trial
            continue

        if current:
            lines.append(current)

        word_bbox = draw.textbbox((0, 0), word, font=font)
        word_width = word_bbox[2] - word_bbox[0]

        if word_width <= max_width:
            current = word
        else:
            split_parts = _split_long_token(draw, word, font, max_width)
            if split_parts:
                lines.extend(split_parts[:-1])
                current = split_parts[-1]
            else:
                current = word

    if current:
        lines.append(current)

    return "\n".join(lines)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
):
    start_size = max(14, min(42, int(max_height * 0.75)))

    for size in range(start_size, 11, -1):
        font = _load_font(size)
        wrapped = _wrap_text(draw, text, max_width, font)
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped,
            font=font,
            spacing=max(3, size // 5),
            align="center",
        )
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width <= max_width and height <= max_height:
            return font, wrapped, width, height

    font = _load_font(12)
    wrapped = _wrap_text(draw, text, max_width, font)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=3, align="center")
    return font, wrapped, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _pick_text_colors(roi_gray: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    mean_luma = int(np.mean(roi_gray)) if roi_gray.size else 255

    if mean_luma >= 150:
        # светлый фон -> чёрный текст, белая обводка
        return (0, 0, 0), (255, 255, 255)

    # тёмный фон -> белый текст, чёрная обводка
    return (255, 255, 255), (0, 0, 0)


def render_translations(cleaned_image: np.ndarray, blocks: Iterable[TextBlock]) -> Image.Image:
    image = Image.fromarray(cleaned_image)
    draw = ImageDraw.Draw(image)
    gray = cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2GRAY)

    for block in blocks:
        if not block.translated_text:
            continue

        x1, y1, x2, y2 = block.bounds
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = max(8, bw // 10)
        pad_y = max(8, bh // 10)

        tx1 = clamp(x1 + pad_x // 2, 0, image.width - 1)
        ty1 = clamp(y1 + pad_y // 2, 0, image.height - 1)
        tx2 = clamp(x2 - pad_x // 2, tx1 + 1, image.width)
        ty2 = clamp(y2 - pad_y // 2, ty1 + 1, image.height)

        max_width = max(24, tx2 - tx1)
        max_height = max(24, ty2 - ty1)

        font, wrapped, text_w, text_h = _fit_text(
            draw,
            block.translated_text,
            max_width=max_width,
            max_height=max_height,
        )

        roi_gray = gray[ty1:ty2, tx1:tx2]
        fill, stroke_fill = _pick_text_colors(roi_gray)

        text_x = tx1 + (max_width - text_w) / 2
        text_y = ty1 + (max_height - text_h) / 2

        font_size = getattr(font, "size", 14)
        spacing = max(3, font_size // 5)
        stroke_width = 2 if font_size >= 16 else 1

        draw.multiline_text(
            (text_x, text_y),
            wrapped,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            spacing=spacing,
            align="center",
        )

    return image