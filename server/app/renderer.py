from __future__ import annotations

from functools import lru_cache
import re
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import TextBlock
from .utils import clamp, first_existing_path

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30FF]")
HANGUL_RE = re.compile(r"[\uAC00-\uD7AF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")

FONT_CANDIDATES = {
    "default": [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "cyrillic": [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "ja": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ],
    "ko": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
    ],
    "zh": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ],
}


def _font_candidates_for_text(text: str) -> list[str]:
    text = text or ""
    if HIRAGANA_KATAKANA_RE.search(text):
        return FONT_CANDIDATES["ja"] + FONT_CANDIDATES["default"]
    if HANGUL_RE.search(text):
        return FONT_CANDIDATES["ko"] + FONT_CANDIDATES["default"]
    if CJK_RE.search(text):
        return FONT_CANDIDATES["zh"] + FONT_CANDIDATES["default"]
    if CYRILLIC_RE.search(text):
        return FONT_CANDIDATES["cyrillic"] + FONT_CANDIDATES["default"]
    return FONT_CANDIDATES["default"]


@lru_cache(maxsize=256)
def _load_font_from_path(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def _load_font(
    size: int,
    text: str,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = first_existing_path(_font_candidates_for_text(text))
    if not font_path:
        return ImageFont.load_default()
    return _load_font_from_path(font_path, size=size)


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


def _xor_sum(a: np.ndarray, b: np.ndarray) -> int:
    return int(cv2.bitwise_xor(a, b).sum())


def _minxor_mask(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    candidate = np.ascontiguousarray(candidate)
    reference = np.ascontiguousarray(reference)

    inverted = cv2.bitwise_not(candidate)
    candidate_error = _xor_sum(candidate, reference)
    inverted_error = _xor_sum(inverted, reference)
    if inverted_error < candidate_error:
        return inverted, inverted_error
    return candidate, candidate_error


def _top_histogram_levels(values: np.ndarray, top_k: int = 3, min_gap: int = 10) -> list[int]:
    if values.size == 0:
        return []

    hist = np.bincount(values.astype(np.uint8), minlength=256)
    threshold = max(5, int(values.size * 0.005))

    picked: list[int] = []
    for level in np.argsort(hist)[::-1]:
        if hist[level] < threshold:
            break
        if any(abs(int(level) - prev) < min_gap for prev in picked):
            continue
        picked.append(int(level))
        if len(picked) >= top_k:
            break
    return picked


def _candidate_masks_from_gray(gray: np.ndarray, reference: np.ndarray) -> list[tuple[np.ndarray, int]]:
    eroded = cv2.erode(reference, np.ones((3, 3), np.uint8), iterations=1)
    values = gray[eroded > 0]
    masks: list[tuple[np.ndarray, int]] = []

    for level in _top_histogram_levels(values):
        low = max(0, level - 28)
        high = min(255, level + 28)
        candidate = cv2.inRange(gray, low, high)
        masks.append(_minxor_mask(candidate, reference))

    return masks


def _candidate_masks_from_otsu(crop_rgb: np.ndarray, gray: np.ndarray, reference: np.ndarray) -> list[tuple[np.ndarray, int]]:
    masks: list[tuple[np.ndarray, int]] = []

    channels = [gray, crop_rgb[:, :, 0], crop_rgb[:, :, 1], crop_rgb[:, :, 2]]
    for channel in channels:
        _, candidate = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(_minxor_mask(candidate, reference))

    return masks


def _merge_mask_candidates(
    mask_list: list[tuple[np.ndarray, int]],
    reference: np.ndarray,
) -> np.ndarray:
    if not mask_list:
        return reference.copy()

    mask_merged = np.zeros_like(reference)
    for candidate_mask, _ in sorted(mask_list, key=lambda item: item[1]):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8, cv2.CV_16U)
        for label_index in range(1, num_labels):
            x, y, w, h, area = stats[label_index]
            if area < 3:
                continue

            local = labels[y:y + h, x:x + w]
            component = np.zeros((h, w), dtype=np.uint8)
            component[local == label_index] = 255

            current = mask_merged[y:y + h, x:x + w]
            trial = cv2.bitwise_or(current, component)
            reference_local = reference[y:y + h, x:x + w]
            if _xor_sum(trial, reference_local) < _xor_sum(current, reference_local):
                mask_merged[y:y + h, x:x + w] = trial

    if np.count_nonzero(mask_merged) == 0:
        return reference.copy()
    return mask_merged


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    h, w = filled.shape[:2]
    inverse = cv2.bitwise_not(filled)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8, cv2.CV_16U)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches_border:
            continue
        if area > max(48, (h * w) // 250):
            continue
        filled[labels == label] = 255

    return filled


def _build_local_text_mask(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, mask_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_light = cv2.threshold(255 - blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rough_mask = cv2.bitwise_or(mask_dark, mask_light)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    rough_mask = _filter_components(rough_mask)

    rough_density = float(np.count_nonzero(rough_mask)) / float(max(1, rough_mask.size))
    if rough_density < 0.003 or rough_density > 0.55:
        return np.zeros_like(rough_mask)

    candidates = _candidate_masks_from_gray(gray, rough_mask)
    candidates.extend(_candidate_masks_from_otsu(crop_rgb, gray, rough_mask))

    refined = _merge_mask_candidates(candidates, rough_mask)
    refined = _filter_components(refined)
    refined = _fill_small_holes(refined)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    refined = cv2.dilate(refined, kernel_close, iterations=1)

    refined_density = float(np.count_nonzero(refined)) / float(max(1, refined.size))
    if refined_density < 0.002 or refined_density > 0.45:
        return rough_mask

    if np.count_nonzero(refined) < np.count_nonzero(rough_mask) * 0.2:
        return rough_mask

    return refined


def build_inpaint_mask(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        x1, y1, x2, y2 = block.bounds
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = max(8, bw // 10)
        pad_y = max(8, bh // 10)

        rx1 = clamp(x1 - pad_x, 0, w - 1)
        ry1 = clamp(y1 - pad_y, 0, h - 1)
        rx2 = clamp(x2 + pad_x, rx1 + 1, w)
        ry2 = clamp(y2 + pad_y, ry1 + 1, h)

        crop = image[ry1:ry2, rx1:rx2]
        local_mask = _build_local_text_mask(crop)

        if np.count_nonzero(local_mask) == 0:
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            local_mask = cv2.dilate(fallback, kernel, iterations=1)

        full_mask[ry1:ry2, rx1:rx2] = cv2.bitwise_or(full_mask[ry1:ry2, rx1:rx2], local_mask)

    return full_mask


def inpaint_text(image: np.ndarray, blocks: Iterable[TextBlock]) -> np.ndarray:
    mask = build_inpaint_mask(image, blocks)
    if np.count_nonzero(mask) == 0:
        return image.copy()

    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


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
        return "\n".join(_split_long_token(draw, text, font, max_width))

    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        trial = word if not current else f"{current} {word}"
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
            continue

        split_parts = _split_long_token(draw, word, font, max_width)
        lines.extend(split_parts[:-1])
        current = split_parts[-1] if split_parts else word

    if current:
        lines.append(current)

    return "\n".join(lines)


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
):
    start_size = max(14, min(42, int(max_height * 0.78)))

    for size in range(start_size, 11, -1):
        font = _load_font(size, text)
        wrapped = _wrap_text(draw, text, max_width, font)
        spacing = max(4, int(round(size * 0.25)))
        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped,
            font=font,
            spacing=spacing,
            align="center",
        )
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width <= max_width and height <= max_height:
            return font, wrapped, width, height

    font = _load_font(12, text)
    wrapped = _wrap_text(draw, text, max_width, font)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4, align="center")
    return font, wrapped, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _luma(color: tuple[int, int, int]) -> float:
    r, g, b = color
    return float(0.299 * r + 0.587 * g + 0.114 * b)


def _median_color(
    pixels: np.ndarray,
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    if pixels.size == 0:
        return fallback
    return tuple(int(round(v)) for v in np.median(pixels, axis=0))


def _canonicalize_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    if _luma(color) <= 20:
        return (0, 0, 0)
    if _luma(color) >= 235:
        return (255, 255, 255)
    return color


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(av - bv))


def _fallback_text_colors(roi_rgb: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    bg = _median_color(roi_rgb.reshape(-1, 3), (255, 255, 255))
    if _luma(bg) >= 150:
        return (0, 0, 0), (255, 255, 255), bg
    return (255, 255, 255), (0, 0, 0), bg


def _pick_text_style(roi_rgb: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    mask = _build_local_text_mask(roi_rgb)
    if np.count_nonzero(mask) == 0:
        return _fallback_text_colors(roi_rgb)

    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

    text_pixels = roi_rgb[mask > 0]
    bg_pixels = roi_rgb[ring > 0]
    if bg_pixels.size == 0:
        bg_pixels = roi_rgb[mask == 0]

    if text_pixels.size == 0 or bg_pixels.size == 0:
        return _fallback_text_colors(roi_rgb)

    bg = _canonicalize_color(_median_color(bg_pixels, (255, 255, 255)))
    text_luma = 0.299 * text_pixels[:, 0] + 0.587 * text_pixels[:, 1] + 0.114 * text_pixels[:, 2]
    bg_luma = _luma(bg)

    if float(np.mean(text_luma)) <= bg_luma:
        cutoff = np.quantile(text_luma, 0.45)
        fill_pixels = text_pixels[text_luma <= cutoff]
    else:
        cutoff = np.quantile(text_luma, 0.55)
        fill_pixels = text_pixels[text_luma >= cutoff]

    fill = _canonicalize_color(_median_color(fill_pixels, (0, 0, 0)))
    stroke = bg

    if _color_distance(fill, stroke) < 60:
        return _fallback_text_colors(roi_rgb)

    return fill, stroke, bg


def _pick_stroke_width(
    font_size: int,
    fill: tuple[int, int, int],
    background: tuple[int, int, int],
) -> int:
    contrast = _color_distance(fill, background)
    if contrast >= 180:
        return 0
    if contrast >= 115:
        return 1 if font_size < 24 else 2
    return 2 if font_size < 28 else 3


def render_translations(
    cleaned_image: np.ndarray,
    blocks: Iterable[TextBlock],
    original_image: np.ndarray | None = None,
) -> Image.Image:
    image = Image.fromarray(cleaned_image)
    draw = ImageDraw.Draw(image)
    style_source = original_image if original_image is not None else cleaned_image

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

        style_roi = style_source[ty1:ty2, tx1:tx2]
        fill, stroke_fill, background = _pick_text_style(style_roi)

        text_x = tx1 + (max_width - text_w) / 2
        text_y = ty1 + (max_height - text_h) / 2

        font_size = getattr(font, "size", 14)
        spacing = max(4, int(round(font_size * 0.25)))
        stroke_width = _pick_stroke_width(font_size, fill, background)

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
