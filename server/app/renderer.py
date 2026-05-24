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

# ---------------------------------------------------------------------------
# Шрифты: Bold-варианты для комиксов (приоритет 🟢)
# ---------------------------------------------------------------------------
FONT_CANDIDATES = {
    "default": [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "cyrillic": [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "ja": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/YuGothB.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ],
    "ko": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ],
    "zh": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
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


def _lang_key(text: str) -> str:
    """Возвращает ключ языка для кеша пути к шрифту."""
    text = text or ""
    if HIRAGANA_KATAKANA_RE.search(text):
        return "ja"
    if HANGUL_RE.search(text):
        return "ko"
    if CJK_RE.search(text):
        return "zh"
    if CYRILLIC_RE.search(text):
        return "cyrillic"
    return "default"


@lru_cache(maxsize=8)
def _resolve_font_path(lang_key: str) -> str | None:
    """Кешируем результат поиска шрифта на диске по ключу языка.

    first_existing_path делает Path.exists() на каждый вызов —
    с кешем это происходит один раз на язык за весь процесс.
    """
    candidates = FONT_CANDIDATES.get(lang_key, FONT_CANDIDATES["default"])
    if lang_key != "default":
        candidates = candidates + FONT_CANDIDATES["default"]
    return first_existing_path(candidates)


@lru_cache(maxsize=256)
def _load_font_from_path(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def _load_font(size: int, text: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _resolve_font_path(_lang_key(text))
    if not font_path:
        return ImageFont.load_default()
    return _load_font_from_path(font_path, size=size)


# ---------------------------------------------------------------------------
# Маска текста
# ---------------------------------------------------------------------------

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


def _merge_mask_candidates(mask_list: list[tuple[np.ndarray, int]], reference: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Inpaint: PatchMatch + TELEA fallback + кеш масок (приоритеты 🔴🟡)
# ---------------------------------------------------------------------------

def build_inpaint_mask(
    image: np.ndarray,
    blocks: list[TextBlock],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Возвращает (full_mask, block_masks).

    block_masks[idx] — маска в координатах полного изображения для блока idx.
    Передаётся в render_translations, чтобы не пересчитывать маску повторно.
    """
    h, w = image.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    block_masks: dict[int, np.ndarray] = {}

    for idx, block in enumerate(blocks):
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

        block_mask_full = np.zeros((h, w), dtype=np.uint8)
        block_mask_full[ry1:ry2, rx1:rx2] = local_mask
        block_masks[idx] = block_mask_full

    return full_mask, block_masks


def inpaint_text(
    image: np.ndarray,
    blocks: Iterable[TextBlock],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Стирает текст по каждому блоку отдельно (crop-inpaint).

    Всегда используем TELEA на crop-ах — это быстро (десятки мс на блок)
    и качественно для небольших областей. PatchMatch на CPU слишком
    медленный даже на маленьких crop-ах (~10-60с на блок).
    """
    from time import perf_counter

    blocks = list(blocks)
    full_mask, block_masks = build_inpaint_mask(image, blocks)

    if np.count_nonzero(full_mask) == 0:
        return image.copy(), block_masks

    cleaned = image.copy()
    h, w = image.shape[:2]

    for i, block in enumerate(blocks):
        x1, y1, x2, y2 = block.bounds
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(8, bw // 10)
        pad_y = max(8, bh // 10)

        rx1 = clamp(x1 - pad_x, 0, w - 1)
        ry1 = clamp(y1 - pad_y, 0, h - 1)
        rx2 = clamp(x2 + pad_x, rx1 + 1, w)
        ry2 = clamp(y2 + pad_y, ry1 + 1, h)

        crop_mask = full_mask[ry1:ry2, rx1:rx2]
        if np.count_nonzero(crop_mask) == 0:
            continue

        crop = np.ascontiguousarray(cleaned[ry1:ry2, rx1:rx2])
        t0 = perf_counter()
        result = cv2.inpaint(crop, crop_mask, 3, cv2.INPAINT_TELEA)
        ms = round((perf_counter() - t0) * 1000)
        print(f"[INPAINT] block[{i+1}] crop=({rx2-rx1}x{ry2-ry1}) mask_px={np.count_nonzero(crop_mask)} telea={ms}ms")
        cleaned[ry1:ry2, rx1:rx2] = result

    return cleaned, block_masks


# ---------------------------------------------------------------------------
# Текстовый движок
# ---------------------------------------------------------------------------

def _split_long_token(draw, token, font, max_width):
    if not token:
        return [""]
    parts = []
    current = ""
    for ch in token:
        trial = current + ch
        bbox = draw.textbbox((0, 0), trial, font=font)
        if current and bbox[2] - bbox[0] > max_width:
            parts.append(current)
            current = ch
        else:
            current = trial
    if current:
        parts.append(current)
    return parts


def _wrap_text(draw, text, max_width, font):
    text = (text or "").strip()
    if not text:
        return ""
    if " " not in text:
        return "\n".join(_split_long_token(draw, text, font, max_width))
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        word_bbox = draw.textbbox((0, 0), word, font=font)
        if word_bbox[2] - word_bbox[0] <= max_width:
            current = word
            continue
        split_parts = _split_long_token(draw, word, font, max_width)
        lines.extend(split_parts[:-1])
        current = split_parts[-1] if split_parts else word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _fit_text(draw, text, max_width, max_height):
    """Подбирает максимальный размер шрифта бинарным поиском.

    Линейный перебор делал ~30 итераций; бинарный — ~5.
    Но wrap зависит от размера шрифта, поэтому после нахождения
    кандидата проверяем соседей вверх чтобы не пропустить лучший fit.
    """
    lo = 11
    hi = max(14, min(42, int(max_height * 0.78)))

    best: tuple | None = None

    # Бинарный поиск: ищем наибольший размер при котором текст влезает
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(mid, text)
        wrapped = _wrap_text(draw, text, max_width, font)
        spacing = max(4, int(round(mid * 0.25)))
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= max_width and h <= max_height:
            best = (font, wrapped, w, h)
            lo = mid + 1  # пробуем больше
        else:
            hi = mid - 1

    if best is not None:
        return best

    # Fallback: минимальный размер
    font = _load_font(11, text)
    wrapped = _wrap_text(draw, text, max_width, font)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4, align="center")
    return font, wrapped, bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# Цвет
# ---------------------------------------------------------------------------

def _luma(color):
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b


def _median_color(pixels, fallback):
    if pixels.size == 0:
        return fallback
    return tuple(int(round(v)) for v in np.median(pixels, axis=0))


def _canonicalize_color(color):
    if _luma(color) <= 20:
        return (0, 0, 0)
    if _luma(color) >= 235:
        return (255, 255, 255)
    return color


def _color_distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _fallback_text_colors(roi_rgb):
    bg = _median_color(roi_rgb.reshape(-1, 3), (255, 255, 255))
    if _luma(bg) >= 150:
        return (0, 0, 0), (255, 255, 255), bg
    return (255, 255, 255), (0, 0, 0), bg


def _sample_border_pixels(roi_rgb, border=4):
    """Сэмплирует пиксели по краям roi — там гарантированно фон (приоритет 🔴)."""
    h, w = roi_rgb.shape[:2]
    border = min(border, h // 3, w // 3, 4)
    if border < 1:
        return roi_rgb.reshape(-1, 3)
    top    = roi_rgb[:border,       :          ].reshape(-1, 3)
    bottom = roi_rgb[h - border:,   :          ].reshape(-1, 3)
    left   = roi_rgb[border:h-border, :border  ].reshape(-1, 3)
    right  = roi_rgb[border:h-border, w-border:].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def _pick_text_style(roi_rgb, cached_mask=None):
    """Определяет (fill, stroke_fill, background).

    cached_mask позволяет пропустить повторное вычисление маски (приоритет 🟡).
    Фон сэмплируется по краям roi из оригинала (приоритет 🔴).
    """
    mask = cached_mask if cached_mask is not None else _build_local_text_mask(roi_rgb)

    border_pixels = _sample_border_pixels(roi_rgb)
    bg_from_border = _canonicalize_color(_median_color(border_pixels, (255, 255, 255)))

    if np.count_nonzero(mask) == 0:
        bg = bg_from_border
        if _luma(bg) >= 150:
            return (0, 0, 0), (255, 255, 255), bg
        return (255, 255, 255), (0, 0, 0), bg

    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))

    text_pixels = roi_rgb[mask > 0]
    bg_pixels   = roi_rgb[ring > 0]
    if bg_pixels.size == 0:
        bg_pixels = roi_rgb[mask == 0]

    if text_pixels.size == 0 or bg_pixels.size == 0:
        return _fallback_text_colors(roi_rgb)

    bg_from_ring = _canonicalize_color(_median_color(bg_pixels, (255, 255, 255)))
    # Предпочитаем border-медиану если они согласуются (стабильнее)
    bg = bg_from_border if _color_distance(bg_from_border, bg_from_ring) < 80 else bg_from_ring

    text_luma = 0.299 * text_pixels[:, 0] + 0.587 * text_pixels[:, 1] + 0.114 * text_pixels[:, 2]
    bg_luma = _luma(bg)

    if float(np.mean(text_luma)) <= bg_luma:
        cutoff = np.quantile(text_luma, 0.45)
        fill_pixels = text_pixels[text_luma <= cutoff]
    else:
        cutoff = np.quantile(text_luma, 0.55)
        fill_pixels = text_pixels[text_luma >= cutoff]

    fill   = _canonicalize_color(_median_color(fill_pixels, (0, 0, 0)))
    stroke = bg

    if _color_distance(fill, stroke) < 60:
        return _fallback_text_colors(roi_rgb)

    return fill, stroke, bg


def _pick_stroke_width(font_size, fill, background):
    contrast = _color_distance(fill, background)
    if contrast >= 180:
        return 0
    if contrast >= 115:
        return 1 if font_size < 24 else 2
    return 2 if font_size < 28 else 3


# ---------------------------------------------------------------------------
# Наибольший вписанный прямоугольник (приоритет 🟡)
# ---------------------------------------------------------------------------

def _largest_inscribed_rect(mask: np.ndarray):
    """Наибольший вписанный прямоугольник.

    Heights обновляются векторно (NumPy), стек — один цикл по строкам.
    Для больших масок работает на уменьшенной копии (масштаб ~150px).
    Возвращает (x1, y1, x2, y2) или None.
    """
    if np.count_nonzero(mask) == 0:
        return None

    h, w = mask.shape
    scale = 1
    if max(h, w) > 300:
        scale = max(h, w) // 150
        small = mask[::scale, ::scale]
    else:
        small = mask

    sh, sw = small.shape
    heights = np.zeros(sw, dtype=np.int32)
    best_area = 0
    best_rect = None

    for row in range(sh):
        heights = np.where(small[row] > 0, heights + 1, 0)

        h_row = heights.tolist() + [0]
        stack: list[int] = []
        for col, h_cur in enumerate(h_row):
            while stack and heights[stack[-1]] > h_cur:
                ht = int(heights[stack.pop()])
                wd = col if not stack else col - stack[-1] - 1
                area = ht * wd
                if area > best_area:
                    best_area = area
                    left = col - wd if not stack else stack[-1] + 1
                    best_rect = (left, row - ht + 1, left + wd, row + 1)
            stack.append(col)

    if best_rect is None:
        return None

    rx1, ry1, rx2, ry2 = best_rect
    return (rx1 * scale, ry1 * scale, rx2 * scale, ry2 * scale)


def _bubble_text_bounds(block_mask, x1, y1, x2, y2, image_w, image_h, pad_x, pad_y):
    """Текстовая область с учётом формы облака (приоритет 🟡)."""
    if block_mask is not None:
        roi_mask = block_mask[y1:y2, x1:x2]
        if roi_mask.shape[0] > 0 and roi_mask.shape[1] > 0:
            erode_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (max(3, pad_x), max(3, pad_y))
            )
            eroded_roi = cv2.erode(roi_mask, erode_k, iterations=1)
            rect = _largest_inscribed_rect(eroded_roi)
            if rect is not None:
                rx1, ry1, rx2, ry2 = rect
                tx1 = clamp(x1 + rx1, 0, image_w - 1)
                ty1 = clamp(y1 + ry1, 0, image_h - 1)
                tx2 = clamp(x1 + rx2, tx1 + 1, image_w)
                ty2 = clamp(y1 + ry2, ty1 + 1, image_h)
                if (tx2 - tx1) >= 20 and (ty2 - ty1) >= 16:
                    return tx1, ty1, tx2, ty2

    tx1 = clamp(x1 + pad_x // 2, 0, image_w - 1)
    ty1 = clamp(y1 + pad_y // 2, 0, image_h - 1)
    tx2 = clamp(x2 - pad_x // 2, tx1 + 1, image_w)
    ty2 = clamp(y2 - pad_y // 2, ty1 + 1, image_h)
    return tx1, ty1, tx2, ty2


# ---------------------------------------------------------------------------
# Рисование текста с гало / тенью для читаемости на сложном фоне
# ---------------------------------------------------------------------------

def _draw_text_with_halo(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    text_x: float,
    text_y: float,
    wrapped: str,
    font,
    fill: tuple,
    stroke_fill: tuple,
    stroke_width: int,
    spacing: int,
    text_w: float,
    text_h: float,
    use_halo: bool = True,
) -> None:
    """Рисует текст с размытым гало для читаемости на любом фоне.

    Алгоритм:
    1. Рендерим текст на отдельном прозрачном слое
    2. Размываем его — получаем «ореол» контрастного цвета
    3. Накладываем ореол на изображение, поверх — чёткий текст

    use_halo=False — просто рисуем текст без гало (для чистых пузырей).
    """
    if not use_halo:
        draw.multiline_text(
            (text_x, text_y), wrapped, font=font, fill=fill,
            stroke_width=stroke_width, stroke_fill=stroke_fill,
            spacing=spacing, align="center",
        )
        return

    # Определяем цвет гало: противоположный яркости текста
    fill_luma = _luma(fill)
    halo_color = (0, 0, 0) if fill_luma >= 128 else (255, 255, 255)

    # Радиус размытия зависит от размера шрифта
    font_size = getattr(font, "size", 14)
    blur_radius = max(2, font_size // 5)

    # Padding вокруг текста для гало
    pad = blur_radius * 2
    layer_w = int(text_w) + pad * 2 + 2
    layer_h = int(text_h) + pad * 2 + 2

    # Слой для гало: рисуем текст гало-цветом, потом размываем
    halo_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    halo_draw  = ImageDraw.Draw(halo_layer)
    halo_draw.multiline_text(
        (pad, pad), wrapped, font=font,
        fill=(*halo_color, 255),
        stroke_width=max(1, blur_radius // 2),
        stroke_fill=(*halo_color, 255),
        spacing=spacing, align="center",
    )

    # Размываем гало
    from PIL import ImageFilter
    halo_layer = halo_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Накладываем гало на изображение
    paste_x = int(text_x) - pad
    paste_y = int(text_y) - pad
    image.paste(halo_layer, (paste_x, paste_y), mask=halo_layer)

    # Поверх гало — чёткий текст
    draw.multiline_text(
        (text_x, text_y), wrapped, font=font, fill=fill,
        stroke_width=stroke_width, stroke_fill=stroke_fill,
        spacing=spacing, align="center",
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_translations(
    cleaned_image: np.ndarray,
    blocks: Iterable[TextBlock],
    original_image: np.ndarray | None = None,
    block_masks: dict[int, np.ndarray] | None = None,
) -> Image.Image:
    """Рисует переведённый текст.

    original_image — для точного определения цвета фона из оригинала.
    block_masks    — кеш масок из inpaint_text (не пересчитываем повторно).
    """
    image = Image.fromarray(cleaned_image)
    draw  = ImageDraw.Draw(image)
    style_source = original_image if original_image is not None else cleaned_image
    iw, ih = image.width, image.height

    for idx, block in enumerate(blocks):
        if not block.translated_text:
            continue

        x1, y1, x2, y2 = block.bounds
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(8, bw // 10)
        pad_y = max(8, bh // 10)

        cached_block_mask = block_masks.get(idx) if block_masks else None

        # Текстовая область по форме облака (приоритет 🟡)
        tx1, ty1, tx2, ty2 = _bubble_text_bounds(
            cached_block_mask, x1, y1, x2, y2, iw, ih, pad_x, pad_y,
        )

        max_width  = max(24, tx2 - tx1)
        max_height = max(24, ty2 - ty1)

        font, wrapped, text_w, text_h = _fit_text(draw, block.translated_text, max_width, max_height)

        # Цвет берём из оригинала (приоритет 🔴)
        style_roi = style_source[ty1:ty2, tx1:tx2]
        local_mask_roi = cached_block_mask[ty1:ty2, tx1:tx2] if cached_block_mask is not None else None

        fill, stroke_fill, background = _pick_text_style(style_roi, cached_mask=local_mask_roi)

        text_x = tx1 + (max_width - text_w) / 2
        text_y = ty1 + (max_height - text_h) / 2

        font_size    = getattr(font, "size", 14)
        spacing      = max(4, int(round(font_size * 0.25)))
        stroke_width = _pick_stroke_width(font_size, fill, background)

        contrast = _color_distance(fill, background)
        bg_luma  = _luma(background)
        is_light_bg = bg_luma >= 200

        # Непрозрачная подложка: светлый фон → чёрный текст, тёмный фон → белый текст.
        pad_bg = max(6, font_size // 4)
        bg_x1 = int(text_x - pad_bg)
        bg_y1 = int(text_y - pad_bg)
        bg_x2 = int(text_x + text_w + pad_bg)
        bg_y2 = int(text_y + text_h + pad_bg)

        if bg_luma >= 128:
            fill = (0, 0, 0)
            bg_color = (255, 255, 255)
        else:
            fill = (255, 255, 255)
            bg_color = (0, 0, 0)

        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)

        draw.multiline_text(
            (text_x, text_y), wrapped, font=font, fill=fill,
            stroke_width=0, stroke_fill=None,
            spacing=spacing, align="center",
        )

    return image
