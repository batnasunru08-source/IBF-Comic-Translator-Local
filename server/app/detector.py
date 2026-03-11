from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(slots=True)
class CandidateBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


def _boxes_close(a: CandidateBox, b: CandidateBox, gap_x: int, gap_y: int) -> bool:
    return not (
        a.x2 + gap_x < b.x1
        or b.x2 + gap_x < a.x1
        or a.y2 + gap_y < b.y1
        or b.y2 + gap_y < a.y1
    )


def _merge_two(a: CandidateBox, b: CandidateBox) -> CandidateBox:
    return CandidateBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def _merge_nearby_boxes(boxes: List[CandidateBox], gap_x: int, gap_y: int) -> List[CandidateBox]:
    merged = boxes[:]
    changed = True

    while changed:
        changed = False
        result: List[CandidateBox] = []
        used = [False] * len(merged)

        for i, a in enumerate(merged):
            if used[i]:
                continue

            current = a
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                if _boxes_close(current, b, gap_x, gap_y):
                    current = _merge_two(current, b)
                    used[j] = True
                    changed = True

            used[i] = True
            result.append(current)

        merged = result

    return merged


def detect_text_regions(image: np.ndarray) -> List[CandidateBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Немного сглаживаем шум
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Ищем и тёмный текст на светлом, и светлый текст на тёмном
    mask_dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )

    mask_light = cv2.adaptiveThreshold(
        255 - blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )

    mask = cv2.bitwise_or(mask_dark, mask_light)

    h, w = gray.shape
    image_area = h * w

    # Более агрессивное соединение символов в один блок
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w // 140), max(7, h // 140)))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (max(11, w // 90), max(11, h // 90)))

    merged = cv2.dilate(mask, kernel1, iterations=1)
    merged = cv2.dilate(merged, kernel2, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[CandidateBox] = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh

        if bw < 20 or bh < 20:
            continue
        if area < max(900, image_area // 3500):
            continue
        if area > image_area * 0.5:
            continue

        aspect = bw / max(1, bh)
        if aspect > 20 or aspect < 0.05:
            continue

        roi = mask[y:y + bh, x:x + bw]
        density = float(np.count_nonzero(roi)) / float(area)
        if density < 0.006 or density > 0.75:
            continue

        # Увеличиваем рамку заметно сильнее, чтобы захватить всю фразу
        pad_x = max(16, bw // 6)
        pad_y = max(16, bh // 6)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        boxes.append(CandidateBox(x1=x1, y1=y1, x2=x2, y2=y2))

    boxes = _merge_nearby_boxes(
        boxes,
        gap_x=max(24, w // 40),
        gap_y=max(24, h // 40),
    )

    boxes.sort(key=lambda b: (b.y1, b.x1))
    return boxes