from __future__ import annotations

from io import BytesIO
from time import perf_counter
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from PIL import Image

from .detector import detect_text_regions
from .models import TextBlock
from .ocr import recognize_blocks
from .renderer import inpaint_text, render_translations
from .utils import sha1_bytes


def _overlap_len(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, min(a2, b2) - max(a1, b1))


def _axis_gap(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, max(a1, b1) - min(a2, b2))


def _block_area(block: TextBlock) -> int:
    x1, y1, x2, y2 = block.bounds
    return max(1, x2 - x1) * max(1, y2 - y1)


def _block_center(block: TextBlock) -> tuple[float, float]:
    x1, y1, x2, y2 = block.bounds
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _component_bounds(items: list[TextBlock]) -> tuple[int, int, int, int]:
    xs = [point[0] for item in items for point in item.box]
    ys = [point[1] for item in items for point in item.box]
    return min(xs), min(ys), max(xs), max(ys)


def _assign_region_ids(
    blocks: list[TextBlock],
    image_rgb: np.ndarray,
) -> tuple[list[int | None], list[Any]]:
    regions = detect_text_regions(image_rgb)
    if not regions:
        return [None] * len(blocks), []

    region_ids: list[int | None] = []
    for block in blocks:
        cx, cy = _block_center(block)
        matched: list[tuple[int, int]] = []

        for region_index, region in enumerate(regions):
            inside = region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2
            if inside:
                matched.append((region.area, region_index))

        if matched:
            matched.sort()
            region_ids.append(matched[0][1])
        else:
            region_ids.append(None)

    return region_ids, regions


def _blocks_are_neighbors(
    a: TextBlock,
    b: TextBlock,
    region_a: int | None,
    region_b: int | None,
) -> bool:
    if region_a is not None and region_b is not None and region_a != region_b:
        return False

    ax1, ay1, ax2, ay2 = a.bounds
    bx1, by1, bx2, by2 = b.bounds

    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    x_overlap = _overlap_len(ax1, ax2, bx1, bx2)
    y_overlap = _overlap_len(ay1, ay2, by1, by2)
    x_overlap_ratio = x_overlap / float(max(1, min(aw, bw)))
    y_overlap_ratio = y_overlap / float(max(1, min(ah, bh)))

    h_gap = _axis_gap(ax1, ax2, bx1, bx2)
    v_gap = _axis_gap(ay1, ay2, by1, by2)

    acx, acy = _block_center(a)
    bcx, bcy = _block_center(b)
    center_dx = abs(acx - bcx)
    center_dy = abs(acy - bcy)

    a_tall = ah > aw * 1.15
    b_tall = bh > bw * 1.15

    vertical_neighbor = (
        x_overlap_ratio >= 0.32
        and center_dx <= max(16, int(min(aw, bw) * 0.45))
        and v_gap <= max(12, int(max(ah, bh) * 0.60))
    )
    horizontal_neighbor = (
        y_overlap_ratio >= 0.58
        and center_dy <= max(10, int(min(ah, bh) * 0.32))
        and h_gap <= max(14, int(min(aw, bw) * 0.45))
    )

    if horizontal_neighbor and a_tall and b_tall and h_gap > max(8, int(min(aw, bw) * 0.22)):
        horizontal_neighbor = False

    touching = h_gap == 0 and v_gap == 0 and (x_overlap_ratio >= 0.22 or y_overlap_ratio >= 0.22)
    return touching or vertical_neighbor or horizontal_neighbor


def _split_component_by_x_gap(items: list[TextBlock]) -> list[list[TextBlock]]:
    if len(items) < 4:
        return [items]

    ordered = sorted(items, key=lambda item: _block_center(item)[0])
    centers = [_block_center(item)[0] for item in ordered]
    widths = [max(1, item.bounds[2] - item.bounds[0]) for item in ordered]
    median_width = float(np.median(np.array(widths, dtype=np.float32))) if widths else 0.0

    best_index: int | None = None
    best_gap = 0.0

    for split_index in range(2, len(ordered) - 1):
        gap = centers[split_index] - centers[split_index - 1]
        threshold = max(
            26.0,
            median_width * 0.75,
            float(min(widths[split_index - 1], widths[split_index])) * 0.85,
        )
        if gap < threshold:
            continue

        left_items = ordered[:split_index]
        right_items = ordered[split_index:]
        lx1, ly1, lx2, ly2 = _component_bounds(left_items)
        rx1, ry1, rx2, ry2 = _component_bounds(right_items)

        left_height = max(1, ly2 - ly1)
        right_height = max(1, ry2 - ry1)
        y_overlap = _overlap_len(ly1, ly2, ry1, ry2)
        if y_overlap < max(18, int(min(left_height, right_height) * 0.20)):
            continue

        left_width = max(1, lx2 - lx1)
        right_width = max(1, rx2 - rx1)
        x_overlap = _overlap_len(lx1, lx2, rx1, rx2)
        if x_overlap > max(8, int(min(left_width, right_width) * 0.08)):
            continue

        if gap > best_gap:
            best_gap = gap
            best_index = split_index

    if best_index is None:
        return [items]

    left = ordered[:best_index]
    right = ordered[best_index:]
    split_groups: list[list[TextBlock]] = []
    split_groups.extend(_split_component_by_x_gap(left))
    split_groups.extend(_split_component_by_x_gap(right))
    return split_groups


def _sort_group_items(items: list[TextBlock]) -> list[TextBlock]:
    sorted_items = sorted(items, key=lambda b: (b.bounds[1], b.bounds[0]))
    lines: list[dict[str, Any]] = []

    for item in sorted_items:
        x1, y1, x2, y2 = item.bounds
        item_center_y = (y1 + y2) / 2.0
        item_height = max(1, y2 - y1)
        placed = False

        for line in lines:
            avg_height = max(1.0, float(line["avg_height"]))
            if abs(item_center_y - float(line["center_y"])) <= max(12.0, avg_height * 0.55):
                line["items"].append(item)
                count = len(line["items"])
                line["center_y"] = ((float(line["center_y"]) * (count - 1)) + item_center_y) / count
                line["avg_height"] = ((avg_height * (count - 1)) + item_height) / count
                placed = True
                break

        if not placed:
            lines.append(
                {
                    "center_y": item_center_y,
                    "avg_height": float(item_height),
                    "items": [item],
                }
            )

    ordered: list[TextBlock] = []
    for line in sorted(lines, key=lambda entry: float(entry["center_y"])):
        ordered.extend(sorted(line["items"], key=lambda block: block.bounds[0]))
    return ordered


def group_blocks(
    blocks: list[TextBlock],
    image_rgb: np.ndarray,
    region_ids: list[int | None] | None = None,
) -> list[TextBlock]:
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b.bounds[1], b.bounds[0]))
    if region_ids is None:
        region_ids, _ = _assign_region_ids(blocks, image_rgb)

    adjacency: list[list[int]] = [[] for _ in blocks]
    for left_index, left_block in enumerate(blocks):
        _, left_y1, _, left_y2 = left_block.bounds
        for right_index in range(left_index + 1, len(blocks)):
            right_block = blocks[right_index]
            _, right_y1, _, _ = right_block.bounds
            if right_y1 - left_y2 > 140:
                break

            if _blocks_are_neighbors(
                left_block,
                right_block,
                region_ids[left_index],
                region_ids[right_index],
            ):
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)

    merged: list[TextBlock] = []
    visited = [False] * len(blocks)
    for start_index in range(len(blocks)):
        if visited[start_index]:
            continue

        stack = [start_index]
        component: list[TextBlock] = []
        while stack:
            current_index = stack.pop()
            if visited[current_index]:
                continue
            visited[current_index] = True
            component.append(blocks[current_index])
            stack.extend(adjacency[current_index])

        for subgroup in _split_component_by_x_gap(component):
            items = _sort_group_items(subgroup)
            gx1, gy1, gx2, gy2 = _component_bounds(items)

            text = " ".join(
                item.source_text.strip(" \n\t,.;:_")
                for item in items
                if item.source_text.strip()
            ).strip()

            if not text:
                continue

            merged.append(
                TextBlock(
                    box=[(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2)],
                    source_text=text,
                )
            )

    return sorted(merged, key=lambda block: (block.bounds[1], block.bounds[0]))


def process_image_bytes(
    content: bytes,
    results_dir: Path,
    source_ocr_lang: str = "en",
    target_lang: str = "Russian",
) -> tuple[Path, dict[str, Any]]:
    total_started = perf_counter()
    image = Image.open(BytesIO(content)).convert("RGB")
    np_image = np.array(image)

    digest = sha1_bytes(content)
    debug_dir = results_dir / f"{digest}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    ocr_started = perf_counter()
    raw_blocks = recognize_blocks(
        image,
        debug_dir=debug_dir,
        source_ocr_lang=source_ocr_lang,
    )
    ocr_ms = round((perf_counter() - ocr_started) * 1000)
    print(f"[PIPELINE] blocks recognized raw: {len(raw_blocks)}")

    group_started = perf_counter()
    region_ids, regions = _assign_region_ids(raw_blocks, np_image)
    blocks = group_blocks(raw_blocks, np_image, region_ids=region_ids)
    group_ms = round((perf_counter() - group_started) * 1000)
    print(f"[PIPELINE] blocks grouped: {len(blocks)}")

    debug_img = np_image.copy()
    region_debug_img = np_image.copy()
    for region_index, region in enumerate(regions, start=1):
        cv2.rectangle(
            region_debug_img,
            (region.x1, region.y1),
            (region.x2, region.y2),
            (0, 180, 0),
            2,
        )
        cv2.putText(
            region_debug_img,
            str(region_index),
            (region.x1, max(20, region.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(region_debug_img).save(debug_dir / "text_regions.png")

    for i, block in enumerate(blocks, start=1):
        x1, y1, x2, y2 = block.bounds
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            debug_img,
            str(i),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(debug_img).save(debug_dir / "grouped_blocks.png")
    
    
    from .translator import get_translator

    translator_started = perf_counter()
    translator = get_translator()

    texts = [block.source_text for block in blocks]
    for i, text in enumerate(texts, start=1):
        print(f"[PIPELINE] source[{i}]: {text!r}")

    translation_started = perf_counter()
    translated_texts = translator.translate_batch(texts, target_language=target_lang)
    translation_ms = round((perf_counter() - translation_started) * 1000)

    for i, (block, translated) in enumerate(zip(blocks, translated_texts), start=1):
        block.translated_text = translated
        print(f"[PIPELINE] translated[{i}]: {translated!r}")

    inpaint_started = perf_counter()
    cleaned = inpaint_text(np_image, blocks)
    inpaint_ms = round((perf_counter() - inpaint_started) * 1000)

    render_started = perf_counter()
    rendered = render_translations(cleaned, blocks, original_image=np_image)
    render_ms = round((perf_counter() - render_started) * 1000)

    save_started = perf_counter()
    out_path = results_dir / f"{digest}.png"
    rendered.save(out_path)
    save_ms = round((perf_counter() - save_started) * 1000)
    translator_init_ms = round((translation_started - translator_started) * 1000)
    total_ms = round((perf_counter() - total_started) * 1000)

    meta = {
        "source_ocr_lang": source_ocr_lang,
        "target_lang": target_lang,
        "boxes_detected": len(raw_blocks),
        "region_candidates": len(regions),
        "boxes_grouped": len(blocks),
        "boxes_used": len(blocks),
        "source_texts": [block.source_text for block in blocks],
        "translated_texts": [block.translated_text for block in blocks],
        "debug_dir": str(debug_dir),
        "timings_ms": {
            "ocr": ocr_ms,
            "group": group_ms,
            "translator_init": translator_init_ms,
            "translate": translation_ms,
            "inpaint": inpaint_ms,
            "render": render_ms,
            "save": save_ms,
            "total": total_ms,
        },
    }

    print("[PIPELINE] meta =", json.dumps(meta, ensure_ascii=False, indent=2))
    return out_path, meta
