from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
from PIL import Image

from .models import TextBlock
from .ocr import recognize_blocks
from .renderer import inpaint_text, render_translations
from .translator import get_translator
from .utils import sha1_bytes


def group_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b.bounds[1], b.bounds[0]))
    groups: list[dict[str, Any]] = []

    for block in blocks:
        x1, y1, x2, y2 = block.bounds
        bw = x2 - x1
        bh = y2 - y1
        placed = False

        for group in groups:
            gx1, gy1, gx2, gy2 = group["bounds"]
            gbh = gy2 - gy1

            similar_x = abs(x1 - gx1) <= max(50, bw // 2)
            vertical_gap = y1 - gy2
            close_y = vertical_gap <= max(35, int(max(bh, gbh) * 1.6))
            x_overlap = min(x2, gx2) - max(x1, gx1)

            if similar_x and close_y and x_overlap > -30:
                group["items"].append(block)
                group["bounds"] = [
                    min(gx1, x1),
                    min(gy1, y1),
                    max(gx2, x2),
                    max(gy2, y2),
                ]
                placed = True
                break

        if not placed:
            groups.append(
                {
                    "items": [block],
                    "bounds": [x1, y1, x2, y2],
                }
            )

    merged: list[TextBlock] = []
    for group in groups:
        gx1, gy1, gx2, gy2 = group["bounds"]
        items = sorted(group["items"], key=lambda b: (b.bounds[1], b.bounds[0]))

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

    return merged


def process_image_bytes(
    content: bytes,
    results_dir: Path,
    source_ocr_lang: str = "en",
    target_lang: str = "Russian",
) -> tuple[Path, dict[str, Any]]:
    image = Image.open(BytesIO(content)).convert("RGB")
    np_image = np.array(image)

    digest = sha1_bytes(content)
    debug_dir = results_dir / f"{digest}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    raw_blocks = recognize_blocks(
        image,
        debug_dir=debug_dir,
        source_ocr_lang=source_ocr_lang,
    )
    print(f"[PIPELINE] blocks recognized raw: {len(raw_blocks)}")

    blocks = group_blocks(raw_blocks)
    print(f"[PIPELINE] blocks grouped: {len(blocks)}")

    debug_img = np_image.copy()
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

    translator = get_translator()
    for i, block in enumerate(blocks, start=1):
        print(f"[PIPELINE] source[{i}]: {block.source_text!r}")
        block.translated_text = translator.translate(block.source_text, target_language=target_lang)
        print(f"[PIPELINE] translated[{i}]: {block.translated_text!r}")

    cleaned = inpaint_text(np_image, blocks)
    rendered = render_translations(cleaned, blocks)

    out_path = results_dir / f"{digest}.png"
    rendered.save(out_path)

    meta = {
        "source_ocr_lang": source_ocr_lang,
        "target_lang": target_lang,
        "boxes_detected": len(raw_blocks),
        "boxes_grouped": len(blocks),
        "boxes_used": len(blocks),
        "source_texts": [block.source_text for block in blocks],
        "translated_texts": [block.translated_text for block in blocks],
        "debug_dir": str(debug_dir),
    }

    print("[PIPELINE] meta =", json.dumps(meta, ensure_ascii=False, indent=2))
    return out_path, meta