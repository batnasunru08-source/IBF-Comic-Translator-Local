from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Point = Tuple[int, int]


@dataclass(slots=True)
class TextBlock:
    box: List[Point]
    source_text: str
    translated_text: str = ""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)
        return x1, y1, x2, y2