from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple


Point = Tuple[int, int]


@dataclass
class TextBlock:
    box: List[Point]
    source_text: str
    translated_text: str = ""

    @cached_property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return min(xs), min(ys), max(xs), max(ys)
