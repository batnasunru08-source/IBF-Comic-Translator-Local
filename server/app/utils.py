from __future__ import annotations

import hashlib
import re
from pathlib import Path

JP_RE = re.compile(r"[一-龯ぁ-ゔァ-ヴー々〆〤]")
LATIN_RE = re.compile(r"[A-Za-z]")
ONLY_PUNCT_RE = re.compile(r"^[\s\.\,·•…．。・!！?？:;\"'`~ー〜～「」『』（）()\[\]{}【】\-—─_=+|/\\]+$")


def sha1_bytes(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def looks_like_meaningful_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False

    if ONLY_PUNCT_RE.fullmatch(text):
        return False

    if JP_RE.search(text):
        return True

    if LATIN_RE.search(text):
        return True

    alnum_count = sum(ch.isalnum() for ch in text)
    return alnum_count >= 2


def first_existing_path(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None