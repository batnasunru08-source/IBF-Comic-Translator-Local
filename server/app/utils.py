from __future__ import annotations

import hashlib
import json
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


_translation_filter_cache: dict[str, tuple[float, dict[str, frozenset[str]]]] = {}


def load_translation_filter(config_path: Path | None = None) -> dict[str, frozenset[str]]:
    """Загружает фильтр слов, которые не надо переводить.

    По умолчанию ищет config в server/data/translation_filter.json.
    Если файл отсутствует — возвращает пустые множества.
    Кеширует результат, но перезагружает при изменении mtime файла.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "data" / "translation_filter.json"

    if not config_path.exists():
        return {"watermark_tokens": frozenset(), "known_repeats": frozenset()}

    key = str(config_path)
    mtime = config_path.stat().st_mtime
    cached = _translation_filter_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    result = {
        "watermark_tokens": frozenset(t.lower() for t in data.get("watermark_tokens", [])),
        "known_repeats": frozenset(t.lower() for t in data.get("known_repeats", [])),
        "noise_tokens": frozenset(t.lower() for t in data.get("noise_tokens", [])),
    }
    _translation_filter_cache[key] = (mtime, result)
    return result