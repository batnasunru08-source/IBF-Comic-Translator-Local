from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

JP_RE = re.compile(r"[一-龯ぁ-ゔァ-ヴー々〆〤]")
LATIN_RE = re.compile(r"[A-Za-z]")
ONLY_PUNCT_RE = re.compile(r"^[\s\.\,·•…．。・!！?？:;\"'`~ー〜～「」『』（）()\[\]{}【】\-—─_=+|/\\]+$")
CJK_RE = re.compile(r"[一-龯ぁ-ゔァ-ヴー々〆〤가-힯]")


def is_cjk_text(text: str) -> bool:
    """True, если текст содержит CJK-скрипты (кандзи/кана/хангыль) и не содержит латиницы."""
    return bool(CJK_RE.search(text)) and not LATIN_RE.search(text)


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


_translation_filter_cache: dict[str, tuple[int, dict]] = {}

# Дефолтный набор звукоподражаний (EN). Переопределяется ключом sfx_tokens
# в data/translation_filter.json.
_DEFAULT_SFX_TOKENS = (
    "boom", "bang", "crash", "thud", "thunk", "thump", "whoosh", "woosh",
    "swoosh", "pow", "zap", "bam", "wham", "kaboom", "clang", "clank",
    "screech", "rumble", "splash", "hiss", "buzz", "snap", "crack",
    "crackle", "crunch", "grr", "roar", "whack", "smack", "slam",
    "whirr", "vroom", "ding", "plop", "drip", "shatter", "ratatat",
)

_FILTER_DEFAULTS: dict = {
    "watermark_tokens": frozenset(),
    "known_repeats": frozenset(),
    "noise_tokens": frozenset(),
    "sfx_tokens": frozenset(_DEFAULT_SFX_TOKENS),
    "sfx_mode": "skip",
    "min_text_len": 5,
    "min_alnum_ratio": 0.40,
    "ocr_conf_min": 0.15,
    "low_conf_threshold": 0.45,
    "low_conf_max_len": 8,
    "cjk_min_text_len": 1,
    "huge_block_area_ratio": 0.20,
    "huge_block_min_chars": 30,
    "huge_block_min_density": 0.0003,
}

_TOKEN_KEYS = ("watermark_tokens", "known_repeats", "noise_tokens", "sfx_tokens")

_SCALAR_CASTS = (
    ("sfx_mode", str),
    ("min_text_len", int),
    ("min_alnum_ratio", float),
    ("ocr_conf_min", float),
    ("low_conf_threshold", float),
    ("low_conf_max_len", int),
    ("cjk_min_text_len", int),
    ("huge_block_area_ratio", float),
    ("huge_block_min_chars", int),
    ("huge_block_min_density", float),
)


def load_translation_filter(config_path: Path | None = None) -> dict:
    """Загружает фильтр слов, которые не надо переводить, а также
    пороги и режимы фильтрации OCR-блоков (sfx_mode, min_text_len и т.п.).

    По умолчанию ищет config в server/data/translation_filter.json.
    Ключи JSON накладываются на _FILTER_DEFAULTS, поэтому частичный конфиг валиден.
    Кеширует результат, перезагружает при изменении mtime файла.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "data" / "translation_filter.json"

    if not config_path.exists():
        return dict(_FILTER_DEFAULTS)

    key = str(config_path)
    mtime = config_path.stat().st_mtime_ns
    cached = _translation_filter_cache.get(key)
    if cached is not None and cached[0] == mtime:
        # Копия — чтобы мутации результата вызывающим кодом не портили кеш.
        return dict(cached[1])

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    result = dict(_FILTER_DEFAULTS)
    for token_key in _TOKEN_KEYS:
        if token_key in data:
            if not isinstance(data[token_key], list):
                print(f"[FILTER] invalid {token_key}={data[token_key]!r}, using default")
                continue
            result[token_key] = frozenset(str(t).lower() for t in data[token_key])
    for scalar_key, cast in _SCALAR_CASTS:
        if scalar_key in data:
            try:
                result[scalar_key] = cast(data[scalar_key])
            except (TypeError, ValueError):
                print(f"[FILTER] invalid {scalar_key}={data[scalar_key]!r}, using default")

    _translation_filter_cache[key] = (mtime, result)
    # Копия — кеш хранит оригинал, мутации результата его не заденут.
    return dict(result)


_SFX_CAPS_LEN_MAX = 15
_SFX_CAPS_ALPHA_MAX = 10
_SFX_CAPS_WORDS_MAX = 2


def _is_caps_shout(text: str) -> bool:
    """Признак короткого капс-текста (SFX или крик): ALL CAPS, ≤10 букв, ≤2 слов."""
    if not text.isupper() or len(text) > _SFX_CAPS_LEN_MAX:
        return False
    alpha = sum(c.isalpha() for c in text)
    return alpha <= _SFX_CAPS_ALPHA_MAX and len(text.split()) <= _SFX_CAPS_WORDS_MAX


def looks_translatable(text: str | None, conf: float = 1.0, cfg: dict | None = None) -> tuple[bool, str]:
    """Решает, отправлять ли текст на перевод. Возвращает (ok, reason).

    reason: ok, empty, too_short, low_conf, low_alnum, too_few_letters,
    single_char_repeat, ocr_artifact, single_letters, syllable_repeat,
    watermark, noise_token, sfx_token.

    Языковой профиль определяется по скрипту текста: для CJK-only минимальная
    длина — cjk_min_text_len (одиночный кандзи — валидное слово), латинские
    правила (одиночные буквы, слоги, капс-SFX) не применяются.
    Пороги и словари — из data/translation_filter.json (см. load_translation_filter).
    """
    if cfg is None:
        cfg = load_translation_filter()
    text = (text or "").strip()
    if not text:
        return False, "empty"

    cjk = is_cjk_text(text)
    shout = not cjk and _is_caps_shout(text)

    # Капс-реплики и SFX короче min_text_len по длине не отбрасываем: у короткого
    # капса свой путь — словарь SFX (ниже) либо перевод ("NO!", "BOOM").
    min_len = int(cfg["cjk_min_text_len"]) if cjk else int(cfg["min_text_len"])
    if len(text) < min_len and not shout:
        return False, "too_short"

    # Низкий confidence + короткий текст — типичный OCR-шум.
    if conf < float(cfg["low_conf_threshold"]) and len(text) < int(cfg["low_conf_max_len"]):
        return False, "low_conf"

    alnum = sum(c.isalnum() for c in text)
    if alnum / len(text) < float(cfg["min_alnum_ratio"]):
        return False, "low_alnum"

    letters = sum(c.isalpha() for c in text)
    if letters < (1 if cjk else 2):
        return False, "too_few_letters"

    # Повтор одного символа — OCR-шум в латинице: "####", "....", "AAAA".
    # К CJK не применяем: удвоение иероглифа/каны — продуктивная лексика
    # (谢谢, 妈妈, ああ), а не шум.
    if not cjk and len(text) >= 2 and len(set(text.replace(" ", ""))) <= 1:
        return False, "single_char_repeat"

    if "\\" in text or re.search(r"\$[^$]*\$", text) or re.search(r"[\{\}\[\]]", text):
        return False, "ocr_artifact"

    if not cjk:
        # Набор одиночных букв: "W B", "V T", "HM? J"
        words = text.split()
        alpha_words = [re.sub(r"[^a-zA-ZЀ-ӿ぀-ヿ一-鿿]", "", w) for w in words]
        alpha_words = [w for w in alpha_words if w]
        if alpha_words and all(len(w) == 1 for w in alpha_words):
            return False, "single_letters"
        if len(alpha_words) >= 2:
            single_ratio = sum(1 for w in alpha_words if len(w) == 1) / len(alpha_words)
            threshold = 0.5 if len(alpha_words) <= 3 else 0.6
            if single_ratio >= threshold:
                return False, "single_letters"

        # Повторяющийся слог: "ofof", "abab" (но не "haha", "mama")
        t_clean = text.lower().replace(" ", "")
        if 4 <= len(t_clean) <= 8 and t_clean.isalpha() and t_clean not in cfg["known_repeats"]:
            half = len(t_clean) // 2
            if t_clean[:half] == t_clean[half:half * 2]:
                return False, "syllable_repeat"

    text_lower = text.lower()
    if any(token in text_lower for token in cfg["watermark_tokens"]):
        return False, "watermark"
    if any(token in text_lower for token in cfg["noise_tokens"]):
        return False, "noise_token"

    # SFX: короткий капс из словаря звукоподражаний. Капс-реплики вне словаря
    # ("WHAT?!") переводим. sfx_mode=translate отключает правило полностью.
    if shout and cfg["sfx_mode"] != "translate":
        token = re.sub(r"[^a-z]", "", text_lower)
        if token and token in cfg["sfx_tokens"]:
            return False, "sfx_token"

    return True, "ok"