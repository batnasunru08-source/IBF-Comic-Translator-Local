from __future__ import annotations


import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "Hy-MT2-1.8B-Q8_0.gguf"

_TEMPERATURE        = 0.7
_TOP_P              = 0.6
_TOP_K              = 20
_REPETITION_PENALTY = 1.05
_MAX_TOKENS         = 1024  # больше для батча

# Промпт для одного текста
_SINGLE_PROMPT = (
    "Translate the following text into {lang}. "
    "Output only the translated result, no explanations:\n\n{text}"
)

# Промпт для батча — нумерованные строки, без JSON (надёжнее при кавычках)
_BATCH_PROMPT = (
    "Translate each numbered item into {lang}.\n"
    "Rules:\n"
    "- Output ONLY the translations, one per line\n"
    "- Keep the same numbering: 1. 2. 3. etc\n"
    "- No explanations, no extra text\n\n"
    "{items}"
)

# Паттерн для парсинга нумерованных строк: "1. текст" или "1) текст"
_NUMBERED_LINE_RE = re.compile(r"^\d+[.)\s]\s*(.+)$", re.MULTILINE)


class HyMt2Translator:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"GGUF модель не найдена: {MODEL_PATH}\n"
                f"Скачай: bash download-model.sh"
            )
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError:
            raise RuntimeError(
                "llama-cpp-python не установлен.\n"
                "CPU: pip install llama-cpp-python\n"
                "GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        n_gpu_layers = self._pick_gpu_layers()
        logger.info(f"[TRANSLATOR] Loading {MODEL_PATH.name} n_gpu_layers={n_gpu_layers}")

        self._llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("[TRANSLATOR] Model loaded OK")

    @staticmethod
    def _pick_gpu_layers() -> int:
        try:
            import torch
            if torch.cuda.is_available():
                return -1
        except Exception:
            pass
        return 0

    def _call(self, prompt: str, max_tokens: int = _MAX_TOKENS) -> str:
        output = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
            top_k=_TOP_K,
            repeat_penalty=_REPETITION_PENALTY,
            max_tokens=max_tokens,
        )
        return output["choices"][0]["message"]["content"].strip()

    def _translate_single(self, text: str, lang: str) -> str:
        prompt = _SINGLE_PROMPT.format(lang=lang, text=text)
        return self._call(prompt, max_tokens=256)

    def _parse_numbered_lines(self, raw: str, expected: int) -> list[str] | None:
        """Парсит ответ вида '1. текст\n2. текст\n...'

        Возвращает список переводов или None если не удалось распознать.
        """
        # Убираем markdown-блоки если модель обернула ответ
        raw = re.sub(r"```[\w]*\n?", "", raw).strip()

        matches = _NUMBERED_LINE_RE.findall(raw)
        if len(matches) == expected:
            return [m.strip() for m in matches]

        # Иногда модель пишет без номеров — пробуем разбить по строкам
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        if len(lines) == expected:
            return lines

        return None

    def _translate_batch_single_call(self, texts: list[str], lang: str) -> list[str]:
        """Отправляет все тексты одним запросом — модель возвращает нумерованные строки."""
        items = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = _BATCH_PROMPT.format(lang=lang, items=items)
        raw = self._call(prompt, max_tokens=_MAX_TOKENS)

        parsed = self._parse_numbered_lines(raw, len(texts))
        if parsed is not None:
            return parsed

        # Fallback: переводим по одному
        logger.warning(
            f"[TRANSLATOR] Batch parse failed (got {len(raw.splitlines())} lines, "
            f"expected {len(texts)}), falling back to sequential"
        )
        return [self._translate_single(t, lang) for t in texts]

    def translate_batch(self, texts: list[str], target_language: str = "Russian") -> list[str]:
        # Разделяем на непустые (для перевода) и пустые (мусор/скип)
        indices_to_translate = [(i, t) for i, t in enumerate(texts) if (t or "").strip()]
        result = [""] * len(texts)

        if not indices_to_translate:
            return result

        indices, clean_texts = zip(*indices_to_translate)

        if len(clean_texts) == 1:
            # Один текст — без оверхеда на JSON
            translations = [self._translate_single(clean_texts[0], target_language)]
        else:
            # Несколько текстов — один запрос
            translations = self._translate_batch_single_call(list(clean_texts), target_language)

        for idx, translation in zip(indices, translations):
            result[idx] = translation

        return result

    def translate(self, source_text: str, target_language: str = "Russian") -> str:
        return self.translate_batch([source_text], target_language=target_language)[0]


@lru_cache(maxsize=1)
def get_translator() -> HyMt2Translator:
    return HyMt2Translator()
