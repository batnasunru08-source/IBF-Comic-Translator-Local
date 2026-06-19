from __future__ import annotations


import logging
import os
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_FILE = "Hy-MT2-1.8B-Q8_0.gguf"

_TEMPERATURE        = 0.7
_TOP_P              = 0.6
_TOP_K              = 20
_REPETITION_PENALTY = 1.05
_MAX_TOKENS         = 2048  # больше для батча

# Промпт для одного текста
_SINGLE_PROMPT = (
    "Translate the following text into {lang}. "
    "Output only the translated result, no explanations:\n\n{text}"
)

# Промпт для батча — нумерованные строки, без JSON (надёжнее при кавычках).
# Список правил вынесен перед items, но без дефисных маркеров, чтобы модель
# не повторяла их в ответе.
_BATCH_PROMPT = (
    "Translate each numbered item below into {lang}. "
    "Output only the translations, one per line, keeping the original numbers. "
    "Do not repeat these instructions.\n\n{items}"
)

# Паттерн для парсинга нумерованных строк: "1. текст" или "1) текст"
_NUMBERED_LINE_RE = re.compile(r"^\d+[.)\s]\s*(.+)$", re.MULTILINE)


class HyMt2Translator:
    def __init__(self) -> None:
        model_file = os.environ.get("MODEL_FILE", "").strip() or DEFAULT_MODEL_FILE
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF модель не найдена: {model_path}\n"
                f"Скачай: bash download-model.sh\n"
                f"  1.8B: bash download-model.sh\n"
                f"  7B:   bash download-model.sh 7b\n"
                f"  Затем: MODEL_FILE={model_file} python -m uvicorn main:app"
            )
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as err:
            raise RuntimeError(
                "llama-cpp-python не установлен.\n"
                "CPU: pip install llama-cpp-python\n"
                "GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            ) from err

        n_gpu_layers = self._pick_gpu_layers()
        logger.info("[TRANSLATOR] Loading %s n_gpu_layers=%d", model_path.name, n_gpu_layers)

        self._llm = Llama(
            model_path=str(model_path),
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

    def reset(self) -> None:
        """Сброс KV-кеша и внутреннего состояния модели для освобождения VRAM.

        Вызывается между независимыми задачами (например, между обработкой
        разных изображений), чтобы GPU-буферы, выделенные под n_ctx, не
        удерживались после завершения генерации.
        """
        try:
            self._llm.reset()
            logger.info("[TRANSLATOR] Context reset OK")
        except Exception as err:
            logger.warning("[TRANSLATOR] reset failed: %s", err)

    # Маркеры эха промпта — строки, которые модель склонна повторять из инструкции.
    _PROMPT_ECHO_MARKERS = (
        "output", "translate", "rules", "numbered", "explanations",
        "сохранять", "выводить", "никаких", "программа",
    )

    @classmethod
    def _looks_like_prompt_echo(cls, text: str) -> bool:
        """True, если строка похожа на повтор инструкции из промпта, а не на перевод."""
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.startswith(("-", "=", "*", "#")):
            return True
        lowered = stripped.lower()
        if any(marker in lowered for marker in cls._PROMPT_ECHO_MARKERS):
            return True
        return False

    def _parse_numbered_lines(
        self, raw: str, expected: int
    ) -> tuple[list[str | None], list[int]]:
        """Парсит ответ вида '1. текст\n2. текст\n...'

        Возвращает (translations, missing_indices):
        - translations: список длиной `expected`, None для нераспознанных
        - missing_indices: индексы (1-based номера) тех, что не нашлись
        """
        # Убираем markdown-блоки если модель обернула ответ
        raw = re.sub(r"```[\w]*\n?", "", raw).strip()

        result: list[str | None] = [None] * expected
        matched_numbers: set[int] = set()

        # Сначала ищем пронумерованные строки
        for match in _NUMBERED_LINE_RE.finditer(raw):
            num_str = raw[match.start():match.end()].split(".", 1)[0].split(")", 1)[0].strip()
            try:
                num = int(num_str)
            except ValueError:
                continue
            if 1 <= num <= expected and num not in matched_numbers:
                text = match.group(1).strip()
                if self._looks_like_prompt_echo(text):
                    # Эхо промпта — не считаем за валидный перевод
                    continue
                result[num - 1] = text
                matched_numbers.add(num)

        if all(r is not None for r in result):
            return [r for r in result], []

        # Если модель писала без номеров — пробуем заполнить нераспознанные по строкам
        unnumbered_lines = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not _NUMBERED_LINE_RE.match(line.strip())
            and not self._looks_like_prompt_echo(line)
        ]
        if unnumbered_lines:
            missing = [i for i, r in enumerate(result) if r is None]
            for idx, line in zip(missing, unnumbered_lines):
                result[idx] = line

        missing_indices = [i + 1 for i, r in enumerate(result) if r is None]
        return result, missing_indices

    def _translate_batch_single_call(self, texts: list[str], lang: str) -> list[str]:
        """Отправляет все тексты одним запросом — модель возвращает нумерованные строки."""
        items = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = _BATCH_PROMPT.format(lang=lang, items=items)
        raw = self._call(prompt, max_tokens=_MAX_TOKENS)

        result, missing = self._parse_numbered_lines(raw, len(texts))

        if missing:
            logger.warning(
                f"[TRANSLATOR] Batch parse incomplete: {len(texts) - len(missing)}/"
                f"{len(texts)} lines ok, missing {missing}, translating sequentially"
            )
            for idx in missing:
                result[idx - 1] = self._translate_single(texts[idx - 1], lang)

        return [r or "" for r in result]

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
