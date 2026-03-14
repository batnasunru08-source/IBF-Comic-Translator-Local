from __future__ import annotations

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
MODEL_ID = "tencent/HY-MT1.5-7B"
# Для более высокой скорости можно попробовать:
# MODEL_ID = "tencent/HY-MT1.5-1.8B"
# MODEL_ID = "tencent/HY-MT1.5-7B"
# MODEL_ID = "tencent/HY-MT1.5-7B-FP8"


class HyMtTranslator:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = self._load_model()
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

    def _load_model(self):
        common_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Trying 4-bit bitsandbytes load on CUDA")
                return AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    quantization_config=quantization_config,
                    dtype=torch.bfloat16,
                    **common_kwargs,
                )
            except Exception as exc:
                logger.warning("4-bit load failed, fallback to non-quantized CUDA load: %s", exc)
                return AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    dtype=torch.float16,
                    **common_kwargs,
                )

        logger.info("CUDA not available, loading on CPU")
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, **common_kwargs)

    def _get_model_input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_prompt(self, source_text: str, target_language: str) -> str:
        return (
            f"Translate the following segment into {target_language}, "
            "without additional explanation.\n\n"
            f"{source_text}"
        )

    def translate_batch(self, texts: list[str], target_language: str = "Russian") -> list[str]:
        clean_texts = [(t or "").strip() for t in texts]
        prompts = [self._build_prompt(t, target_language) for t in clean_texts if t]

        if not prompts:
            return [""] * len(texts)

        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]

        rendered_prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in messages_batch
        ]

        model_inputs = self.tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_inputs.pop("token_type_ids", None)
        
        device = self._get_model_input_device()
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.02,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        decoded: list[str] = []
        for i, input_len in enumerate(input_lengths):
            new_tokens = outputs[i][int(input_len):]
            decoded.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        result: list[str] = []
        decoded_iter = iter(decoded)
        for original in texts:
            if (original or "").strip():
                result.append(next(decoded_iter))
            else:
                result.append("")

        return result

    def translate(self, source_text: str, target_language: str = "Russian") -> str:
        return self.translate_batch([source_text], target_language=target_language)[0]


@lru_cache(maxsize=1)
def get_translator() -> HyMtTranslator:
    return HyMtTranslator()