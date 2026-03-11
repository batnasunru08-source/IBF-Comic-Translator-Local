from __future__ import annotations

import logging
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
MODEL_ID = "tencent/HY-MT1.5-7B"


class HyMtTranslator:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = self._load_model()

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
                    torch_dtype=torch.bfloat16,
                    **common_kwargs,
                )
            except Exception as exc:
                logger.warning("4-bit load failed, fallback to non-quantized CUDA load: %s", exc)
                return AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                    **common_kwargs,
                )

        logger.info("CUDA not available, loading on CPU")
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, **common_kwargs)

    def _get_model_input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def translate(self, source_text: str, target_language: str = "Russian") -> str:
        source_text = (source_text or "").strip()
        if not source_text:
            return ""

        prompt = (
            f"Translate the following segment into {target_language}, "
            "without additional explanation.\n\n"
            f"{source_text}"
        )

        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )

        device = self._get_model_input_device()
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        input_length = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.02,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_translator() -> HyMtTranslator:
    return HyMtTranslator()