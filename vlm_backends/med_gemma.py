from __future__ import annotations

from typing import Optional

import torch
from PIL import Image

from .base import VLMBackend


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MedGemmaBackend(VLMBackend):
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    @staticmethod
    def model_id() -> str:
        return "google/medgemma-1.5-4b-it"

    def load(self, load_4bit: bool = False, revision: Optional[str] = None) -> None:
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Gemma3ForConditionalGeneration,
        )

        quantization_config = None
        if load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id(),
            revision=revision,
            token=True,
            trust_remote_code=True,
            use_fast=True,
        )

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id(),
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            token=True,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        model_inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        print(f"[DEBUG] model_inputs keys: {list(model_inputs.keys())}")
        print(f"[DEBUG] input_ids shape: {model_inputs['input_ids'].shape}")
        print(f"[DEBUG] input_ids: {model_inputs['input_ids'][0].tolist()}")
        if "pixel_values" in model_inputs:
            print(f"[DEBUG] pixel_values shape: {model_inputs['pixel_values'].shape}")
        else:
            print(f"[DEBUG] WARNING: no pixel_values in model_inputs!")

        model_inputs = model_inputs.to(self._model.device)
        for key in ("pixel_values", "pixel_attention_mask"):
            if key in model_inputs:
                model_inputs[key] = model_inputs[key].to(dtype=torch.float16)

        model_inputs.pop("token_type_ids", None)

        if seed is not None:
            set_seed(seed)

        do_sample = temperature > 0.0
        gen_kwargs = {
            "pad_token_id": self._processor.tokenizer.eos_token_id,
            "do_sample": do_sample,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95
        if repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        with torch.inference_mode():
            output_ids = self._model.generate(**model_inputs, **gen_kwargs)

        input_len = model_inputs["input_ids"].shape[1]
        print(f"[DEBUG] output_ids shape: {output_ids.shape}, input_len: {input_len}")
        new_tokens = output_ids[:, input_len:]
        print(f"[DEBUG] new_tokens shape: {new_tokens.shape}, new_tokens: {new_tokens[0].tolist()}")
        decoded = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        print(f"[DEBUG] decoded: {repr(decoded)}")

        return decoded.strip()
