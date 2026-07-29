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
        self._revision = None
        self._quantization = None

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
        self._revision = revision
        self._quantization = "4bit-nf4-double" if load_4bit else "none"

    def config_snapshot(self) -> dict:
        return {
            "model_id": self.model_id(),
            "revision": self._revision,
            "quantization": self._quantization,
        }

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

        prompt_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        print(f"[med_gemma] prompt_text:\n{prompt_text}")

        inputs = self._processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        print(f"[med_gemma] inputs keys: {list(inputs.keys())}")
        print(f"[med_gemma] input_ids shape: {inputs['input_ids'].shape}")
        print(f"[med_gemma] input_ids decoded:\n{self._processor.tokenizer.decode(inputs['input_ids'][0])}")
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            print(f"[med_gemma] pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
        if "pixel_attention_mask" in inputs:
            print(f"[med_gemma] pixel_attention_mask shape: {inputs['pixel_attention_mask'].shape}")

        inputs = inputs.to(self._model.device)
        inputs.pop("token_type_ids", None)

        for key in ("pixel_values", "pixel_attention_mask"):
            if key in inputs and inputs[key].dtype != torch.float16:
                inputs[key] = inputs[key].to(dtype=torch.float16)

        if seed is not None:
            set_seed(seed)

        do_sample = temperature > 0.0
        gen_kwargs = {
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
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        print(f"[med_gemma] output_ids shape: {output_ids.shape}")
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        print(f"[med_gemma] new_tokens shape: {new_tokens.shape}")
        print(f"[med_gemma] new_tokens ids: {new_tokens.tolist()}")
        decoded = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        print(f"[med_gemma] decoded: |{decoded}|")
        return decoded.strip()
