from __future__ import annotations

import sys
import traceback
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


_siglip_target_size: int = 896


def _pad_to_siglip(img: Image.Image, target: int = _siglip_target_size) -> Image.Image:
    w, h = img.size
    if w == target and h == target:
        return img
    new = Image.new("RGB", (target, target), (0, 0, 0))
    left = (target - w) // 2
    top = (target - h) // 2
    new.paste(img, (left, top))
    return new


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

        try:
            return self._generate(
                images=images,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        except Exception:
            tb = traceback.format_exc()
            print(f"[med_gemma] CRASH in _generate:\n{tb}", flush=True)
            return f"CRASH: {tb[:200]}"

    def _generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        seed: Optional[int],
    ) -> str:
        padded = [_pad_to_siglip(img) for img in images]
        for i, (orig, p) in enumerate(zip(images, padded)):
            print(f"[med_gemma] img{i}: {orig.size} -> {p.size}", flush=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in padded
                ] + [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        print(f"[med_gemma] text_prompt:\n{text_prompt}", flush=True)

        model_inputs = self._processor(
            text=text_prompt,
            images=padded,
            return_tensors="pt",
            padding=True,
        )
        print(f"[med_gemma] model_inputs keys: {list(model_inputs.keys())}", flush=True)
        for key, val in model_inputs.items():
            if hasattr(val, "shape"):
                print(f"[med_gemma]   {key}: shape={val.shape}, dtype={val.dtype}", flush=True)
            else:
                print(f"[med_gemma]   {key}: {type(val)}", flush=True)

        model_inputs = model_inputs.to(self._model.device)
        for key in ("pixel_values", "pixel_attention_mask"):
            if key in model_inputs:
                model_inputs[key] = model_inputs[key].to(dtype=torch.float16)

        model_inputs.pop("token_type_ids", None)

        print(f"[med_gemma] device: {self._model.device}", flush=True)
        print(f"[med_gemma] input_ids shape: {model_inputs['input_ids'].shape}", flush=True)
        print(f"[med_gemma] pad_token_id: tokenizer={self._processor.tokenizer.pad_token_id}, eos={self._processor.tokenizer.eos_token_id}", flush=True)
        print(f"[med_gemma] config pad: {getattr(self._model.config, 'pad_token_id', None)}, eos: {getattr(self._model.config, 'eos_token_id', None)}", flush=True)
        last5 = model_inputs["input_ids"][0, -5:].tolist()
        print(f"[med_gemma] input_ids last5: {last5}", flush=True)
        decoded_last = self._processor.tokenizer.decode(last5)
        print(f"[med_gemma] decoded last5: {repr(decoded_last)}", flush=True)

        if seed is not None:
            set_seed(seed)

        do_sample = temperature > 0.0
        gen_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
        }
        gen_kwargs["pad_token_id"] = self._processor.tokenizer.eos_token_id
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95
        if repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        print(f"[med_gemma] gen_kwargs: {gen_kwargs}", flush=True)

        with torch.inference_mode():
            output_ids = self._model.generate(**model_inputs, **gen_kwargs)

        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        print(f"[med_gemma] output_ids shape: {output_ids.shape}", flush=True)
        print(f"[med_gemma] new_tokens len: {new_tokens.shape[1]}", flush=True)
        n_first = min(20, new_tokens.shape[1])
        if n_first > 0:
            print(f"[med_gemma] new_tokens first {n_first}: {new_tokens[0, :n_first].tolist()}", flush=True)

        decoded = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]

        print(f"[med_gemma] decoded: {repr(decoded)}", flush=True)
        return decoded.strip()
