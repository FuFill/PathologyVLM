from __future__ import annotations

from typing import Optional

import sys
import torch
from PIL import Image

from .base import VLMBackend


def _dbg(*args, **kwargs):
    print("[med_gemma]", *args, file=sys.stderr, **kwargs)


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
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            token=True,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._revision = revision
        self._quantization = "4bit-nf4-double" if load_4bit else "none"

        proc_id = self._processor.image_token_id
        model_id = getattr(self._model.config, 'image_token_index',
                   getattr(self._model.config, 'image_token_id', None))
        if proc_id != model_id:
            _dbg(f"Aligning image_token_id: processor={proc_id} model={model_id} -> {proc_id}")
            self._model.config.image_token_index = proc_id

        embed_weight = self._model.get_input_embeddings().weight
        _dbg(f"embed.weight: shape={list(embed_weight.shape)}, "
             f"min={embed_weight.min().item():.4f}, max={embed_weight.max().item():.4f}, "
             f"mean={embed_weight.mean().item():.4f}")

        lang = getattr(self._model, 'language_model', None)
        if lang is not None and hasattr(lang, 'lm_head'):
            lm_head = lang.lm_head
            _dbg(f"lm_head.weight (via language_model): shape={list(lm_head.weight.shape)}, "
                 f"min={lm_head.weight.min().item():.4f}, max={lm_head.weight.max().item():.4f}, "
                 f"mean={lm_head.weight.mean().item():.4f}")
            tied = lm_head.weight.data_ptr() == embed_weight.data_ptr()
            _dbg(f"lm_head tied to embed: {tied}")
        else:
            _dbg(f"WARNING: language_model.lm_head not found (attrs: {[a for a in dir(self._model) if not a.startswith('_')]})")

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

        padded = [_pad_to_siglip(img) for img in images]

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

        formatted_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        _dbg(f"formatted_text (first 150): {formatted_text[:150]!r}")

        model_inputs = self._processor(
            text=formatted_text,
            images=padded,
            return_tensors="pt",
            padding=True,
        )
        _dbg(f"processor keys: {list(model_inputs.keys())}")

        model_inputs = model_inputs.to(self._model.device)
        for key in ("pixel_values", "pixel_attention_mask"):
            if key in model_inputs:
                model_inputs[key] = model_inputs[key].to(dtype=torch.bfloat16)

        img_tok_id = getattr(self._model.config, 'image_token_index',
                     getattr(self._model.config, 'image_token_id', None))
        n_img = (model_inputs["input_ids"] == img_tok_id).sum().item()
        _dbg(f"image_token_id={img_tok_id}, count in input_ids={n_img}, "
             f"input_ids shape={model_inputs['input_ids'].shape}")
        _dbg(f"has token_type_ids: {'token_type_ids' in model_inputs}")

        if "pixel_values" in model_inputs:
            pv = model_inputs["pixel_values"]
            _dbg(f"pixel_values: dtype={pv.dtype}, shape={pv.shape}, "
                 f"min={pv.min().item():.4f}, max={pv.max().item():.4f}, "
                 f"mean={pv.mean().item():.4f}")

        proj = None
        for _name, _mod in self._model.named_modules():
            if hasattr(_mod, 'mm_input_projection_weight'):
                proj = _mod.mm_input_projection_weight
                break
        if proj is not None:
            _dbg(f"projector weight from '{_name}': dtype={proj.dtype}, shape={proj.shape}, "
                 f"min={proj.min().item():.6f}, max={proj.max().item():.6f}, "
                 f"mean={proj.mean().item():.6f}")
        else:
            _dbg("projector weight NOT FOUND")

        with torch.inference_mode():
            img_feat = self._model.get_image_features(
                pixel_values=model_inputs["pixel_values"]
            )
            if hasattr(img_feat, 'pooler_output'):
                img_feat = img_feat.pooler_output
            _dbg(f"image_features: dtype={img_feat.dtype}, shape={img_feat.shape}, "
                 f"min={img_feat.min().item():.4f}, max={img_feat.max().item():.4f}, "
                 f"mean={img_feat.mean().item():.4f}, has_nan={torch.isnan(img_feat).any().item()}")

        input_embeds = self._model.get_input_embeddings()(model_inputs["input_ids"])
        _dbg(f"input_embeds before masked_scatter: "
             f"min={input_embeds.min().item():.4f}, max={input_embeds.max().item():.4f}, "
             f"mean={input_embeds.mean().item():.4f}")

        if "attention_mask" in model_inputs:
            am = model_inputs["attention_mask"]
            _dbg(f"attention_mask: dtype={am.dtype}, shape={am.shape}, "
                 f"min={am.min().item()}, max={am.max().item()}, sum={am.sum().item()}")

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

        # ── Diagnostic forward pass before generate ──
        _dbg("Running diagnostic forward pass...")
        with torch.inference_mode():
            diag_out = self._model(
                **model_inputs,
                logits_to_keep=1,
                return_dict=True,
            )
        diag_logits = diag_out.logits  # [1, 1, vocab_size]
        diag_last = diag_logits[0, -1]
        diag_argmax = diag_last.argmax().item()
        top5_vals, top5_ids = diag_last.topk(5)
        _dbg(f"diagnostic forward logits[-1]: argmax={diag_argmax}, "
             f"top5_ids={top5_ids.tolist()}, "
             f"top5_vals={[f'{v:.4f}' for v in top5_vals.tolist()]}")
        del diag_out, diag_logits

        with torch.inference_mode():
            output_ids = self._model.generate(**model_inputs, **gen_kwargs)

        _dbg(f"output_ids shape: {output_ids.shape}")
        input_len = model_inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        _dbg(f"new_tokens ids ({new_tokens.shape[1]} tokens): {new_tokens.tolist()}")

        all_zero = (new_tokens == 0).all().item()
        if all_zero:
            _dbg("ALL NEW TOKENS ARE 0 — retrying with attention_mask=None")
            gen_inputs = {k: v for k, v in model_inputs.items() if k != "attention_mask"}
            with torch.inference_mode():
                output_ids = self._model.generate(**gen_inputs, **gen_kwargs)
            new_tokens = output_ids[:, input_len:]
            _dbg(f"retry new_tokens ids: {new_tokens.tolist()}")

        decoded = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        _dbg(f"decoded: |{decoded}|")

        return decoded.strip()
