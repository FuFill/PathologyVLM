from __future__ import annotations

import random
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import VLMBackend


QUILT_LLAVA_GIT = (
    "git+https://github.com/aldraus/quilt-llava"
    "@7e70fc39f792ac55de010eb37bff0a6d6f491c13"
)


def _free_transformers_llava_slot() -> None:
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )
    except Exception:
        return
    CONFIG_MAPPING_NAMES.pop("llava", None)
    extra = getattr(CONFIG_MAPPING, "_extra_content", {})
    if isinstance(extra, dict):
        extra.pop("llava", None)


def _stub_llava_mpt() -> None:
    mpt_stub = types.ModuleType("llava.model.language_model.llava_mpt")

    class _LlavaMPTConfig:
        model_type = "llava_mpt_stub"

    class _LlavaMPTForCausalLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LlavaMPTForCausalLM is stubbed out in this environment."
            )

    mpt_stub.LlavaMPTForCausalLM = _LlavaMPTForCausalLM
    mpt_stub.LlavaMPTConfig = _LlavaMPTConfig
    sys.modules["llava.model.language_model.llava_mpt"] = mpt_stub


def _bootstrap_llava() -> None:
    import subprocess

    print(f"[quilt_llava] Installing pinned llava fork: {QUILT_LLAVA_GIT}")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
        "--force-reinstall",
        QUILT_LLAVA_GIT,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"pip install of llava failed (exit {res.returncode})")

    for key in list(sys.modules):
        if key == "llava" or key.startswith("llava."):
            del sys.modules[key]

    _free_transformers_llava_slot()
    _stub_llava_mpt()

    import importlib

    importlib.invalidate_caches()
    import llava  # noqa: F401

    print(f"[quilt_llava] llava importable from: {llava.__file__}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _SafeRepetitionPenaltyLogitsProcessor(torch.nn.Module):
    def __init__(self, penalty: float):
        super().__init__()
        self.penalty = penalty

    def __call__(
        self, input_ids_tensor: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        safe_ids = input_ids_tensor.clone()
        safe_ids[safe_ids < 0] = 0
        score = torch.gather(scores, 1, safe_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, safe_ids, score)
        return scores


class QuiltLLaVABackend(VLMBackend):
    requires_cuda = True

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._device = None
        self._revision = None
        self._diagnostics_printed = False

    @staticmethod
    def model_id() -> str:
        return "wisdomik/Quilt-Llava-v1.5-7b"

    def load(self, load_4bit: bool = False, revision: Optional[str] = None) -> None:
        _bootstrap_llava()

        import transformers

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        print(
            f"[quilt_llava] env: torch={torch.__version__} "
            f"transformers={transformers.__version__} cuda={torch.cuda.is_available()}"
        )

        model_name = get_model_name_from_path(self.model_id())
        self._tokenizer, self._model, self._image_processor, context_len = (
            load_pretrained_model(
                model_path=self.model_id(),
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=load_4bit,
                device_map="auto" if torch.cuda.is_available() else None,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._revision = revision or getattr(self._model.config, '_commit_hash', None)

        state_keys = list(self._model.state_dict().keys())
        n_double = sum(
            1 for k in state_keys
            if k.startswith("model.vision_tower.vision_tower.vision_model")
        )
        n_single = sum(
            1 for k in state_keys
            if k.startswith("model.vision_tower.vision_model")
            and not k.startswith("model.vision_tower.vision_tower")
        )
        n_vt_total = sum(1 for k in state_keys if k.startswith("model.vision_tower"))
        print(
            f"[quilt_llava] vision tower keys in model: "
            f"double_vision_tower={n_double} single_vision_tower={n_single} "
            f"vision_tower_total={n_vt_total}"
        )
        if n_double == 0 and n_vt_total > 0:
            raise RuntimeError(
                "[quilt_llava] vision tower structure mismatch: checkpoint uses "
                "model.vision_tower.vision_tower.vision_model.* (double) but loaded "
                "model has single-level vision tower -> weights were silently dropped. "
                "The pinned aldraus fork is NOT what got imported. "
                "Check llava.__file__ and the pip install log above."
            )
        print(
            f"[quilt_llava] Loaded OK. context_len={context_len} "
            f"image_size={getattr(self._image_processor, 'size', None)} "
            f"device={self._device} dtype={next(self._model.parameters()).dtype}"
        )

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if len(images) != 1:
            raise ValueError("Quilt-LLaVA supports only single-image input.")

        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (
            KeywordsStoppingCriteria,
            process_images,
            tokenizer_image_token,
        )

        conv = conv_templates["llava_v1"].copy()

        mm_use_im_start_end = getattr(self._model.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
            user_msg = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
            )
        else:
            user_msg = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        class _ImgCfg:
            image_aspect_ratio = getattr(self._model.config, "image_aspect_ratio", "pad")

        image_tensor = process_images(images, self._image_processor, _ImgCfg())
        if isinstance(image_tensor, list):
            image_tensor = [t.to(self._device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(self._device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(
                full_prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self._device)
        )

        if not self._diagnostics_printed:
            self._diagnostics_printed = True
            try:
                if isinstance(image_tensor, list):
                    diag_t = image_tensor[0]
                else:
                    diag_t = image_tensor

                print(f"[quilt_llava] full_prompt: {full_prompt!r}")
                print(
                    f"[quilt_llava] config: mm_use_im_start_end="
                    f"{getattr(self._model.config, 'mm_use_im_start_end', None)} "
                    f"image_aspect_ratio="
                    f"{getattr(self._model.config, 'image_aspect_ratio', None)} "
                    f"mm_vision_tower={getattr(self._model.config, 'mm_vision_tower', None)}"
                )

                n_img_tokens = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
                print(
                    f"[quilt_llava] input_ids: dtype={input_ids.dtype} "
                    f"shape={tuple(input_ids.shape)} n_image_token_index={n_img_tokens} "
                    f"min={int(input_ids.min().item())} max={int(input_ids.max().item())}"
                )
                if n_img_tokens == 0:
                    print(
                        "[quilt_llava] WARNING: input_ids has NO IMAGE_TOKEN_INDEX (-200). "
                        "The image will be silently DROPPED by the fork's "
                        "prepare_inputs_labels_for_multimodal hacky-fix branch -> "
                        "model runs text-only -> constant answers!"
                    )

                print(
                    f"[quilt_llava] pixel_values: dtype={diag_t.dtype} "
                    f"shape={tuple(diag_t.shape)} min={diag_t.min().item():.4f} "
                    f"max={diag_t.max().item():.4f} mean={diag_t.mean().item():.4f} "
                    f"has_nan={bool(torch.isnan(diag_t).any().item())}"
                )
                with torch.inference_mode():
                    image_features = self._model.encode_images(diag_t)
                print(
                    f"[quilt_llava] image_features: dtype={image_features.dtype} "
                    f"shape={tuple(image_features.shape)} min={image_features.min().item():.4f} "
                    f"max={image_features.max().item():.4f} mean={image_features.mean().item():.4f} "
                    f"has_nan={bool(torch.isnan(image_features).any().item())}"
                )

                def _top5(logits: torch.Tensor) -> str:
                    vals, idx = logits.float().flatten().topk(5)
                    return (
                        f"argmax={int(idx[0].item())} "
                        f"top5_ids={[int(i) for i in idx.tolist()]} "
                        f"top5_vals={[f'{v:.4f}' for v in vals.tolist()]}"
                    )

                with torch.inference_mode():
                    out_img = self._model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        images=image_tensor,
                        use_cache=False,
                    )
                logits_img = out_img.logits[0, -1]
                print(f"[quilt_llava] diagnostic forward WITH image logits[-1]: {_top5(logits_img)}")

                ids_text = input_ids.clone()
                ids_text[ids_text < 0] = 0  # -200 -> pad, so the text-only path is valid
                with torch.inference_mode():
                    out_text = self._model(
                        input_ids=ids_text,
                        attention_mask=torch.ones_like(ids_text),
                        images=None,
                        use_cache=False,
                    )
                logits_text = out_text.logits[0, -1]
                print(f"[quilt_llava] diagnostic forward WITHOUT image logits[-1]: {_top5(logits_text)}")

                diff = (logits_img - logits_text).abs().max().item()
                print(
                    f"[quilt_llava] max_logit_diff_image_vs_text={diff:.6f} "
                    f"({'IMAGE IS USED' if diff > 0.01 else 'IMAGE IS IGNORED -> root cause'})"
                )
            except Exception as exc:  # noqa: BLE001 — diagnostics must not break inference
                print(f"[quilt_llava] WARNING: image diagnostics failed: {type(exc).__name__}: {exc}")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self._tokenizer, input_ids)

        logits_processors = []
        if repetition_penalty > 1.0:
            logits_processors.append(
                _SafeRepetitionPenaltyLogitsProcessor(float(repetition_penalty))
            )

        do_sample = temperature > 0.0
        gen_kwargs: dict = {
            "images": image_tensor,
            "do_sample": do_sample,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "stopping_criteria": [stopping_criteria],
        }
        if logits_processors:
            from transformers.generation.logits_process import LogitsProcessorList
            gen_kwargs["logits_processor"] = LogitsProcessorList(logits_processors)
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.95

        if seed is not None:
            set_seed(seed)

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        if output_ids.shape[1] >= input_ids.shape[1] and torch.equal(
            output_ids[:, : input_ids.shape[1]].cpu(), input_ids.cpu()
        ):
            new_tokens = output_ids[:, input_ids.shape[1] :]
        else:
            new_tokens = output_ids

        decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        decoded = decoded.strip()

        if stop_str and decoded.endswith(stop_str):
            decoded = decoded[: -len(stop_str)]

        return decoded.strip()

    def config_snapshot(self) -> dict:
        import transformers
        return {
            "model_id": self.model_id(),
            "revision": self._revision,
            "quantization": "4bit" if getattr(self, '_load_4bit', False) else "none",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(self._device) if self._device else "unknown",
            "dtype": str(next(self._model.parameters()).dtype) if self._model is not None else "unknown",
        }
