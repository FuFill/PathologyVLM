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
    try:
        import importlib

        importlib.import_module("llava")
        already = True
    except ImportError:
        already = False

    if not already:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            QUILT_LLAVA_GIT,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
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

    @staticmethod
    def model_id() -> str:
        return "wisdomik/Quilt-Llava-v1.5-7b"

    def load(self, load_4bit: bool = False, revision: Optional[str] = None) -> None:
        _bootstrap_llava()

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

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
        conv.append_message(conv.roles[1], "{")
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
        decoded = "{" + decoded.strip()

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
