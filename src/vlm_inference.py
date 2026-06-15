"""Vision-language model loading and inference helpers.

Targeted at LLaVA-style models such as ``wisdomik/Quilt-Llava-v1.5-7b``.

The functions are intentionally thin wrappers over Hugging Face Transformers
so that loading errors propagate clearly to the ClearML logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from .image_utils import safe_open_rgb


def build_llava_prompt(prompt: str) -> str:
    """Build a single-turn LLaVA-style prompt string.

    Returns a string of the form::

        USER: <image>
        {prompt}
        ASSISTANT:
    """
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def load_model(
    model_name: str,
    load_4bit: bool,
) -> Tuple[AutoProcessor, LlavaForConditionalGeneration]:
    """Load a LLaVA-style VLM and its processor.

    Parameters
    ----------
    model_name : str
        Hugging Face model id, e.g. ``wisdomik/Quilt-Llava-v1.5-7b``.
    load_4bit : bool
        If True, use bitsandbytes 4-bit quantization to reduce GPU memory.

    Returns
    -------
    (processor, model)
    """
    cuda_available = torch.cuda.is_available()
    dtype = torch.float16 if cuda_available else torch.float32

    print(f"[vlm_inference] Loading processor for: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs: dict = {
        "device_map": "auto",
        "torch_dtype": dtype,
    }

    if load_4bit:
        if not cuda_available:
            print(
                "[vlm_inference] WARNING: load_4bit=True but CUDA is not available. "
                "bitsandbytes 4-bit requires a GPU. Falling back to no quantization."
            )
        else:
            print("[vlm_inference] Using 4-bit quantization via bitsandbytes.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

    print(f"[vlm_inference] Loading model: {model_name}")
    # NOTE: If the upstream QUILT-LLaVA repository requires a custom class
    # (e.g. a LlavaLlama variant), this call will raise a clear error from
    # transformers. We intentionally do not silently fall back.
    model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return processor, model


def generate_answer(
    image_path: str | Path,
    processor: AutoProcessor,
    model: LlavaForConditionalGeneration,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run inference on a single image and return the raw decoded text.

    The ``ASSISTANT:`` prefix is stripped if present so callers receive
    only the model's reply.
    """
    image = safe_open_rgb(image_path)
    full_prompt = build_llava_prompt(prompt)

    inputs = processor(images=image, text=full_prompt, return_tensors="pt")

    # Move tensors to the model device. With device_map="auto" the model
    # may have its first parameter on cuda:0 (or cpu).
    try:
        target_device = next(model.parameters()).device
        inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    except StopIteration:
        # Model has no parameters? Leave tensors as-is.
        pass

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs: dict = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Strip everything up to and including the last 'ASSISTANT:' marker so
    # we return only the model's reply, not the echoed prompt.
    marker = "ASSISTANT:"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[-1]

    return decoded.strip()
