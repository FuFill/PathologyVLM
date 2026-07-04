"""Vision-language model loading and inference helpers for Quilt-LLaVA.

Loads ``wisdomik/Quilt-Llava-v1.5-7b`` (and other original-LLaVA-format
checkpoints) via the upstream ``llava`` package (Quilt-LLaVA fork of
LLaVA-1.5). The upstream loader is required because:

* The HF checkpoint uses the original ``LlavaLlamaForCausalLM`` naming
  scheme (``model.vision_tower.vision_tower.vision_model.*``,
  ``model.mm_projector.0/2.*``), which is NOT compatible with
  ``transformers.LlavaForConditionalGeneration`` (which expects
  ``multi_modal_projector.linear_1/2`` and ``language_model.model.*``).
* No ``preprocessor_config.json`` is shipped with the model; the CLIP
  vision tower from ``mm_vision_tower`` carries the image preprocessor
  and is set up automatically by ``load_pretrained_model``.

The ``llava`` package is installed at runtime by ``run_remote_vlm.py``
with ``--no-deps`` (its setup.py pins torch==2.0.1 which would conflict
with the rest of our requirements).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .image_utils import safe_open_rgb


def load_model(
    model_name: str,
    load_4bit: bool,
):
    """Load a Quilt-LLaVA / LLaVA-1.5 model via the upstream loader.

    Parameters
    ----------
    model_name : str
        Hugging Face model id, e.g. ``wisdomik/Quilt-Llava-v1.5-7b``.
        Must contain the substring ``llava`` (case-insensitive) so the
        upstream builder selects the LLaVA branch.
    load_4bit : bool
        If True, use bitsandbytes 4-bit quantization (nf4, double-quant,
        fp16 compute) to reduce GPU memory.

    Returns
    -------
    (tokenizer, model, image_processor, context_len)
    """
    # Imported lazily so that ``import src.vlm_inference`` does not fail
    # on machines that have not yet bootstrapped the ``llava`` package.
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    cuda_available = torch.cuda.is_available()

    if load_4bit and not cuda_available:
        print(
            "[vlm_inference] WARNING: load_4bit=True but CUDA is not available. "
            "bitsandbytes 4-bit requires a GPU. Falling back to fp16 on CPU "
            "(this will likely fail; intended for environment probing only)."
        )
        load_4bit = False

    short_name = get_model_name_from_path(model_name)
    print(f"[vlm_inference] Loading model: {model_name} (short_name={short_name!r})")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_name,
        model_base=None,
        model_name=short_name,
        load_8bit=False,
        load_4bit=bool(load_4bit),
        device_map="auto" if cuda_available else None,
        device="cuda" if cuda_available else "cpu",
    )
    model.eval()

    print(
        f"[vlm_inference] Loaded OK. context_len={context_len} "
        f"image_size={getattr(image_processor, 'size', None)}"
    )
    return tokenizer, model, image_processor, context_len


def generate_answer(
    image_path: str | Path,
    tokenizer,
    model,
    image_processor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float = 1.08,
) -> str:
    """Run inference on a single image and return the raw decoded text.

    Uses the LLaVA-1.5 ``vicuna_v1`` / ``llava_v1`` conversation template
    (the default for any model with ``v1`` in its name in the upstream
    Quilt-LLaVA CLI), prepending the ``<image>`` token to the user turn
    as required by ``tokenizer_image_token``.

    Note: Prefills the ASSISTANT turn with `{` to prevent conversational
    prose fallback and guarantee 100% JSON grammar compliance.
    """
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )

    image = safe_open_rgb(image_path)

    # --- Build the conversation prompt -----------------------------------
    # Quilt-LLaVA / LLaVA-1.5-7b uses the "llava_v1" template (vicuna-style
    # USER:/ASSISTANT: with the </s> stop token).
    conv = conv_templates["llava_v1"].copy()

    if getattr(model.config, "mm_use_im_start_end", False):
        user_msg = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + prompt
        )
    else:
        user_msg = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv.append_message(conv.roles[0], user_msg)
    # Prefill the assistant turn with `{` to lock token generation into strict JSON syntax
    conv.append_message(conv.roles[1], "{")
    full_prompt = conv.get_prompt()

    # --- Image tensor ----------------------------------------------------
    # process_images honours model.config.image_aspect_ratio == 'pad'
    # (which Quilt-LLaVA uses). Returns either a stacked tensor or a list.
    class _ImgCfg:
        image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")

    image_tensor = process_images([image], image_processor, _ImgCfg())
    target_device = next(model.parameters()).device
    if isinstance(image_tensor, list):
        image_tensor = [t.to(target_device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(target_device, dtype=torch.float16)

    # --- Text tensor with <image> token replaced by IMAGE_TOKEN_INDEX ----
    input_ids = (
        tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(target_device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs: dict = {
        "images": image_tensor,
        "do_sample": do_sample,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 0.95

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # The upstream LLaVA model returns ONLY the new tokens (input is
    # consumed by the multimodal projector pathway), so we decode from
    # position 0. Defensive: if shape matches input length + new, slice.
    if output_ids.shape[1] >= input_ids.shape[1] and torch.equal(
        output_ids[:, : input_ids.shape[1]].cpu(), input_ids.cpu()
    ):
        new_tokens = output_ids[:, input_ids.shape[1] :]
    else:
        new_tokens = output_ids

    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    # Re-attach the prefilled opening brace
    decoded = "{" + decoded.strip()

    # Strip the stop string if the model emitted it.
    if stop_str and decoded.endswith(stop_str):
        decoded = decoded[: -len(stop_str)]

    return decoded.strip()
