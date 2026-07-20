"""Quilt-LLaVA model bootstrap, loading, and single-image inference.

Self-contained: no dependency on the parent research repo. All the subtle
parts of getting ``wisdomik/Quilt-Llava-v1.5-7b`` to load and generate strict
JSON live here.

Why the ``llava`` bootstrap is needed
-------------------------------------
The Quilt-LLaVA checkpoint uses the *original* LLaVA weight-naming scheme
(``model.vision_tower...``, ``model.mm_projector.0/2``), which
``transformers.LlavaForConditionalGeneration`` cannot load. So the upstream
``llava`` package (Quilt-LLaVA fork) is required at runtime. It is NOT a normal
pip dependency because its setup.py pins ``torch==2.0.1`` (conflicts with our
torch). We therefore:
  1. pip-install it ``--no-deps`` at runtime,
  2. free the transformers auto-mapping ``llava`` slot (else it shadows the
     package), and
  3. stub out ``llava_mpt`` (incompatible with transformers>=4.36),
before importing ``llava``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError

# Pinned upstream Quilt-LLaVA commit (matches the production baseline run).
QUILT_LLAVA_GIT = (
    "git+https://github.com/aldraus/quilt-llava"
    "@7e70fc39f792ac55de010eb37bff0a6d6f491c13"
)

# Pinned Hugging Face weights revision. ``None`` means the loader takes the repo
# default (current ``main``); set this to a specific commit sha / tag to pin the
# exact weights used for a run and make it fully reproducible.
MODEL_REVISION: Optional[str] = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible sampling.

    Makes a sampled (temperature>0) run repeatable given the same model, inputs
    and kernels; does NOT make temperature>0 behave like greedy decoding. For a
    deterministic control run use temperature=0.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001 — numpy optional
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Image discovery / loading
# --------------------------------------------------------------------------- #
def find_images(image_dir: str | Path, max_images: int) -> list[Path]:
    """Recursively find supported image files under ``image_dir`` (sorted)."""
    base = Path(image_dir)
    if not base.exists():
        raise FileNotFoundError(f"Image directory does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {base}")
    results = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    results.sort(key=lambda p: str(p).lower())
    if max_images is not None and max_images > 0:
        results = results[:max_images]
    return results


def safe_open_rgb(path: str | Path) -> Image.Image:
    """Open an image and convert to RGB, with clear errors on failure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: {p}")
    try:
        img = Image.open(p)
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Failed to open image '{p}': {exc}") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# --------------------------------------------------------------------------- #
# llava bootstrap
# --------------------------------------------------------------------------- #
def _free_transformers_llava_slot() -> None:
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[quilt_vlm] WARNING: could not free transformers 'llava' slot: {exc}")
        return
    CONFIG_MAPPING_NAMES.pop("llava", None)
    extra = getattr(CONFIG_MAPPING, "_extra_content", {})
    if isinstance(extra, dict):
        extra.pop("llava", None)
    print("[quilt_vlm] Freed transformers 'llava' slot")


def _stub_llava_mpt() -> None:
    import types

    mpt_stub = types.ModuleType("llava.model.language_model.llava_mpt")

    class _LlavaMPTConfig:
        model_type = "llava_mpt_stub"

    class _LlavaMPTForCausalLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LlavaMPTForCausalLM is stubbed out (MPT path needs transformers<4.36)."
            )

    mpt_stub.LlavaMPTForCausalLM = _LlavaMPTForCausalLM
    mpt_stub.LlavaMPTConfig = _LlavaMPTConfig
    sys.modules["llava.model.language_model.llava_mpt"] = mpt_stub
    print("[quilt_vlm] Stubbed llava.model.language_model.llava_mpt")


def bootstrap_llava() -> None:
    """Ensure the upstream ``llava`` package is importable. Idempotent."""
    try:
        import importlib

        importlib.import_module("llava")
        already = True
    except ImportError:
        already = False

    if not already:
        import subprocess

        print(f"[quilt_vlm] Installing llava (--no-deps) from {QUILT_LLAVA_GIT}")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-deps", "--no-cache-dir", QUILT_LLAVA_GIT,
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

    print(f"[quilt_vlm] llava importable from: {llava.__file__}")


# --------------------------------------------------------------------------- #
# Model load + generation
# --------------------------------------------------------------------------- #
def load_model(model_name: str, load_4bit: bool, revision: Optional[str] = None):
    """Load a Quilt-LLaVA / LLaVA-1.5 model via the upstream loader.

    ``revision`` pins the Hugging Face weights commit/tag. The upstream
    ``load_pretrained_model`` does NOT forward a ``revision`` argument, so we
    enforce the pin by pre-downloading that exact revision into the HF cache
    (via ``huggingface_hub.snapshot_download``) before the loader resolves the
    repo. If the download step is unavailable we fall back to the default
    revision but still record the requested value on every output row so the
    discrepancy is auditable.

    Returns (tokenizer, model, image_processor, context_len).
    """
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    cuda_available = torch.cuda.is_available()
    if load_4bit and not cuda_available:
        print("[quilt_vlm] WARNING: load_4bit=True but no CUDA; disabling 4bit.")
        load_4bit = False

    if revision:
        try:
            from huggingface_hub import snapshot_download

            print(f"[quilt_vlm] Pinning weights revision {revision} via snapshot_download")
            snapshot_download(repo_id=model_name, revision=revision)
        except Exception as exc:  # noqa: BLE001
            print(f"[quilt_vlm] WARNING: could not pin revision {revision}: {exc} "
                  "(loading default revision; recorded value may not match weights).")

    short_name = get_model_name_from_path(model_name)
    print(f"[quilt_vlm] Loading model: {model_name} (short_name={short_name!r})")
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
    print(f"[quilt_vlm] Loaded OK. context_len={context_len}")
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
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> str:
    """Run inference on a single image; return the raw decoded text.

    Uses the LLaVA-1.5 ``llava_v1`` conversation template, prepending the
    ``<image>`` token. The assistant turn is prefilled with ``{`` to force
    JSON-only output. A custom repetition-penalty processor clamps LLaVA's
    negative IMAGE_TOKEN_INDEX (-200) before gathering (the stock transformers
    processor crashes with a CUDA device-side assert on it).
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

    conv = conv_templates["llava_v1"].copy()
    if getattr(model.config, "mm_use_im_start_end", False):
        user_msg = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            + "\n" + prompt
        )
    else:
        user_msg = DEFAULT_IMAGE_TOKEN + "\n" + prompt

    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], "{")  # prefill to lock JSON syntax
    full_prompt = conv.get_prompt()

    class _ImgCfg:
        image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")

    image_tensor = process_images([image], image_processor, _ImgCfg())
    target_device = next(model.parameters()).device
    if isinstance(image_tensor, list):
        image_tensor = [t.to(target_device, dtype=torch.float16) for t in image_tensor]
    else:
        image_tensor = image_tensor.to(target_device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(target_device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty > 1.0:
        from transformers.generation.logits_process import LogitsProcessor

        class _SafeRepetitionPenaltyLogitsProcessor(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty = penalty

            def __call__(self, input_ids_tensor, scores):
                safe_ids = input_ids_tensor.clone()
                safe_ids[safe_ids < 0] = 0  # -200 placeholder -> safe index
                score = torch.gather(scores, 1, safe_ids)
                score = torch.where(score < 0, score * self.penalty, score / self.penalty)
                scores.scatter_(1, safe_ids, score)
                return scores

        logits_processors.append(
            _SafeRepetitionPenaltyLogitsProcessor(float(repetition_penalty))
        )

    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs: dict = {
        "images": image_tensor,
        "do_sample": do_sample,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
    }
    if logits_processors:
        gen_kwargs["logits_processor"] = logits_processors
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    # Seed immediately before generate so a sampled run is repeatable per-image.
    if seed is not None:
        set_seed(int(seed))

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    if output_ids.shape[1] >= input_ids.shape[1] and torch.equal(
        output_ids[:, : input_ids.shape[1]].cpu(), input_ids.cpu()
    ):
        new_tokens = output_ids[:, input_ids.shape[1] :]
    else:
        new_tokens = output_ids

    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    decoded = "{" + decoded.strip()  # re-attach prefilled brace
    if stop_str and decoded.endswith(stop_str):
        decoded = decoded[: -len(stop_str)]
    return decoded.strip()
