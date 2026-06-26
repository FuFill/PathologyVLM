"""Run the QUILT-LLaVA baseline on local Quilt-1M images or a ClearML Dataset.

Can run locally on a folder such as ``data/quilt-1m`` or be dispatched to
a remote ClearML GPU agent via ``--run_remote`` when the inputs are stored
in a ClearML Dataset. Results are saved as JSONL/CSV and uploaded to
ClearML as artifacts, with input images logged as debug images and basic
scalar metrics reported.

The model must not provide a final diagnosis. The prompt below enforces
this constraint and requires a structured JSON reply.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Make the local 'src' package importable when the script is launched from
# the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_templates import get_prompt


# Columns in the output CSV / JSONL. Keep stable for downstream consumers.
OUTPUT_FIELDS = [
    "image_id",
    "image_path",
    "model_name",
    "raw_response",
    "json_valid",
    "tissue_organ",
    "tissue_description",
    "cellularity",
    "architecture",
    "visible_abnormalities",
    "tumor_suspicious",
    "evidence",
    "artifacts",
    "limitations",
    "visual_description_confidence",
    "conclusion_confidence",
    "should_abstain",
    "error",
]


# Pinned Quilt-LLaVA git ref. We install at runtime with --no-deps to avoid
# its setup.py downgrading torch/transformers from our requirements.txt.
QUILT_LLAVA_GIT = (
    "git+https://github.com/aldraus/quilt-llava"
    "@7e70fc39f792ac55de010eb37bff0a6d6f491c13"
)


def _free_transformers_llava_slot() -> None:
    """Unregister HF transformers' built-in ``llava`` config so the upstream
    Quilt-LLaVA package can claim the same ``model_type`` key.

    Background
    ----------
    HF transformers >= 4.36 ships a ``LlavaForConditionalGeneration`` whose
    ``model_type`` is ``"llava"``. The upstream Quilt-LLaVA / LLaVA-1.5
    package contains, at import time:

        AutoConfig.register("llava", LlavaConfig)

    Without intervention, importing ``llava`` raises:

        ValueError: 'llava' is already used by a Transformers config,
                    pick another name.

    We remove the entry from both ``CONFIG_MAPPING_NAMES`` (lazy lookup
    table) and ``CONFIG_MAPPING._extra_content`` (set by previous
    registrations) BEFORE importing ``llava``. The upstream
    ``AutoConfig.register("llava", ...)`` then succeeds and points the
    ``"llava"`` model_type at ``LlavaLlamaForCausalLM`` (Llama-derived,
    matches the wisdomik/Quilt-Llava-v1.5-7b checkpoint).

    The official ``LlavaForConditionalGeneration`` is incompatible with
    that checkpoint anyway (different state-dict key naming), so we lose
    nothing by reclaiming the slot.
    """
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[run_remote_vlm] WARNING: could not access transformers auto "
            f"mappings to free 'llava' slot: {exc}"
        )
        return

    removed_names = CONFIG_MAPPING_NAMES.pop("llava", None)
    extra = getattr(CONFIG_MAPPING, "_extra_content", {})
    removed_extra = extra.pop("llava", None) if isinstance(extra, dict) else None
    print(
        f"[run_remote_vlm] Freed transformers 'llava' slot "
        f"(names_entry={removed_names!r}, extra_entry={type(removed_extra).__name__ if removed_extra else None})"
    )


def _stub_llava_mpt() -> None:
    """Pre-insert a fake ``llava.model.language_model.llava_mpt`` module
    so the real one (which imports ``_expand_mask`` from
    ``transformers.models.bloom.modeling_bloom`` -- removed in transformers
    >= 4.36) is never executed.

    Our checkpoint ``wisdomik/Quilt-Llava-v1.5-7b`` is Llama-based; the
    MPT branch of the upstream package is dead code for our path.
    Builder selects Llama vs MPT via ``'mpt' in model_name.lower()``,
    so as long as the model name does not contain 'mpt', the stubbed
    ``LlavaMPTForCausalLM`` is never instantiated.

    IMPORTANT: We must ONLY pre-stub the leaf ``llava_mpt`` module. If we
    also pre-stub ``llava`` / ``llava.model`` / etc. with empty placeholder
    modules, Python's import system will happily return them instead of
    executing the real ``__init__.py`` files on disk, leaving the real
    package un-initialised (no ``LlavaLlamaForCausalLM``, no builder...).
    """
    import sys
    import types

    mpt_stub = types.ModuleType("llava.model.language_model.llava_mpt")

    class _LlavaMPTConfig:  # noqa: D401, WPS431
        """Stub - real MPT path disabled (transformers>=4.36 incompatibility)."""

        model_type = "llava_mpt_stub"

    class _LlavaMPTForCausalLM:  # noqa: D401, WPS431
        """Stub - real MPT path disabled (transformers>=4.36 incompatibility)."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LlavaMPTForCausalLM is stubbed out in this environment "
                "(MPT path requires transformers<4.36)."
            )

    mpt_stub.LlavaMPTForCausalLM = _LlavaMPTForCausalLM
    mpt_stub.LlavaMPTConfig = _LlavaMPTConfig
    sys.modules["llava.model.language_model.llava_mpt"] = mpt_stub
    print("[run_remote_vlm] Stubbed llava.model.language_model.llava_mpt")


def _bootstrap_llava() -> None:
    """Ensure the upstream ``llava`` package is importable.

    Quilt-LLaVA's setup.py pins ``torch==2.0.1`` and ``transformers==4.31``,
    which would clobber our requirements.txt stack. We therefore install
    it with ``--no-deps`` only if it is not already present.

    Before importing we also:
      * free the transformers ``llava`` config slot (otherwise
        ``AutoConfig.register('llava', ...)`` blows up);
      * stub the MPT submodule (otherwise it tries to import
        ``_expand_mask`` from transformers.models.bloom, which was
        removed in transformers>=4.36).
    """
    # Install first (if missing). We do this BEFORE pre-stubbing so that
    # the parent ``llava`` package on disk has the chance to register
    # itself normally; the stubs in sys.modules only intercept the
    # specific incompatible submodule.
    try:
        import importlib
        importlib.import_module("llava")  # probe only
        already = True
    except ImportError:
        already = False

    if not already:
        import subprocess
        print(f"[run_remote_vlm] Installing llava (--no-deps) from {QUILT_LLAVA_GIT}")
        cmd = [
            sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
            QUILT_LLAVA_GIT,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"pip install of llava failed (exit {res.returncode})")

    # Make sure any half-imported ``llava`` modules from a prior failed
    # import attempt are evicted; then apply patches and import fresh.
    import sys as _sys
    for key in list(_sys.modules):
        if key == "llava" or key.startswith("llava."):
            del _sys.modules[key]

    _free_transformers_llava_slot()
    _stub_llava_mpt()

    import importlib
    importlib.invalidate_caches()
    import llava  # noqa: F401
    print(f"[run_remote_vlm] llava importable from: {llava.__file__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quilt-LLaVA inference locally or via ClearML.")
    parser.add_argument("--project_name", default="Pathology/VLM", help="ClearML project name.")
    parser.add_argument("--task_name", default="pathgen_llava_test_10_promptv2", help="ClearML task name.")
    parser.add_argument("--queue_name", default="default", help="ClearML queue for remote execution.")

    parser.add_argument(
        "--image_dir",
        default="",
        help="Local image directory. If set, images are read directly from this folder.",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=tuple(sorted({"standard", "safe"})),
        default="standard",
        help="Which JSON prompt template to use.",
    )
    parser.add_argument("--dataset_project", default="Pathology/VLM", help="ClearML dataset project.")
    parser.add_argument("--dataset_name", default="pathgen_he_test_10", help="ClearML dataset name.")

    parser.add_argument(
        "--model_name",
        default="wisdomik/Quilt-Llava-v1.5-7b",
        help="Hugging Face model id of the VLM.",
    )

    parser.add_argument("--output_dir", default="outputs", help="Local output directory.")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to process.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Generation length limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--load_4bit", action="store_true", help="Load model with bitsandbytes 4-bit quantization.")
    parser.add_argument(
        "--run_remote",
        action="store_true",
        help="If set, dispatch the task to the given ClearML queue and exit the local process.",
    )
    return parser.parse_args()


def _csv_safe(value: Any) -> Any:
    """Make list/dict values CSV-safe by JSON-encoding them."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def main() -> int:
    args = _parse_args()

    class _LocalLogger:
        def report_image(self, *args, **kwargs):
            return None

        def report_scalar(self, *args, **kwargs):
            return None

        def report_single_value(self, *args, **kwargs):
            return None

    # ------------------------------------------------------------------
    # 1. ClearML task init (must happen as early as possible).
    # ------------------------------------------------------------------
    if args.run_remote and args.image_dir:
        print(
            "[run_remote_vlm] ERROR: --image_dir cannot be combined with --run_remote. "
            "Upload the folder as a ClearML Dataset instead.",
            file=sys.stderr,
        )
        return 2

    task = None
    if args.image_dir:
        logger = _LocalLogger()
        print("[run_remote_vlm] Running locally without ClearML task init.")
    else:
        try:
            from clearml import Dataset, Task
        except ImportError as exc:
            print(f"[run_remote_vlm] ERROR: clearml is not installed: {exc}", file=sys.stderr)
            return 2

        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            reuse_last_task_id=False,
            # Disable the agent's default S3 output_uri inheritance. The remote
            # clearml.conf on this server points sdk.development.default_output_uri
            # at an s3:// bucket whose driver (boto3) is not installed, which
            # makes Task.init() raise "Could not get access credentials". We do
            # not need remote artifact storage for this run -- artifacts will be
            # served by the ClearML files server.
            output_uri=False,
        )
        task.connect(vars(args))
        logger = task.get_logger()

        # Pin the remote Python environment to our repo's requirements.txt
        # instead of letting ClearML auto-freeze the local (Windows) venv,
        # which would otherwise produce an incompatible stack on the agent
        # (e.g. transformers==5.x, torch==2.12, pillow==12.2 which do not work
        # with wisdomik/Quilt-Llava-v1.5-7b).
        try:
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                task.set_packages(str(req_path))
                print(f"[run_remote_vlm] Pinned remote packages from: {req_path}")
            else:
                print(f"[run_remote_vlm] WARNING: requirements.txt not found at {req_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: could not set packages: {exc}")

    # ------------------------------------------------------------------
    # 2. If requested, switch to remote execution and exit locally.
    # ------------------------------------------------------------------
    if args.run_remote:
        print(f"[run_remote_vlm] Dispatching task to queue: {args.queue_name!r}")
        task.execute_remotely(queue_name=args.queue_name, exit_process=True)
        # execute_remotely exits the process; nothing below runs locally.

    # ------------------------------------------------------------------
    # 3. Heavy imports (only done where the code actually runs - locally
    #    or on the remote agent).
    #
    # Before importing src.vlm_inference (which imports `llava`), make
    # sure the upstream Quilt-LLaVA package is installed. We install with
    # --no-deps so its pinned torch==2.0.1 / transformers==4.31 do NOT
    # downgrade the rest of our stack from requirements.txt.
    # ------------------------------------------------------------------
    try:
        _bootstrap_llava()
    except Exception as exc:  # noqa: BLE001
        print(f"[run_remote_vlm] ERROR bootstrapping llava: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    try:
        import torch  # noqa: WPS433 (local import on purpose)

        from src.image_utils import find_images
        from src.json_utils import extract_json, normalize_json
        from src.vlm_inference import generate_answer, load_model
    except ImportError as exc:
        print(f"[run_remote_vlm] ERROR importing dependencies: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    # ------------------------------------------------------------------
    # 4. Resolve inputs to a local path.
    # ------------------------------------------------------------------
    if args.image_dir:
        dataset_path = Path(args.image_dir)
        print(f"[run_remote_vlm] Using local image directory: {dataset_path}")
    else:
        print(
            f"[run_remote_vlm] Retrieving ClearML dataset: "
            f"project={args.dataset_project!r} name={args.dataset_name!r}"
        )
        try:
            dataset = Dataset.get(
                dataset_project=args.dataset_project,
                dataset_name=args.dataset_name,
            )
            dataset_path = dataset.get_local_copy()
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] ERROR retrieving dataset: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    print(f"[run_remote_vlm] Input local path: {dataset_path}")

    image_paths = find_images(dataset_path, max_images=args.max_images)
    print(f"[run_remote_vlm] Found {len(image_paths)} image(s).")
    if not image_paths:
        print("[run_remote_vlm] ERROR: no images found in dataset.", file=sys.stderr)
        return 1

    # Environment info.
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"[run_remote_vlm] CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[run_remote_vlm] GPU: {gpu_name}")
    print(f"[run_remote_vlm] Model: {args.model_name}")
    print(f"[run_remote_vlm] Prompt variant: {args.prompt_variant}")

    # ------------------------------------------------------------------
    # 5. Load model.
    # ------------------------------------------------------------------
    try:
        tokenizer, model, image_processor, context_len = load_model(
            args.model_name, load_4bit=bool(args.load_4bit)
        )
        print(f"[run_remote_vlm] context_len={context_len}")
    except Exception as exc:  # noqa: BLE001
        print(f"[run_remote_vlm] ERROR loading model {args.model_name!r}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 6. Inference loop.
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "vlm_outputs.jsonl"
    csv_path = output_dir / "vlm_outputs.csv"

    rows: list[dict] = []
    n_valid_json = 0

    with jsonl_path.open("w", encoding="utf-8") as jfout:
        for idx, img_path in enumerate(image_paths):
            image_id = img_path.stem
            print(f"[run_remote_vlm] [{idx + 1}/{len(image_paths)}] {img_path}")

            row: dict[str, Any] = {
                "image_id": image_id,
                "image_path": str(img_path),
                "model_name": args.model_name,
                "raw_response": "",
                "json_valid": False,
                "error": "",
            }
            # Pre-fill with normalized defaults so the row is always complete.
            row.update(normalize_json(None))

            # Log the input image (best-effort).
            try:
                from PIL import Image as _PILImage

                with _PILImage.open(img_path) as im:
                    logger.report_image(
                        title="input_images",
                        series=image_id,
                        iteration=idx,
                        image=im.convert("RGB"),
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[run_remote_vlm] WARNING: could not log image {img_path}: {exc}")

            try:
                raw = generate_answer(
                    image_path=img_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    prompt=get_prompt(args.prompt_variant),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                row["raw_response"] = raw

                parsed = extract_json(raw)
                row["json_valid"] = parsed is not None
                if parsed is not None:
                    n_valid_json += 1
                row.update(normalize_json(parsed))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                print(f"[run_remote_vlm] ERROR on {img_path}: {err}")
                traceback.print_exc()
                row["error"] = err

            rows.append(row)
            jfout.write(json.dumps(row, ensure_ascii=False) + "\n")
            jfout.flush()

            # Scalar progress logging.
            processed = idx + 1
            logger.report_scalar(
                title="progress",
                series="processed_images",
                value=processed,
                iteration=processed,
            )
            valid_rate = n_valid_json / processed
            logger.report_scalar(
                title="quality",
                series="json_valid_rate",
                value=valid_rate,
                iteration=processed,
            )

    # ------------------------------------------------------------------
    # 7. Save CSV.
    # ------------------------------------------------------------------
    with csv_path.open("w", encoding="utf-8", newline="") as cfout:
        writer = csv.DictWriter(cfout, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_safe(row.get(k, "")) for k in OUTPUT_FIELDS})

    n_total = len(rows)
    valid_rate = (n_valid_json / n_total) if n_total else 0.0
    print(f"[run_remote_vlm] Wrote {jsonl_path}")
    print(f"[run_remote_vlm] Wrote {csv_path}")
    print(f"[run_remote_vlm] num_images={n_total} valid_json_rate={valid_rate:.3f}")

    # ------------------------------------------------------------------
    # 8. Final ClearML reporting + artifacts.
    # ------------------------------------------------------------------
    if task is not None:
        try:
            logger.report_single_value(name="num_images", value=n_total)
            logger.report_single_value(name="valid_json_rate", value=valid_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: could not report single values: {exc}")

        try:
            task.upload_artifact(name="vlm_outputs_jsonl", artifact_object=str(jsonl_path))
            task.upload_artifact(name="vlm_outputs_csv", artifact_object=str(csv_path))
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: artifact upload failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
