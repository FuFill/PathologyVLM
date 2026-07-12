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
import copy
import csv
import json
import sys
import tarfile
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_templates import PROMPTS, get_prompt, get_prompt_version

OUTPUT_FIELDS = [
    "image_id",
    "image_path",
    "model_name",
    "prompt_variant",
    "prompt_version",
    "schema_version",
    "temperature",
    "repetition_penalty",
    "max_new_tokens",
    "git_commit",
    "raw_response",
    "json_valid",
    "parse_valid",
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
    "source",
    "label",
    "prediction",
    "confidence",
    "x",
    "y",
    "patch_path",
    "slide_id",
    "patch_id",
    "attention_score",
    "attention_rank",
    "tile_size",
    "tile_in_mask",
    "dataset",
    "split",
    "fold",
    "mil_task_id",
    "mil_model_name",
    "minio_path",
    "original_patch_path",
    "error",
]

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
    import sys
    import types

    mpt_stub = types.ModuleType("llava.model.language_model.llava_mpt")

    class _LlavaMPTConfig:  # noqa: D401, WPS431
        model_type = "llava_mpt_stub"

    class _LlavaMPTForCausalLM:  # noqa: D401, WPS431
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
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            QUILT_LLAVA_GIT,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"pip install of llava failed (exit {res.returncode})")

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
    parser = argparse.ArgumentParser(
        description="Run Quilt-LLaVA inference locally or via ClearML."
    )
    parser.add_argument(
        "--project_name", default="Pathology/VLM", help="ClearML project name."
    )
    parser.add_argument(
        "--task_name",
        default="pathgen_llava_test_10_promptv2",
        help="ClearML task name.",
    )
    parser.add_argument(
        "--queue_name", default="default", help="ClearML queue for remote execution."
    )

    parser.add_argument(
        "--image_dir",
        default="",
        help="Local image directory. If set, images are read directly from this folder.",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=tuple(sorted(PROMPTS.keys())),
        default="standard",
        help="Which JSON prompt template to use.",
    )
    parser.add_argument(
        "--dataset_project", default="Pathology/VLM", help="ClearML dataset project."
    )
    parser.add_argument(
        "--dataset_name", default="pathgen_he_test_10", help="ClearML dataset name."
    )
    parser.add_argument(
        "--metadata_csv",
        default="",
        help=(
            "Patch metadata CSV. If omitted, auto-detect a single CSV in dataset root. "
            "Used to carry source/label/prediction/confidence/x/y into outputs."
        ),
    )

    parser.add_argument(
        "--model_name",
        default="wisdomik/Quilt-Llava-v1.5-7b",
        help="Hugging Face model id of the VLM.",
    )

    parser.add_argument(
        "--output_dir", default="outputs", help="Local output directory."
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=768, help="Generation length limit."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature."
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.08, help="Repetition penalty."
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--run_remote",
        action="store_true",
        help="If set, dispatch the task to the given ClearML queue and exit the local process.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    """Return the current git commit hash, or '' if unavailable.

    Recorded on every output row so a run can be traced back to the exact
    code state (the ClearML task also records this, but embedding it in the
    outputs makes downloaded artifacts self-describing).
    """
    import subprocess

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _norm_path(text: Any) -> str:
    return str(text or "").replace("\\", "/").strip()


def _tail_path_key(text: Any, depth: int = 3) -> str:
    norm = _norm_path(text).lower()
    if not norm:
        return ""
    parts = [p for p in norm.split("/") if p]
    if not parts:
        return ""
    return "/".join(parts[-depth:])


def _source_dir_name(source: Any) -> str:
    value = str(source or "unknown").strip().lower().replace(" ", "_")
    value = value.replace("_", "-")
    return value or "unknown"


def _first_non_empty(
    row: dict[str, Any], keys: tuple[str, ...], default: Any = ""
) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        elif value != "":
            return value
    return default


def _resolve_metadata_csv(dataset_path: Path, metadata_csv_arg: str) -> Path | None:
    if metadata_csv_arg:
        candidate = Path(metadata_csv_arg)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        local = (dataset_path / metadata_csv_arg).resolve()
        if local.is_file():
            return local
        repo_local = (PROJECT_ROOT / metadata_csv_arg).resolve()
        if repo_local.is_file():
            return repo_local
        raise FileNotFoundError(
            f"Metadata CSV not found: {metadata_csv_arg!r} "
            f"(checked absolute path, dataset-relative, and repo-relative locations)."
        )

    root_csv = sorted(dataset_path.glob("*.csv"))
    if len(root_csv) == 1:
        return root_csv[0]
    if len(root_csv) > 1:
        preferred = [p for p in root_csv if "metadata" in p.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            "Multiple CSV files found in dataset root; pass --metadata_csv explicitly."
        )

    nested_csv = sorted(dataset_path.rglob("*.csv"))
    if len(nested_csv) == 1:
        return nested_csv[0]
    if len(nested_csv) > 1:
        preferred = [p for p in nested_csv if "metadata" in p.name.lower()]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            "Multiple CSV files found in dataset; pass --metadata_csv explicitly."
        )

    return None


def _load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {path}")
        return list(reader)


def _build_metadata_lookup(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, deque[int]], dict[str, deque[int]]]:
    by_basename: dict[str, deque[int]] = defaultdict(deque)
    by_tail: dict[str, deque[int]] = defaultdict(deque)
    for idx, row in enumerate(rows):
        patch_ref = _first_non_empty(
            row, ("patch_path", "image_path", "minio_path"), default=""
        )
        basename = Path(_norm_path(patch_ref)).name.lower()
        if basename:
            by_basename[basename].append(idx)
        tail = _tail_path_key(patch_ref, depth=3)
        if tail:
            by_tail[tail].append(idx)
    return by_basename, by_tail


def _match_metadata_row(
    img_path: Path,
    dataset_path: Path,
    metadata_rows: list[dict[str, Any]],
    by_basename: dict[str, deque[int]],
    by_tail: dict[str, deque[int]],
) -> dict[str, Any] | None:
    try:
        rel = img_path.relative_to(dataset_path)
        rel_norm = rel.as_posix().lower()
    except ValueError:
        rel_norm = _norm_path(img_path).lower()

    tail = _tail_path_key(rel_norm, depth=3)
    basename = img_path.name.lower()

    if tail and tail in by_tail and by_tail[tail]:
        return metadata_rows[by_tail[tail].popleft()]
    if basename in by_basename and by_basename[basename]:
        return metadata_rows[by_basename[basename].popleft()]
    return None


def _make_metadata_record(
    img_path: Path,
    dataset_path: Path,
    metadata_row: dict[str, Any] | None,
) -> dict[str, Any]:
    patch_rel: str
    try:
        patch_rel = img_path.relative_to(dataset_path).as_posix()
    except ValueError:
        patch_rel = _norm_path(img_path)

    if metadata_row is None:
        return {
            "source": "",
            "label": "",
            "prediction": "",
            "confidence": "",
            "x": "",
            "y": "",
            "patch_path": patch_rel,
            "slide_id": "",
            "patch_id": img_path.stem,
            "attention_score": "",
            "attention_rank": "",
            "tile_size": "",
            "tile_in_mask": "",
            "dataset": "",
            "split": "",
            "fold": "",
            "mil_task_id": "",
            "mil_model_name": "",
            "minio_path": "",
            "original_patch_path": "",
        }

    return {
        "source": _first_non_empty(metadata_row, ("source",), default=""),
        "label": _first_non_empty(metadata_row, ("label", "slide_label"), default=""),
        "prediction": _first_non_empty(metadata_row, ("prediction",), default=""),
        "confidence": _first_non_empty(metadata_row, ("confidence",), default=""),
        "x": _first_non_empty(metadata_row, ("x",), default=""),
        "y": _first_non_empty(metadata_row, ("y",), default=""),
        "patch_path": patch_rel,
        "slide_id": _first_non_empty(metadata_row, ("slide_id",), default=""),
        "patch_id": _first_non_empty(
            metadata_row, ("patch_id",), default=img_path.stem
        ),
        "attention_score": _first_non_empty(
            metadata_row, ("attention_score",), default=""
        ),
        "attention_rank": _first_non_empty(
            metadata_row, ("attention_rank",), default=""
        ),
        "tile_size": _first_non_empty(metadata_row, ("tile_size",), default=""),
        "tile_in_mask": _first_non_empty(metadata_row, ("tile_in_mask",), default=""),
        "dataset": _first_non_empty(metadata_row, ("dataset",), default=""),
        "split": _first_non_empty(metadata_row, ("split",), default=""),
        "fold": _first_non_empty(metadata_row, ("fold",), default=""),
        "mil_task_id": _first_non_empty(metadata_row, ("task_id",), default=""),
        "mil_model_name": _first_non_empty(metadata_row, ("model_name",), default=""),
        "minio_path": _first_non_empty(metadata_row, ("minio_path",), default=""),
        "original_patch_path": _first_non_empty(
            metadata_row, ("patch_path",), default=""
        ),
    }


def _write_vlm_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "image_id",
        "patch_path",
        "source",
        "label",
        "prediction",
        "confidence",
        "x",
        "y",
        "model_name",
        "prompt_variant",
        "prompt_version",
        "schema_version",
        "temperature",
        "repetition_penalty",
        "max_new_tokens",
        "git_commit",
        "json_valid",
        "parse_valid",
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
        "slide_id",
        "patch_id",
        "attention_score",
        "attention_rank",
        "tile_size",
        "tile_in_mask",
        "dataset",
        "split",
        "fold",
        "mil_task_id",
        "mil_model_name",
        "minio_path",
        "original_patch_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_safe(row.get(k, "")) for k in fieldnames})


def _build_visualizations_tar(
    tar_path: Path,
    vis_items: list[dict[str, Any]],
) -> None:
    with tarfile.open(tar_path, "w:gz") as tf:
        seen: set[tuple[str, str]] = set()
        for item in vis_items:
            src = Path(str(item.get("image_path", "")))
            if not src.is_file():
                continue
            source_dir = _source_dir_name(item.get("source", "unknown"))
            slide_id = str(item.get("slide_id", "unknown")).strip() or "unknown"
            arcname = f"visualizations/{source_dir}/{slide_id}/{src.name}"
            key = (str(src), arcname.lower())
            if key in seen:
                continue
            seen.add(key)
            tf.add(str(src), arcname=arcname)


def main() -> int:
    args = _parse_args()

    class _LocalLogger:
        def report_image(self, *args, **kwargs):
            return None

        def report_scalar(self, *args, **kwargs):
            return None

        def report_single_value(self, *args, **kwargs):
            return None

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
            print(
                f"[run_remote_vlm] ERROR: clearml is not installed: {exc}",
                file=sys.stderr,
            )
            return 2

        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            reuse_last_task_id=False,
            output_uri=False,
        )
        task.connect(vars(args))
        logger = task.get_logger()

        try:
            req_path = PROJECT_ROOT / "requirements.txt"
            if req_path.is_file():
                task.set_packages(str(req_path))
                print(f"[run_remote_vlm] Pinned remote packages from: {req_path}")
            else:
                print(
                    f"[run_remote_vlm] WARNING: requirements.txt not found at {req_path}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: could not set packages: {exc}")

    if args.run_remote:
        print(f"[run_remote_vlm] Dispatching task to queue: {args.queue_name!r}")
        task.execute_remotely(queue_name=args.queue_name, exit_process=True)

    try:
        _bootstrap_llava()
    except Exception as exc:  # noqa: BLE001
        print(f"[run_remote_vlm] ERROR bootstrapping llava: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    try:
        import torch  # noqa: WPS433 (local import on purpose)

        from src.image_utils import find_images
        from src.json_utils import SCHEMA_VERSION, extract_json, normalize_json
        from src.vlm_inference import generate_answer, load_model
    except ImportError as exc:
        print(f"[run_remote_vlm] ERROR importing dependencies: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

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
            dataset_path = Path(dataset.get_local_copy())
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] ERROR retrieving dataset: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    print(f"[run_remote_vlm] Input local path: {dataset_path}")

    metadata_csv_path: Path | None = None
    metadata_rows: list[dict[str, Any]] = []
    metadata_by_basename: dict[str, deque[int]] = {}
    metadata_by_tail: dict[str, deque[int]] = {}
    try:
        metadata_csv_path = _resolve_metadata_csv(dataset_path, args.metadata_csv)
        if metadata_csv_path is not None:
            metadata_rows = _load_metadata_rows(metadata_csv_path)
            metadata_by_basename, metadata_by_tail = _build_metadata_lookup(
                metadata_rows
            )
            print(
                f"[run_remote_vlm] Metadata CSV: {metadata_csv_path} "
                f"(rows={len(metadata_rows)})"
            )
        else:
            print(
                "[run_remote_vlm] Metadata CSV: not found (continuing without patch metadata)."
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[run_remote_vlm] ERROR loading metadata CSV: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    image_paths = find_images(dataset_path, max_images=args.max_images)
    print(f"[run_remote_vlm] Found {len(image_paths)} image(s).")
    if not image_paths:
        print("[run_remote_vlm] ERROR: no images found in dataset.", file=sys.stderr)
        return 1

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"[run_remote_vlm] CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[run_remote_vlm] GPU: {gpu_name}")
    print(f"[run_remote_vlm] Model: {args.model_name}")
    print(f"[run_remote_vlm] Prompt variant: {args.prompt_variant}")

    try:
        tokenizer, model, image_processor, context_len = load_model(
            args.model_name, load_4bit=bool(args.load_4bit)
        )
        print(f"[run_remote_vlm] context_len={context_len}")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[run_remote_vlm] ERROR loading model {args.model_name!r}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _git_commit()
    prompt_version = get_prompt_version(args.prompt_variant)
    provenance = {
        "prompt_variant": args.prompt_variant,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "git_commit": git_commit,
    }
    print(
        f"[run_remote_vlm] Provenance: prompt_version={prompt_version} "
        f"schema_version={SCHEMA_VERSION} git_commit={git_commit or '(unknown)'}"
    )

    jsonl_path = output_dir / "vlm_outputs.jsonl"
    csv_path = output_dir / "vlm_outputs.csv"
    vlm_metadata_path = output_dir / "vlm_metadata.csv"
    visualizations_tar_path = output_dir / "visualizations.tar.gz"

    rows: list[dict] = []
    vlm_metadata_rows: list[dict[str, Any]] = []
    visualization_items: list[dict[str, Any]] = []

    n_valid_json = 0
    metadata_matched = 0

    with jsonl_path.open("w", encoding="utf-8") as jfout:
        for idx, img_path in enumerate(image_paths):
            image_id = img_path.stem
            print(f"[run_remote_vlm] [{idx + 1}/{len(image_paths)}] {img_path}")

            matched_metadata: dict[str, Any] | None = None
            if metadata_rows:
                matched_metadata = _match_metadata_row(
                    img_path=img_path,
                    dataset_path=dataset_path,
                    metadata_rows=metadata_rows,
                    by_basename=metadata_by_basename,
                    by_tail=metadata_by_tail,
                )
                if matched_metadata is not None:
                    metadata_matched += 1
            metadata_record = _make_metadata_record(
                img_path=img_path,
                dataset_path=dataset_path,
                metadata_row=matched_metadata,
            )

            row: dict[str, Any] = {
                "image_id": image_id,
                "image_path": str(img_path),
                "model_name": args.model_name,
                "raw_response": "",
                "json_valid": False,
                "parse_valid": False,
                "error": "",
            }
            row.update(provenance)

            row.update(normalize_json(None))
            row.update(metadata_record)

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
                print(
                    f"[run_remote_vlm] WARNING: could not log image {img_path}: {exc}"
                )

            try:
                raw = generate_answer(
                    image_path=img_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    prompt=get_prompt(args.prompt_variant),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                )
                row["raw_response"] = raw

                parsed = extract_json(raw)
                row["json_valid"] = parsed is not None
                row["parse_valid"] = parsed is not None
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

            export_row = {
                "image_id": image_id,
                "model_name": args.model_name,
            }
            export_row.update(provenance)
            export_row.update(copy.deepcopy(metadata_record))

            for key in (
                "json_valid",
                "parse_valid",
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
            ):
                export_row[key] = copy.deepcopy(row.get(key, ""))

            vlm_metadata_rows.append(export_row)
            visualization_items.append(
                {
                    "image_path": str(img_path),
                    "source": metadata_record.get("source", ""),
                    "slide_id": metadata_record.get("slide_id", ""),
                }
            )

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

    with csv_path.open("w", encoding="utf-8", newline="") as cfout:
        writer = csv.DictWriter(cfout, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_safe(row.get(k, "")) for k in OUTPUT_FIELDS})

    _write_vlm_metadata_csv(vlm_metadata_path, vlm_metadata_rows)
    _build_visualizations_tar(visualizations_tar_path, visualization_items)

    n_total = len(rows)
    valid_rate = (n_valid_json / n_total) if n_total else 0.0
    print(f"[run_remote_vlm] Wrote {jsonl_path}")
    print(f"[run_remote_vlm] Wrote {csv_path}")
    print(f"[run_remote_vlm] Wrote {vlm_metadata_path}")
    print(f"[run_remote_vlm] Wrote {visualizations_tar_path}")
    print(f"[run_remote_vlm] num_images={n_total} valid_json_rate={valid_rate:.3f}")

    if metadata_rows:
        print(
            f"[run_remote_vlm] metadata matches: {metadata_matched}/{n_total} "
            f"(metadata rows: {len(metadata_rows)})"
        )

    if task is not None:
        try:
            logger.report_single_value(name="num_images", value=n_total)
            logger.report_single_value(name="valid_json_rate", value=valid_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: could not report single values: {exc}")

        try:
            task.upload_artifact(
                name="vlm_outputs_jsonl", artifact_object=str(jsonl_path)
            )
            task.upload_artifact(name="vlm_outputs_csv", artifact_object=str(csv_path))
            task.upload_artifact(
                name="vlm_metadata", artifact_object=str(vlm_metadata_path)
            )
            task.upload_artifact(
                name="visualizations", artifact_object=str(visualizations_tar_path)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[run_remote_vlm] WARNING: artifact upload failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
