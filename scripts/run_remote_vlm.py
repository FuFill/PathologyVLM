"""Run the QUILT-LLaVA baseline on a ClearML Dataset of H&E images.

Can run locally or be dispatched to a remote ClearML GPU agent via
``--run_remote``. Results are saved as JSONL/CSV and uploaded to ClearML
as artifacts, with input images logged as debug images and basic scalar
metrics reported.

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

# ----------------------------------------------------------------------------
# Fixed VLM prompt. Do not edit without versioning the task name.
# ----------------------------------------------------------------------------
PROMPT = """You are a pathology image assistant. Analyze the provided H&E histology image.

Important rules:
1. Do not provide a final diagnosis.
2. Describe only visible morphological features.
3. If there is not enough visual evidence, set should_abstain to true.
4. Return only valid JSON. Do not add explanations outside JSON.

Return JSON with exactly these fields:
{
  "tissue_description": "",
  "cellularity": "",
  "architecture": "",
  "visible_abnormalities": [],
  "tumor_suspicious": "yes/no/uncertain",
  "evidence": [],
  "artifacts": [],
  "limitations": [],
  "confidence": "low/medium/high",
  "should_abstain": true
}"""


# Columns in the output CSV / JSONL. Keep stable for downstream consumers.
OUTPUT_FIELDS = [
    "image_id",
    "image_path",
    "model_name",
    "raw_response",
    "json_valid",
    "tissue_description",
    "cellularity",
    "architecture",
    "visible_abnormalities",
    "tumor_suspicious",
    "evidence",
    "artifacts",
    "limitations",
    "confidence",
    "should_abstain",
    "error",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote VLM inference via ClearML.")
    parser.add_argument("--project_name", default="Pathology/VLM", help="ClearML project name.")
    parser.add_argument("--task_name", default="quilt_llava_test_10", help="ClearML task name.")
    parser.add_argument("--queue_name", default="gpu", help="ClearML queue for remote execution.")

    parser.add_argument("--dataset_project", default="Pathology/VLM", help="ClearML dataset project.")
    parser.add_argument("--dataset_name", default="quilt_he_test_10", help="ClearML dataset name.")

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

    # ------------------------------------------------------------------
    # 1. ClearML task init (must happen as early as possible).
    # ------------------------------------------------------------------
    try:
        from clearml import Dataset, Task
    except ImportError as exc:
        print(f"[run_remote_vlm] ERROR: clearml is not installed: {exc}", file=sys.stderr)
        return 2

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        reuse_last_task_id=False,
    )
    task.connect(vars(args))
    logger = task.get_logger()

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
    # ------------------------------------------------------------------
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
    # 4. Resolve ClearML Dataset to a local path.
    # ------------------------------------------------------------------
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

    print(f"[run_remote_vlm] Dataset local path: {dataset_path}")

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

    # ------------------------------------------------------------------
    # 5. Load model.
    # ------------------------------------------------------------------
    try:
        processor, model = load_model(args.model_name, load_4bit=bool(args.load_4bit))
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
                    processor=processor,
                    model=model,
                    prompt=PROMPT,
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
