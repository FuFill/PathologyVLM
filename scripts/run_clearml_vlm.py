"""ClearML task launcher for VLM pipeline steps.

Creates a ClearML task with the right packages and dispatches it
to a GPU queue, or runs locally if --run_local is set.

Usage:
  # Build patch registry (CPU, no ClearML needed)
  python scripts/build_patch_registry.py

  # C16 benchmark on GPU
  python scripts/run_clearml_vlm.py --step benchmark_c16 --queue gpu

  # C17 final run (Quilt-LLaVA)
  python scripts/run_clearml_vlm.py --step run_c17 --model quilt_llava --queue gpu

  # C17 final run (MedGemma)
  python scripts/run_clearml_vlm.py --step run_c17 --model med_gemma --queue gpu

  # Evaluate results (CPU)
  python scripts/run_clearml_vlm.py --step evaluate --run_local
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REGISTRY_CSV = "mil/vlm_patches_registry/patch_registry.csv"
REGISTRY_S3 = f"s3://pershin-medailab/Pathomorphology/CAMELYON/{REGISTRY_CSV}"

STEPS = {
    "build_registry": {
        "script": "build_patch_registry.py",
        "packages": "requirements.txt",
        "gpu": False,
        "args": [],
    },
    "benchmark_c16": {
        "script": "benchmark_vlm_c16.py",
        "packages": "requirements-medgemma.txt",
        "gpu": True,
        "args": [
            "--registry_csv", REGISTRY_S3,
            "--max_patches", "200",
        ],
    },
    "run_c17": {
        "script": "run_vlm.py",
        "gpu": True,
        "packages": "requirements-medgemma.txt",  # override with --model for quilt
        "args": [
            "--registry_csv", REGISTRY_S3,
            "--dataset", "c17_native",
            "--source", "top_k",
            "--mode", "context",
            "--n_patches", "3",
            "--max_slides", "100",
            "--temperature", "0.0",
        ],
    },
    "run_c17_random": {
        "script": "run_vlm.py",
        "gpu": True,
        "packages": "requirements-medgemma.txt",
        "args": [
            "--registry_csv", REGISTRY_S3,
            "--dataset", "c17_native",
            "--source", "random",
            "--mode", "context",
            "--n_patches", "3",
            "--max_slides", "100",
            "--random_seed", "42",
            "--temperature", "0.0",
        ],
    },
    "run_c17_control": {
        "script": "run_vlm.py",
        "gpu": True,
        "packages": "requirements-medgemma.txt",
        "args": [
            "--registry_csv", REGISTRY_S3,
            "--dataset", "c17_native",
            "--source", "all",
            "--mode", "context",
            "--n_patches", "3",
            "--max_slides", "100",
            "--temperature", "0.0",
        ],
    },
    "evaluate": {
        "script": "evaluate_pipeline.py",
        "packages": "requirements.txt",
        "gpu": False,
        "args": [
            "--registry_csv", REGISTRY_S3,
        ],
    },
    "evaluate_patches": {
        "script": "evaluate_patches.py",
        "packages": "requirements.txt",
        "gpu": False,
        "args": [
            "--registry_csv", REGISTRY_S3,
        ],
    },
}


def _resolve_s3_csv(s3_path: str) -> str:
    """Download S3 CSV to local tmp and return local path."""
    import tempfile
    from src.s3_utils import get_s3_client

    path = s3_path.replace("s3://pershin-medailab/", "").replace("s3://pershin-medailab/", "")
    parts = path.split("/", 1)
    if len(parts) < 2:
        return s3_path

    client = get_s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    client.download_file("pershin-medailab", path, tmp.name)
    print(f"[clearml_vlm] Downloaded registry to {tmp.name}")
    return tmp.name


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearML VLM task launcher")
    parser.add_argument("--step", choices=list(STEPS.keys()), required=True)
    parser.add_argument("--model", default="med_gemma", choices=["quilt_llava", "med_gemma", "med_siglip"])
    parser.add_argument("--queue", default="default")
    parser.add_argument("--run_local", action="store_true")
    parser.add_argument("--project", default="Pathology/VLM")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    step_cfg = STEPS[args.step]

    if args.model == "quilt_llava":
        step_cfg["packages"] = "requirements-quilt.txt"
    elif args.model == "med_gemma":
        step_cfg["packages"] = "requirements-medgemma.txt"

    script_path = str(PROJECT_ROOT / "scripts" / step_cfg["script"])
    req_path = str(PROJECT_ROOT / step_cfg["packages"])

    add_model = args.step != "build_registry"
    task_name = f"vlm_{args.step}_{args.model if add_model else 'cpu'}"

    build_args = step_cfg["args"] + args.extra_args
    if add_model:
        build_args += ["--model", args.model]
    build_args = [a for a in build_args if a]

    registry_s3 = next((a for a in build_args if a.startswith("s3://") and a.endswith(".csv")), None)

    if registry_s3 and not args.run_local:
        build_args = [REGISTRY_S3 if a == registry_s3 else a for a in build_args]
    elif registry_s3 and args.run_local:
        local_path = _resolve_s3_csv(registry_s3)
        build_args = [local_path if a == registry_s3 else a for a in build_args]

    if args.run_local:
        print(f"[clearml_vlm] Running locally: {script_path}")
        print(f"  Args: {build_args}")
        cmd = [sys.executable, script_path] + build_args
        env = os.environ.copy()
        result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        return result.returncode

    try:
        from clearml import Task
    except ImportError:
        print("[clearml_vlm] clearml not installed. Add --run_local to run without ClearML.")
        return 1

    task = Task.init(
        project_name=args.project,
        task_name=task_name,
        reuse_last_task_id=False,
        output_uri=False,
    )

    task.set_packages(req_path)
    task.connect(build_args, name="script_args")

    print(f"[clearml_vlm] Dispatching {task_name} to queue {args.queue!r}")
    task.execute_remotely(queue_name=args.queue, exit_process=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
