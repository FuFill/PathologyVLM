"""Upload a local image folder as a ClearML Dataset.

Usage
-----
    python scripts/upload_clearml_dataset.py \
        --dataset_project Pathology/VLM \
        --dataset_name quilt_he_test_10 \
        --folder data/he_test_10

        python scripts/upload_clearml_dataset.py \
            --dataset_project Pathology/VLM \
            --dataset_name quilt-1m_test_40 \
            --folder data/quilt-1m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a folder as a ClearML Dataset."
    )
    parser.add_argument(
        "--dataset_project", required=True, help="ClearML dataset project name."
    )
    parser.add_argument("--dataset_name", required=True, help="ClearML dataset name.")
    parser.add_argument("--folder", required=True, help="Local folder to upload.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(
            f"[upload_clearml_dataset] ERROR: folder does not exist: {folder}",
            file=sys.stderr,
        )
        return 1

    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        print(
            f"[upload_clearml_dataset] ERROR: folder is empty: {folder}",
            file=sys.stderr,
        )
        return 1

    try:
        from clearml import Dataset
    except ImportError as exc:
        print(
            f"[upload_clearml_dataset] ERROR: 'clearml' is not installed: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"[upload_clearml_dataset] Creating ClearML dataset "
        f"project={args.dataset_project!r} name={args.dataset_name!r}"
    )
    dataset = Dataset.create(
        dataset_project=args.dataset_project,
        dataset_name=args.dataset_name,
    )

    print(f"[upload_clearml_dataset] Adding {len(files)} files from {folder} ...")
    dataset.add_files(path=str(folder))

    print("[upload_clearml_dataset] Uploading ...")
    dataset.upload()

    print("[upload_clearml_dataset] Finalizing ...")
    dataset.finalize()

    print("[upload_clearml_dataset] Done.")
    print(f"  project : {args.dataset_project}")
    print(f"  name    : {args.dataset_name}")
    print(f"  id      : {dataset.id}")
    print(f"  n_files : {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
