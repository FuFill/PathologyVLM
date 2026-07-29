"""Build new VLM patch registry with tissue fraction and required fields.

Reads original MIL metadata CSVs, computes tissue_fraction from PNGs,
deduplicates coordinates, generates 3 random seed selections, and saves
registry CSV + Parquet to S3.

For each physical patch saves:
  - dataset, patient_id, slide_id
  - region_uid (unique physical area hash)
  - x, y, tile_size, mag
  - tissue_fraction
  - tumor_mask_overlap
  - selection_source + rank
  - task_id + model_hash
  - relative_path inside archive
  - context_set (standard/diverse) for group labelling
  - random_seed (0 for non-random, seed value for random)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import get_s3_client, read_csv_from_s3, upload_to_s3

# --- Task ID mapping ---
C16_TASK = "a50fbde29aa04e9d829a4580fd5c68b8"
C17_TO_C16_TASK = "84ae21b1f8b744b39866e35a257bb8c7"
C17_TASK = "1d5657e342c149ef8305d568410dbc96"
C16_TO_C17_TASK = "b3982f50afbd489dba24f254c1967aaa"

METADATA_CSV_PATHS = {
    "c16_native": f"mil/vlm_patches/c16_abmil_vlm_metadata_{C16_TASK}.csv",
    "c17_to_c16": f"mil/vlm_patches/c16_abmil_vlm_metadata_{C17_TO_C16_TASK}.csv",
    "c17_native": f"mil/vlm_patches/c17_abmil_vlm_metadata_{C17_TASK}.csv",
    "c16_to_c17": f"mil/vlm_patches/c17_abmil_vlm_metadata_{C16_TO_C17_TASK}.csv",
}

TASK_HASH_MAP = {
    "c16_native": (C16_TASK, C16_TASK),
    "c17_to_c16": (C17_TO_C16_TASK, C17_TO_C16_TASK),
    "c17_native": (C17_TASK, C17_TASK),
    "c16_to_c17": (C16_TO_C17_TASK, C16_TO_C17_TASK),
}

ALLOWED_SOURCES = {"top_k", "random", "oracle_tumor", "oracle_non_tumor", "hard_negative"}

RANDOM_SEEDS = (42, 123, 456)

TISSUE_THRESHOLD = 200


def _compute_tissue_fraction(img: Image.Image) -> float:
    arr = np.array(img)
    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    tissue = np.sum(gray < TISSUE_THRESHOLD)
    total = gray.size
    return float(tissue / total) if total > 0 else 0.0


def _region_uid(row: dict) -> str:
    raw = f"{row.get('slide_id', '')}|{row.get('x', '')}|{row.get('y', '')}|{row.get('tile_size', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_patient(slide_id: str) -> str:
    s = str(slide_id).strip()
    if s.startswith("patient_"):
        return s.split("_node_")[0]
    if s.startswith("test_"):
        return s.rsplit("_tile_embeddings", 1)[0]
    return s


def _load_metadata_csv(name: str, s3_path: str) -> pd.DataFrame:
    print(f"  Loading {name} from s3://pershin-medailab/{s3_path}")
    df = read_csv_from_s3(s3_path)
    print(f"    -> {len(df)} rows, columns: {list(df.columns)}")

    task_id, model_hash = TASK_HASH_MAP[name]

    if "tile_in_mask" in df.columns:
        mask_col = "tile_in_mask"
    elif "is_positive" in df.columns:
        mask_col = "is_positive"
    else:
        mask_col = None

    records = []
    for _, row in df.iterrows():
        slide = str(row.get("slide_id", ""))
        source = str(row.get("source", "")).strip().lower()
        if source not in ALLOWED_SOURCES:
            continue

        r = {
            "dataset": name,
            "patient_id": _parse_patient(slide),
            "slide_id": slide,
            "x": pd.to_numeric(row.get("x", 0), errors="coerce"),
            "y": pd.to_numeric(row.get("y", 0), errors="coerce"),
            "tile_size": pd.to_numeric(row.get("tile_size", 256), errors="coerce"),
            "mag": 20,
            "tumor_mask_overlap": int(row.get(mask_col, 0)) if mask_col and pd.notna(row.get(mask_col)) else 0,
            "selection_source": source,
            "rank": pd.to_numeric(row.get("rank", 0), errors="coerce"),
            "task_id": task_id,
            "model_hash": model_hash,
            "context_set": str(row.get("context_set", "standard")).strip().lower(),
            "is_diverse": 1 if str(row.get("context_set", "")).strip().lower() == "diverse" or row.get("is_diverse_topk", 0) == 1 else 0,
            "patch_uid": str(row.get("patch_uid", row.get("patch_id", ""))),
            "minio_path": str(row.get("minio_path", "")),
            "relative_path": str(row.get("relative_path", row.get("patch_path", ""))),
        }
        records.append(r)

    result = pd.DataFrame(records)
    print(f"    After source filter: {len(result)} rows")
    return result


def _extract_png_from_s3(minio_path: str) -> Image.Image | None:
    if "!" not in minio_path:
        return None
    tar_part, internal = minio_path.split("!", 1)
    tar_part = tar_part.replace("s3://", "").split("/", 1)[1] if "s3://" in tar_part else tar_part
    tar_part = tar_part.lstrip("/")
    internal = internal.lstrip("/")

    client = get_s3_client()
    try:
        obj = client.get_object(Bucket="pershin-medailab", Key=tar_part)
        body = obj["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            try:
                member = tf.getmember(internal)
            except KeyError:
                alt = internal.replace("vlm_patches/", "vlm_patches_standard/")
                member = tf.getmember(alt)
            f = tf.extractfile(member)
            if f is None:
                return None
            img = Image.open(io.BytesIO(f.read()))
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
    except Exception as exc:
        print(f"    WARNING: extract failed for {minio_path}: {exc}")
        return None


def _extract_png_local(relative_path: str, local_archive_dir: Path) -> Image.Image | None:
    full = local_archive_dir / relative_path
    if full.exists():
        try:
            img = Image.open(full)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            return None
    if local_archive_dir.name == "vlm_patches":
        alt = local_archive_dir.parent / relative_path.replace("vlm_patches/", "vlm_patches_standard/")
        if alt.exists():
            try:
                img = Image.open(alt)
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception:
                return None
    alt2 = local_archive_dir.parent / relative_path
    if alt2.exists():
        try:
            img = Image.open(alt2)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            return None
    return None


def _compute_tissue_fractions(
    df: pd.DataFrame,
    local_dir: Path | None,
    use_s3: bool,
) -> pd.DataFrame:
    print(f"  Computing tissue_fraction for {len(df)} patches...")
    fractions = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 500 == 0:
            print(f"    [{idx}/{total}]")
        img = None
        if local_dir:
            img = _extract_png_local(row["relative_path"], local_dir)
        if img is None and use_s3:
            img = _extract_png_from_s3(row["minio_path"])
        if img is not None:
            fractions.append(_compute_tissue_fraction(img))
        else:
            fractions.append(0.0)
    df["tissue_fraction"] = fractions
    print(f"    Done. Mean tissue_fraction: {np.mean(fractions):.3f}")
    return df


def _build_output_registry(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["dataset"] = df["dataset"]
    out["patient_id"] = df["patient_id"]
    out["slide_id"] = df["slide_id"]
    out["region_uid"] = df["region_uid"]
    out["x"] = df["x"]
    out["y"] = df["y"]
    out["tile_size"] = df["tile_size"]
    out["mag"] = df["mag"]
    out["tissue_fraction"] = df["tissue_fraction"]
    out["tumor_mask_overlap"] = df["tumor_mask_overlap"]
    out["selection_source"] = df["selection_source"]
    out["rank"] = df["rank"]
    out["task_id"] = df["task_id"]
    out["model_hash"] = df["model_hash"]
    out["context_set"] = df["context_set"]
    out["is_diverse"] = df["is_diverse"]
    out["random_seed"] = df["random_seed"]
    out["relative_path"] = df["relative_path"]
    out["minio_path"] = df["minio_path"]
    out["patch_uid"] = df["patch_uid"]
    return out


def _deduplicate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values(["slide_id", "selection_source", "rank"])
    df = df.drop_duplicates(subset=["slide_id", "x", "y", "tile_size"], keep="first")
    after = len(df)
    print(f"  Dedup coordinates: {before} -> {after} rows ({before - after} removed)")
    return df


def _generate_random_subsets(df: pd.DataFrame) -> pd.DataFrame:
    random_rows = df[df["selection_source"] == "random"].copy()
    non_random = df[df["selection_source"] != "random"].copy()

    subsets = []
    for seed_val in RANDOM_SEEDS:
        sampled = random_rows.groupby("slide_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), 5), random_state=int(seed_val)),
            include_groups=False,
        )
        sampled = sampled.reset_index(drop=True)
        sampled["random_seed"] = seed_val
        subsets.append(sampled)

    df_random = pd.concat(subsets, ignore_index=True)
    non_random["random_seed"] = 0

    combined = pd.concat([non_random, df_random], ignore_index=True)
    print(f"  Random seeds {RANDOM_SEEDS}: {len(df_random)} random rows generated")
    return combined


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n  Registry summary:")
    print(f"    Total entries: {len(df)}")
    print(f"    Datasets: {sorted(df['dataset'].unique())}")
    print(f"    Slides: {df['slide_id'].nunique()}")
    print(f"    Patients: {df['patient_id'].nunique()}")
    print(f"    Unique regions: {df['region_uid'].nunique()}")
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        print(f"    {ds}: {len(sub)} rows, {sub['slide_id'].nunique()} slides")
    for src in sorted(df["selection_source"].unique()):
        n = len(df[df["selection_source"] == src])
        print(f"      {src}: {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VLM patch registry")
    parser.add_argument("--datasets", nargs="+", default=list(METADATA_CSV_PATHS.keys()),
                        help="Datasets to include")
    parser.add_argument("--local_archive_dir", default="",
                        help="Local path to extracted archive (e.g. c16_abmil_vlm_patches_*/vlm_patches)")
    parser.add_argument("--use_s3_images", action="store_true",
                        help="Download images from S3 for tissue_fraction (slow)")
    parser.add_argument("--output_s3_prefix", default="mil/vlm_patches_registry",
                        help="S3 prefix for output")
    parser.add_argument("--skip_tissue_fraction", action="store_true",
                        help="Skip tissue_fraction computation, set to 1.0")
    args = parser.parse_args()

    local_dir = Path(args.local_archive_dir) if args.local_archive_dir else None

    all_dfs = []
    for name in args.datasets:
        s3_path = METADATA_CSV_PATHS.get(name)
        if s3_path is None:
            print(f"[registry] SKIP unknown dataset: {name}")
            continue
        df = _load_metadata_csv(name, s3_path)
        all_dfs.append(df)

    if not all_dfs:
        print("[registry] No data loaded. Exiting.")
        return 1

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[registry] Total before processing: {len(combined)}")

    combined["region_uid"] = combined.apply(_region_uid, axis=1)
    combined["is_diverse"] = combined["is_diverse"].fillna(0).astype(int)

    if args.skip_tissue_fraction:
        combined["tissue_fraction"] = 1.0
        print("[registry] Skipping tissue_fraction, set to 1.0")
    else:
        combined = _compute_tissue_fractions(combined, local_dir, args.use_s3_images)

    combined = _deduplicate_coordinates(combined)

    combined = _generate_random_subsets(combined)

    registry = _build_output_registry(combined)

    _print_summary(registry)

    csv_key = f"{args.output_s3_prefix}/patch_registry.csv"
    parquet_key = f"{args.output_s3_prefix}/patch_registry.parquet"

    tmp_dir = tempfile.gettempdir()
    local_csv = os.path.join(tmp_dir, "patch_registry.csv")
    local_parquet = os.path.join(tmp_dir, "patch_registry.parquet")

    registry.to_csv(local_csv, index=False)
    try:
        registry.to_parquet(local_parquet, index=False)
    except ImportError:
        print("[registry] WARNING: pyarrow not installed, skipping parquet")
        local_parquet = None

    csv_url = upload_to_s3(local_csv, csv_key)
    print(f"\n[registry] Uploaded:")
    print(f"  CSV:     {csv_url}")
    if local_parquet:
        parquet_url = upload_to_s3(local_parquet, parquet_key)
        print(f"  Parquet: {parquet_url}")

    print(f"[registry] Done. {len(registry)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
