"""Build the unified VLM patch registry from MIL metadata CSVs on S3.

Reads 4 MIL metadata CSVs, deduplicates by (slide_id, x, y, tile_size),
normalises fields, generates 3 random seed selections, and saves the
registry as CSV + Parquet on S3.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import get_s3_client, read_csv_from_s3, upload_to_s3

C16_NATIVE_TASK = "a50fbde29aa04e9d829a4580fd5c68b8"
C17_TO_C16_TASK = "84ae21b1f8b744b39866e35a257bb8c7"
C17_NATIVE_TASK = "1d5657e342c149ef8305d568410dbc96"
C16_TO_C17_TASK = "b3982f50afbd489dba24f254c1967aaa"

CSV_PATHS = {
    "c16_native": f"mil/vlm_patches/c16_abmil_vlm_metadata_{C16_NATIVE_TASK}.csv",
    "c17_to_c16": f"mil/vlm_patches/c16_abmil_vlm_metadata_{C17_TO_C16_TASK}.csv",
    "c17_native": f"mil/vlm_patches/c17_abmil_vlm_metadata_{C17_NATIVE_TASK}.csv",
    "c16_to_c17": f"mil/vlm_patches/c17_abmil_vlm_metadata_{C16_TO_C17_TASK}.csv",
}

TASK_ID_MAP = {
    "c16_native": C16_NATIVE_TASK,
    "c17_to_c16": C17_TO_C16_TASK,
    "c17_native": C17_NATIVE_TASK,
    "c16_to_c17": C16_TO_C17_TASK,
}

ALLOWED_SOURCES = {"top_k", "random", "oracle_tumor", "oracle_non_tumor", "hard_negative"}

PROVENANCE_FIELDS = {
    "label",
    "prediction",
    "confidence",
    "attention_score",
    "attention_rank",
}


def _parse_slide_patient(slide_id: str, dataset: str) -> str:
    slide = str(slide_id).strip()
    if slide.startswith("patient_"):
        parts = slide.split("_node_")
        return parts[0] if len(parts) > 1 else slide
    if slide.startswith("test_"):
        parts = slide.split("_tile_embeddings")
        return parts[0] if len(parts) > 1 else slide
    return slide


def _region_uid(row: dict) -> str:
    raw = f"{row.get('slide_id', '')}|{row.get('x', '')}|{row.get('y', '')}|{row.get('tile_size', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_and_normalise(name: str, csv_path: str) -> pd.DataFrame:
    print(f"[registry] Loading {name} from s3://pershin-medailab/{csv_path}")
    df = read_csv_from_s3(csv_path)

    df["dataset_name"] = name
    df["task_id"] = TASK_ID_MAP[name]
    df["model_hash"] = TASK_ID_MAP[name]

    df["patient_id"] = df.apply(
        lambda r: _parse_slide_patient(r.get("slide_id", ""), r.get("dataset", "")),
        axis=1,
    )

    df["region_uid"] = df.apply(_region_uid, axis=1)

    df["source"] = df["source"].str.strip().str.lower()
    df = df[df["source"].isin(ALLOWED_SOURCES)]

    for col in ("x", "y", "tile_size", "rank", "tile_in_mask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  -> {len(df)} rows, {len(df['region_uid'].unique())} unique regions")
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values(["slide_id", "rank"])
    df = df.drop_duplicates(subset=["slide_id", "x", "y", "tile_size"], keep="first")
    after = len(df)
    print(f"[registry] Dedup: {before} -> {after} rows ({before - after} removed)")
    return df


def _generate_random_seeds(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    random_rows = df[df["source"] == "random"].copy()
    non_random = df[df["source"] != "random"].copy()

    result_rows = []
    for seed_val in seeds:
        sampled = random_rows.groupby("slide_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), 5), random_state=seed_val),
            include_groups=False,
        )
        sampled = sampled.reset_index(drop=True)
        sampled["random_seed"] = seed_val
        result_rows.append(sampled)

    df_random = pd.concat(result_rows, ignore_index=True)
    non_random["random_seed"] = 0

    combined = pd.concat([non_random, df_random], ignore_index=True)
    print(f"[registry] Random seeds {seeds}: {len(df_random)} random rows")
    return combined


def _build_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["dataset"] = df["dataset_name"]
    out["patient_id"] = df["patient_id"]
    out["slide_id"] = df["slide_id"]
    out["region_uid"] = df["region_uid"]
    out["x"] = df["x"]
    out["y"] = df["y"]
    out["tile_size"] = df["tile_size"]
    out["mag"] = 20
    out["tumor_mask_overlap"] = df["tile_in_mask"]
    out["selection_source"] = df["source"]
    out["rank"] = df["rank"]
    out["task_id"] = df["task_id"]
    out["model_hash"] = df["model_hash"]
    out["context_set"] = df.get("context_set", "standard")
    out["is_diverse"] = df.get("is_diverse_topk", 0)
    out["random_seed"] = df.get("random_seed", 0)
    out["relative_path"] = df["relative_path"]
    out["minio_path"] = df.get("minio_path", "")
    out["patch_uid"] = df["patch_uid"]

    return out


def _count_groups(df: pd.DataFrame) -> None:
    print("\n[registry] Per-source counts:")
    for source in sorted(df["selection_source"].unique()):
        n = len(df[df["selection_source"] == source])
        diverse_n = len(df[(df["selection_source"] == source) & (df["is_diverse"] == 1)])
        print(f"  {source}: {n} total ({diverse_n} diverse)")

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n  Dataset {ds}: {len(sub)} rows, {sub['slide_id'].nunique()} slides, "
              f"{sub['patient_id'].nunique()} patients")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VLM patch registry from S3 metadata CSVs")
    parser.add_argument(
        "--output_s3_prefix",
        default="mil/vlm_patches_registry",
        help="S3 prefix for output files",
    )
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for random patch selection",
    )
    parser.add_argument(
        "--local_cache",
        default="",
        help="If set, also save CSV locally",
    )
    args = parser.parse_args()

    all_dfs = []
    for name, csv_path in CSV_PATHS.items():
        df = _load_and_normalise(name, csv_path)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[registry] Total before dedup: {len(combined)}")

    combined = _deduplicate(combined)
    combined = _generate_random_seeds(combined, args.random_seeds)
    registry = _build_output_columns(combined)

    _count_groups(registry)

    csv_key = f"{args.output_s3_prefix}/patch_registry.csv"
    parquet_key = f"{args.output_s3_prefix}/patch_registry.parquet"

    import tempfile
    tmp_dir = tempfile.gettempdir()
    local_csv = os.path.join(tmp_dir, "patch_registry.csv")
    local_parquet = os.path.join(tmp_dir, "patch_registry.parquet")

    registry.to_csv(local_csv, index=False)
    try:
        registry.to_parquet(local_parquet, index=False)
    except ImportError:
        print("[registry] WARNING: pyarrow not installed, skipping parquet export")
        local_parquet = None

    csv_url = upload_to_s3(local_csv, csv_key)
    print(f"\n[registry] Uploaded:")
    print(f"  CSV:     {csv_url}")
    if local_parquet:
        parquet_url = upload_to_s3(local_parquet, parquet_key)
        print(f"  Parquet: {parquet_url}")

    if args.local_cache:
        cache_dir = Path(args.local_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        registry.to_csv(cache_dir / "patch_registry.csv", index=False)
        if local_parquet:
            registry.to_parquet(cache_dir / "patch_registry.parquet", index=False)
        print(f"[registry] Also saved to {cache_dir}")

    print(f"[registry] Done. Registry has {len(registry)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
