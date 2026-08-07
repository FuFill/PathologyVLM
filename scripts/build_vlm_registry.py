"""Build new VLM patch registry with tissue fraction and required fields.

Reads original MIL metadata CSVs, deduplicates coordinates, generates 3
random seed selections, and saves registry CSV + Parquet to S3.

tissue_fraction is computed from patch PNGs only when --local_archive_dir
or --use_s3_images is given; otherwise the values are merged from the
previous registry CSV.

For each physical patch saves:
  - dataset, patient_id, slide_id
  - region_uid (unique physical area hash)
  - x, y, tile_size, mag
  - tissue_fraction
  - tumor_mask_overlap (area fraction >= 20% of patch inside annotation polygons)
  - tumor_overlap_fraction (raw area fraction)
  - tumor_mask_overlap_center (legacy: tile center inside polygons)
  - selection_source + rank
  - task_id + model_hash
  - relative_path inside archive
  - context_set (standard/diverse) for group labelling
  - random_seed (0 for non-random, seed value for random)

Tumor GT is recomputed from the Camelyon annotation XMLs on S3.
XML resolution is based on the slide-id pattern, not the dataset name:
  - patient_*  -> C17: 17/annotations/{slide_id}.xml
  - test_*/normal_* -> C16: 16/{training|testing}/annotations/{slide}.xml
    with the trailing "_tile_embeddings" suffix stripped from slide_id.
Use --skip_tumor_gt_recompute to keep the inherited tile_in_mask values instead.

After the recompute, oracle groups are cleaned against the recomputed mask:
oracle_tumor rows with mask != 1 and oracle_non_tumor rows with mask != 0 are
dropped (see --skip_oracle_cleanup and the uploaded oracle_cleanup_report.csv).

tissue_fraction via --use_s3_images now caches each patch archive locally
(see --cache_dir) so every tar.gz is downloaded and extracted only once.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_utils import (
    get_minio_path_components,
    get_s3_client,
    read_csv_from_s3,
    upload_to_s3,
)

MINIO_BUCKET = "pershin-medailab"
MINIO_PREFIX = "Pathomorphology/CAMELYON"

TUMOR_GT_GRID = 20
TUMOR_GT_THRESHOLD = 0.2

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
            "split": str(row.get("split", "")).strip().lower(),
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


def _download_archive(tar_key: str, cache_dir: Path) -> Optional[Path]:
    local = cache_dir / tar_key.replace("/", "_")
    if local.exists() and local.stat().st_size > 0:
        print(f"    archive cached: {local.name}")
        return local
    tmp = local.with_name(local.name + ".part")
    client = get_s3_client()
    try:
        client.download_file(MINIO_BUCKET, tar_key, str(tmp))
        tmp.replace(local)
        print(f"    downloaded archive: {local.name} ({local.stat().st_size / 1e9:.2f} GB)")
        return local
    except Exception as exc:
        print(f"    WARNING: archive download failed {tar_key}: {exc}")
        if tmp.exists():
            tmp.unlink()
        return None


def _extract_archive_once(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(extract_dir, **kwargs)
    print(f"    extracted: {archive_path.name} -> {extract_dir}")


def _read_extracted(relative_path: str, extract_dir: Path) -> Image.Image | None:
    candidates = [extract_dir / relative_path]
    alt = relative_path.replace("vlm_patches/", "vlm_patches_standard/")
    if alt != relative_path:
        candidates.append(extract_dir / alt)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            img = Image.open(candidate)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            return None
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
    archive_cache_dir: Path | None,
) -> pd.DataFrame:
    print(f"  Computing tissue_fraction for {len(df)} patches...")
    total = len(df)
    fractions: list[float] = [np.nan] * total
    rows = list(df.iterrows())

    if local_dir is not None:
        for idx, (_, row) in enumerate(rows):
            if idx % 500 == 0:
                print(f"    [{idx}/{total}]")
            img = _extract_png_local(str(row.get("relative_path", "")), local_dir)
            if img is not None:
                fractions[idx] = _compute_tissue_fraction(img)

    elif use_s3 and archive_cache_dir is not None:
        archive_cache_dir.mkdir(parents=True, exist_ok=True)
        by_archive: dict[str, list[int]] = defaultdict(list)
        for idx, (_, row) in enumerate(rows):
            try:
                tar_key, _ = get_minio_path_components(str(row.get("minio_path", "")))
            except Exception:
                tar_key = ""
            by_archive[tar_key].append(idx)

        for tar_key, idxs in sorted(by_archive.items(), key=lambda kv: -len(kv[1])):
            if not tar_key:
                continue
            local_archive = _download_archive(tar_key, archive_cache_dir)
            if local_archive is None:
                continue
            extract_dir = archive_cache_dir / "extracted" / tar_key.replace("/", "_")
            _extract_archive_once(local_archive, extract_dir)
            done = 0
            for idx in idxs:
                rel = str(rows[idx][1].get("relative_path", ""))
                img = _read_extracted(rel, extract_dir)
                if img is not None:
                    fractions[idx] = _compute_tissue_fraction(img)
                    done += 1
            print(f"    archive {tar_key.split('/')[-1]}: {done}/{len(idxs)} patches")

    else:
        print("  No local dir / S3 images: tissue_fraction left for previous-registry merge")

    df["tissue_fraction"] = fractions
    n_ok = int(pd.notna(df["tissue_fraction"]).sum())
    print(f"    Done. computed={n_ok}/{total}, missing={total - n_ok}, "
          f"mean={df['tissue_fraction'].mean():.3f}")
    return df


def _parse_camelyon_xml(data: bytes) -> list:
    root = ET.fromstring(data)
    polygons = []
    for annotation in root.findall(".//Annotation"):
        coords = []
        for coord in annotation.findall(".//Coordinate"):
            coords.append([float(coord.get("X")), float(coord.get("Y"))])
        if len(coords) >= 3:
            polygons.append(np.array(coords, dtype=np.float32))
    return polygons


def _raycast_inside(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    n = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > py) != (y2 > py)
        if not cond.any():
            continue
        xc = (x2 - x1) * (py[cond] - y1) / (y2 - y1) + x1
        inside[cond] ^= px[cond] < xc
    return inside


def _points_in_polygons(px: np.ndarray, py: np.ndarray, polygons: list) -> np.ndarray:
    inside = np.zeros(len(px), dtype=bool)
    for poly in polygons:
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        in_bb = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
        if not in_bb.any():
            continue
        inside[in_bb] |= _raycast_inside(px[in_bb], py[in_bb], poly)
    return inside


def _patch_overlap_fraction(
    x: float, y: float, tile_size: float, polygons: list, grid: int = TUMOR_GT_GRID
) -> float:
    offs = (np.arange(grid) + 0.5) * (tile_size / grid)
    px, py = np.meshgrid(x + offs, y + offs)
    px = px.ravel()
    py = py.ravel()
    inside = _points_in_polygons(px, py, polygons)
    return float(inside.mean())


def _load_slide_polygons(client, dataset: str, slide: str, split: str):
    s = str(slide).strip()
    if s.startswith("patient_"):
        prefixes = ["17/annotations"]
    else:
        if s.startswith("normal_"):
            return None
        prefixes = ["16/testing/annotations", "16/training/annotations"]
        if split == "train":
            prefixes = ["16/training/annotations", "16/testing/annotations"]

    slide_xml = s[: -len("_tile_embeddings")] if s.endswith("_tile_embeddings") else s

    for prefix in prefixes:
        key = f"{MINIO_PREFIX}/{prefix}/{slide_xml}.xml"
        try:
            obj = client.get_object(Bucket=MINIO_BUCKET, Key=key)
            polygons = _parse_camelyon_xml(obj["Body"].read())
            if polygons:
                return polygons
        except Exception:
            continue
    return None


def _compute_tumor_gt(df: pd.DataFrame, client) -> pd.DataFrame:
    print(f"  Recomputing tumor GT for {len(df)} patches from annotation XMLs...")
    cache: dict = {}
    fracs = []
    mask20 = []
    mask_center = []
    n_missing = 0
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 1000 == 0:
            print(f"    [{idx}/{total}]")
        ds = str(row.get("dataset", ""))
        slide = str(row.get("slide_id", ""))
        key = (ds, slide)
        if key not in cache:
            cache[key] = _load_slide_polygons(client, ds, slide, str(row.get("split", "")))
            if cache[key] is None:
                n_missing += 1
        polygons = cache[key]

        x = float(row.get("x", 0))
        y = float(row.get("y", 0))
        ts = float(row.get("tile_size", 256))
        if polygons:
            frac = _patch_overlap_fraction(x, y, ts, polygons)
            center_inside = _points_in_polygons(
                np.array([x + ts / 2.0]), np.array([y + ts / 2.0]), polygons
            )[0]
        else:
            frac = 0.0
            center_inside = False

        fracs.append(frac)
        mask20.append(int(frac >= TUMOR_GT_THRESHOLD))
        mask_center.append(int(center_inside))

    df["tumor_overlap_fraction"] = fracs
    df["tumor_mask_overlap"] = mask20
    df["tumor_mask_overlap_center"] = mask_center
    print(f"    Done. Slides without XML: {n_missing}/{df['slide_id'].nunique()}.")
    print(f"    Mask-pos (>=20% area): {sum(mask20)} ({sum(mask20) / max(len(df), 1):.1%}), "
          f"center-pos: {sum(mask_center)} ({sum(mask_center) / max(len(df), 1):.1%})")
    return df


def _merge_previous_tissue_fraction(df: pd.DataFrame, previous_csv: str) -> pd.DataFrame:
    if previous_csv.startswith("s3://"):
        prev = read_csv_from_s3(previous_csv.split("/", 3)[3])
    elif os.path.exists(previous_csv):
        prev = pd.read_csv(previous_csv)
    else:
        prev = read_csv_from_s3(previous_csv)
    uid_to_tf = {}
    for _, row in prev.iterrows():
        uid = str(row.get("region_uid", ""))
        tf = row.get("tissue_fraction", np.nan)
        if uid and pd.notna(tf):
            uid_to_tf[uid] = float(tf)
    vals = [uid_to_tf.get(str(uid), np.nan) for uid in df["region_uid"]]
    df["tissue_fraction"] = vals
    n_found = int(sum(1 for v in vals if pd.notna(v)))
    print(f"  Merged tissue_fraction from previous registry: {n_found}/{len(df)} rows")
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
    out["tissue_fraction"] = df.get("tissue_fraction", np.nan)
    out["tumor_mask_overlap"] = df["tumor_mask_overlap"]
    out["tumor_overlap_fraction"] = df.get("tumor_overlap_fraction", np.nan)
    out["tumor_mask_overlap_center"] = df.get("tumor_mask_overlap_center", np.nan)
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
        sampled_parts = []
        for _, group in random_rows.groupby("slide_id", group_keys=False):
            n = min(len(group), 5)
            part = group.sample(n=n, random_state=int(seed_val)).copy()
            part["random_seed"] = seed_val
            sampled_parts.append(part)
        df_random = pd.concat(sampled_parts, ignore_index=True)
        subsets.append(df_random)

    df_random = pd.concat(subsets, ignore_index=True)
    non_random["random_seed"] = 0

    combined = pd.concat([non_random, df_random], ignore_index=True)
    print(f"  Random seeds {RANDOM_SEEDS}: {len(df_random)} random rows generated "
          f"(from {len(random_rows)} source rows)")
    return combined


def _clean_oracle_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop oracle_* rows whose recomputed mask label disagrees with their group.

    oracle_tumor must have tumor_mask_overlap == 1, oracle_non_tumor must
    have tumor_mask_overlap == 0. Returns (cleaned_df, dropped_df).
    """
    before = len(df)
    bad_tumor = (df["selection_source"] == "oracle_tumor") & (df["tumor_mask_overlap"] != 1)
    bad_non = (df["selection_source"] == "oracle_non_tumor") & (df["tumor_mask_overlap"] != 0)
    drop_mask = bad_tumor | bad_non
    dropped = df[drop_mask].copy()
    cleaned = df[~drop_mask].copy()
    n_t = int(bad_tumor.sum())
    n_n = int(bad_non.sum())
    print(f"  Oracle cleanup: oracle_tumor contaminated={n_t}, "
          f"oracle_non_tumor contaminated={n_n}")
    print(f"  Rows: {before} -> {len(cleaned)} ({before - len(cleaned)} dropped)")
    return cleaned, dropped


def _backup_s3_object(client, key: str) -> None:
    """Copy an existing S3 object to a timestamped backup before overwrite."""
    try:
        client.head_object(Bucket=MINIO_BUCKET, Key=key)
    except Exception:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem, suffix = key.rsplit(".", 1)
    backup_key = f"{stem}_{stamp}.{suffix}"
    client.copy_object(
        Bucket=MINIO_BUCKET,
        CopySource={"Bucket": MINIO_BUCKET, "Key": key},
        Key=backup_key,
    )
    print(f"  Backed up {key} -> {backup_key}")


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
    parser.add_argument("--use_s3_images", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Download patch images from S3 for tissue_fraction "
                             "(each archive is downloaded and extracted only once). "
                             "Default: True; use --no-use-s3-images to merge "
                             "tissue_fraction from the previous registry instead")
    parser.add_argument("--cache_dir", default="/tmp/vlm_archive_cache",
                        help="Local cache dir for downloaded patch archives (used with --use_s3_images)")
    parser.add_argument("--previous_registry_csv", default="",
                        help="Previous registry CSV (s3:// or local) to merge tissue_fraction "
                             "from when tissue_fraction is not computed. "
                             "Default: the current registry at --output_s3_prefix/patch_registry.csv")
    parser.add_argument("--output_s3_prefix", default="mil/vlm_patches_registry",
                        help="S3 prefix for output")
    parser.add_argument("--skip_tumor_gt_recompute", action="store_true",
                        help="Keep inherited tile_in_mask as tumor_mask_overlap instead of "
                             "recomputing from annotation XMLs with the 20% area rule")
    parser.add_argument("--skip_oracle_cleanup", action="store_true",
                        help="Keep oracle_* rows even if their recomputed mask label "
                             "disagrees with the group definition")
    args = parser.parse_args()

    local_dir = Path(args.local_archive_dir) if args.local_archive_dir else None
    archive_cache_dir = Path(args.cache_dir) if args.use_s3_images else None

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

    combined = _deduplicate_coordinates(combined)

    if args.local_archive_dir or args.use_s3_images:
        combined = _compute_tissue_fractions(
            combined, local_dir, args.use_s3_images, archive_cache_dir
        )
    else:
        prev_csv = args.previous_registry_csv or f"{args.output_s3_prefix}/patch_registry.csv"
        combined = _merge_previous_tissue_fraction(combined, prev_csv)
        print(f"[registry] tissue_fraction merged from previous registry: {prev_csv}")

    combined = _generate_random_subsets(combined)

    if args.skip_tumor_gt_recompute:
        print("[registry] Skipping tumor GT recompute, keeping inherited tumor_mask_overlap")
    else:
        combined = _compute_tumor_gt(combined, get_s3_client())

    oracle_dropped = pd.DataFrame()
    if args.skip_oracle_cleanup:
        print("[registry] Skipping oracle group cleanup")
    else:
        combined, oracle_dropped = _clean_oracle_groups(combined)

    registry = _build_output_registry(combined)

    _print_summary(registry)

    csv_key = f"{args.output_s3_prefix}/patch_registry.csv"
    parquet_key = f"{args.output_s3_prefix}/patch_registry.parquet"

    client = get_s3_client()
    _backup_s3_object(client, csv_key)
    _backup_s3_object(client, parquet_key)

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

    if not args.skip_oracle_cleanup and not oracle_dropped.empty:
        report_local = os.path.join(tmp_dir, "oracle_cleanup_report.csv")
        oracle_dropped.to_csv(report_local, index=False)
        report_url = upload_to_s3(
            report_local, f"{args.output_s3_prefix}/oracle_cleanup_report.csv"
        )
        print(f"  Cleanup report: {report_url}")

    print(f"[registry] Done. {len(registry)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
