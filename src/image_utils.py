"""Image discovery and safe loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

# Supported extensions (case-insensitive).
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def find_images(image_dir: str | Path, max_images: int) -> List[Path]:
    """Recursively find image files under ``image_dir``.

    Returns a sorted, length-limited list of ``Path`` objects.

    Parameters
    ----------
    image_dir : str | Path
        Directory to scan recursively.
    max_images : int
        Maximum number of images to return. Use a very large number to
        effectively disable the limit.
    """
    base = Path(image_dir)
    if not base.exists():
        raise FileNotFoundError(f"Image directory does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {base}")

    results: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            results.append(p)

    results.sort(key=lambda p: str(p).lower())

    if max_images is not None and max_images > 0:
        results = results[:max_images]

    return results


def safe_open_rgb(path: str | Path) -> Image.Image:
    """Open an image and convert it to RGB.

    Raises a clear ``RuntimeError`` on failure so the caller can decide
    whether to skip the sample or abort.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: {p}")

    try:
        img = Image.open(p)
        img.load()  # force decode now so corrupt files fail here, not later
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Failed to open image '{p}': {exc}") from exc

    if img.mode != "RGB":
        try:
            img = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - we want a clear wrapped error
            raise RuntimeError(f"Failed to convert image '{p}' to RGB: {exc}") from exc

    return img
