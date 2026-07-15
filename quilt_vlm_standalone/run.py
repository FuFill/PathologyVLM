"""Quilt-LLaVA lymph-node describer — standalone entry point.

Runs the pathology vision-language model ``wisdomik/Quilt-Llava-v1.5-7b`` on a
folder of H&E histology tiles and writes a structured JSON result per image.
Defaults are the production "best config" (safe prompt, temperature 0.4).

Usage
-----
    python run.py --image_dir path/to/tiles --output results.json

    # smoke test on the first 5 images, 4-bit to save GPU memory:
    python run.py --image_dir tiles --output out.json --max_images 5 --load_4bit

Output
------
A single JSON file: a list of objects, one per image. Each object has the
schema fields (tissue_organ, tumor_suspicious, evidence, confidences, ...),
plus per-image provenance (image_id, image_path, raw_response, parse_valid)
and the run parameters. See README.md for the full field reference.

SAFETY: the model never emits a clinical diagnosis by design; ``should_abstain``
signals insufficient evidence. Do not repurpose this as a diagnostic tool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from json_utils import SCHEMA_VERSION, extract_json, normalize_json
from model import bootstrap_llava, find_images, generate_answer, load_model
from prompt import PROMPT, PROMPT_VERSION

# --- Best-config defaults (the production baseline run) --------------------- #
DEFAULT_MODEL = "wisdomik/Quilt-Llava-v1.5-7b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REPETITION_PENALTY = 1.08
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 768


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Quilt-LLaVA lymph-node describer (standalone). Emits JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--image_dir", required=True,
                    help="Folder of H&E tiles (searched recursively).")
    ap.add_argument("--output", default="quilt_vlm_results.json",
                    help="Output JSON file (list of per-image results).")
    ap.add_argument("--model_name", default=DEFAULT_MODEL,
                    help="Hugging Face model id.")
    ap.add_argument("--max_images", type=int, default=0,
                    help="Cap number of images. 0 = all.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help="Sampling temperature. 0 = greedy.")
    ap.add_argument("--repetition_penalty", type=float,
                    default=DEFAULT_REPETITION_PENALTY,
                    help="Repetition penalty (>1.0 applies).")
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                    help="Nucleus sampling top_p (used when sampling).")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="Generation length limit.")
    ap.add_argument("--load_4bit", action="store_true",
                    help="Use bitsandbytes 4-bit quantization (needs a GPU, "
                         "~half the VRAM). Omit for full fp16.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    images = find_images(args.image_dir, max_images=args.max_images)
    if not images:
        print(f"[run] No images found under {args.image_dir!r}", file=sys.stderr)
        return 1
    print(f"[run] Found {len(images)} image(s). Prompt={PROMPT_VERSION}, "
          f"temp={args.temperature}, rep={args.repetition_penalty}, "
          f"top_p={args.top_p}, 4bit={args.load_4bit}")

    bootstrap_llava()
    tokenizer, model, image_processor, _ = load_model(args.model_name, args.load_4bit)

    run_params = {
        "model_name": args.model_name,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "load_4bit": args.load_4bit,
    }

    results: list[dict] = []
    n_valid = 0
    t0 = time.time()
    for i, img_path in enumerate(images, 1):
        row: dict = {"image_id": img_path.stem, "image_path": str(img_path)}
        row.update(run_params)
        try:
            raw = generate_answer(
                image_path=img_path,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                prompt=PROMPT,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p,
            )
            parsed = extract_json(raw)
            parse_valid = parsed is not None
            normalized = normalize_json(parsed)
            row["raw_response"] = raw
            row["parse_valid"] = parse_valid
            row["json_valid"] = parse_valid  # alias kept for compatibility
            row.update(normalized)
            if parse_valid:
                n_valid += 1
            row["error"] = ""
        except Exception as exc:  # noqa: BLE001 — never let one image abort the run
            row["raw_response"] = ""
            row["parse_valid"] = False
            row["json_valid"] = False
            row.update(normalize_json(None))
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[run] ERROR on {img_path.name}: {row['error']}", file=sys.stderr)
        results.append(row)
        if i % 10 == 0 or i == len(images):
            print(f"[run] {i}/{len(images)} done "
                  f"(valid JSON so far: {n_valid}/{i})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    dt = time.time() - t0
    print(f"[run] Wrote {len(results)} result(s) to {out_path}")
    print(f"[run] JSON-valid: {n_valid}/{len(results)} "
          f"({100 * n_valid / len(results):.0f}%) in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
