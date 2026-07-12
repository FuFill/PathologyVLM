"""Standalone runner for ScaleReasoner-R1 (cross-scale pathology VLM).

ScaleReasoner-R1 (``ChiPhan1110/ScaleReasoner-R1``) is a Qwen2.5-VL-7B model
trained with GRPO for **cross-scale multi-image multiple-choice** pathology
VQA. It is a DIFFERENT task shape from this repo's Quilt-LLaVA describer:

    Quilt-LLaVA (this repo) : 1 patch  -> structured JSON description
    ScaleReasoner-R1        : 3 images (10x/40x/200x) + A-D options
                              -> <think>reasoning</think> <answer>X</answer>

Because the input/output contracts differ, this is NOT a drop-in replacement
and NO "which model is better" comparison is made here. This script only
proves the model loads and emits a parseable ``<answer>`` on a triplet MCQ.

Two backends (mirrors the model card):
  * ``--backend hf``   : Hugging Face Transformers, single process, local GPU.
  * ``--backend vllm`` : query a running ``vllm serve`` OpenAI-compatible
                         endpoint (recommended for batched evaluation).

Input is a JSONL file, one MCQ per line, matching the Scale-VQA schema::

    {"question": "...",
     "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
     "answer": "D",                          # optional (for accuracy)
     "image_path": {"low_mag": "...", "mid_mag": "...", "high_mag": "..."}}

See ``README.md`` in this folder for run instructions.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Exact system prompt from the model card. Do not paraphrase — it is part of
# the model's trained I/O contract and changing it degrades reproducibility.
SYSTEM_PROMPT = (
    "You are a pathology expert. Read the question and options about the image carefully. "
    "Think step by step inside <think> </think>. Then output ONLY the SINGLE best option letter "
    "inside <answer> </answer>.\n"
    "Example: <think>Your reasoning</think> <answer>A</answer>. "
    "Do not include the option text or any extra words inside <answer> </answer> tags."
)

# OFF-CONTRACT describe prompt. This is NOT part of ScaleReasoner-R1's trained
# multiple-choice contract — the model was GRPO-trained to answer 3-image MCQs,
# not to free-text-describe a single tile. It is used ONLY to obtain a
# qualitative microdescription for side-by-side viewing against Quilt-LLaVA.
# No performance comparison is drawn from this mode. Mirrors the Quilt-LLaVA
# safety stance: describe visual morphology only, never a clinical diagnosis.
DESCRIBE_SYSTEM_PROMPT = (
    "You are a pathology image assistant analyzing a single H&E histology tile "
    "from a lymph node. Describe ONLY the visual morphology you observe: tissue "
    "architecture, cellularity, cell populations, and any abnormalities. Do NOT "
    "give a clinical diagnosis or a disease name. Report only features you can "
    "actually see in this tile; if the tissue looks unremarkable, say so. Keep "
    "it to a few concise sentences."
)
DESCRIBE_USER_PROMPT = "Describe the visual morphology of this histology tile."

MODEL_ID = "ChiPhan1110/ScaleReasoner-R1"
MAG_KEYS = ("low_mag", "mid_mag", "high_mag")

_ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def load_mcqs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_question_text(mcq: dict[str, Any]) -> str:
    """Render the question + A-D options into the single text block."""
    q = str(mcq.get("question", "")).strip()
    opts = mcq.get("options", {}) or {}
    lines = [q]
    for letter in ("A", "B", "C", "D"):
        if letter in opts:
            lines.append(f"({letter}) {opts[letter]}")
    return "\n".join(lines)


def resolve_images(mcq: dict[str, Any], image_root: Path) -> list[Path]:
    """Return the 3 magnification image paths in low/mid/high order."""
    ip = mcq.get("image_path", {}) or {}
    paths: list[Path] = []
    for key in MAG_KEYS:
        rel = ip.get(key)
        if not rel:
            raise ValueError(f"MCQ missing image_path[{key}]: {mcq.get('question', '')[:60]}")
        p = Path(rel)
        paths.append(p if p.is_absolute() else image_root / p)
    return paths


def parse_output(text: str) -> dict[str, Any]:
    ans = _ANSWER_RE.search(text or "")
    think = _THINK_RE.search(text or "")
    return {
        "raw": text,
        "answer": ans.group(1).upper() if ans else None,
        "answer_valid": ans is not None,
        "think": think.group(1).strip() if think else "",
    }


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def run_hf(mcqs: list[dict[str, Any]], image_root: Path, max_new_tokens: int) -> list[dict[str, Any]]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f"[scalereasoner] Loading {MODEL_ID} via Transformers ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results: list[dict[str, Any]] = []
    for i, mcq in enumerate(mcqs):
        images = resolve_images(mcq, image_root)
        content = [{"type": "image", "image": str(p)} for p in images]
        content.append({"type": "text", "text": build_question_text(mcq)})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.decode(
            output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        parsed = parse_output(decoded)
        parsed.update({"index": i, "gold": mcq.get("answer")})
        results.append(parsed)
        print(f"[scalereasoner] {i + 1}/{len(mcqs)} -> answer={parsed['answer']} "
              f"valid={parsed['answer_valid']}")
    return results


def run_describe_hf(
    image_paths: list[Path], max_new_tokens: int
) -> list[dict[str, Any]]:
    """OFF-CONTRACT single-image free-text description via HF Transformers.

    Loads ScaleReasoner-R1 and asks it to describe one tile at a time using
    DESCRIBE_SYSTEM_PROMPT. This is NOT the model's trained MCQ task; output is
    qualitative only. Row keys mirror Quilt-LLaVA (`image_id`/`patch_id` = file
    stem) so scripts/report_model_compare.py can join the two model outputs.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    print(f"[scalereasoner] (describe) Loading {MODEL_ID} via Transformers ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results: list[dict[str, Any]] = []
    for i, img_path in enumerate(image_paths):
        messages = [
            {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": DESCRIBE_USER_PROMPT},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.decode(
            output[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        stem = img_path.stem
        results.append(
            {
                "index": i,
                "image_id": stem,
                "patch_id": stem,
                "image_path": str(img_path),
                "raw": decoded,
                "description": decoded.strip(),
            }
        )
        print(f"[scalereasoner] (describe) {i + 1}/{len(image_paths)} -> {stem} "
              f"({len(decoded.strip())} chars)")
    return results


def run_vllm(
    mcqs: list[dict[str, Any]], image_root: Path, max_new_tokens: int, base_url: str
) -> list[dict[str, Any]]:
    from openai import OpenAI

    print(f"[scalereasoner] Querying vLLM endpoint at {base_url} ...")
    client = OpenAI(base_url=base_url, api_key="token")

    results: list[dict[str, Any]] = []
    for i, mcq in enumerate(mcqs):
        images = resolve_images(mcq, image_root)
        content: list[dict[str, Any]] = []
        for p in images:
            b64 = _encode_image(p)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        content.append({"type": "text", "text": build_question_text(mcq)})
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=max_new_tokens,
        )
        decoded = response.choices[0].message.content or ""
        parsed = parse_output(decoded)
        parsed.update({"index": i, "gold": mcq.get("answer")})
        results.append(parsed)
        print(f"[scalereasoner] {i + 1}/{len(mcqs)} -> answer={parsed['answer']} "
              f"valid={parsed['answer_valid']}")
    return results


def _describe_main(args: argparse.Namespace) -> int:
    """OFF-CONTRACT describe mode: free-text morphology on single tiles.

    Input is a folder (``--image_root`` / ``--image_dir``) or, under
    ``--run_remote``, a ClearML Dataset resolved on the agent. Only the HF
    backend is supported here (single-image, no MCQ). Output rows join to
    Quilt-LLaVA outputs by ``patch_id`` (file stem).
    """
    # Path (or ClearML-resolved dir) of images to describe.
    if args.run_remote or args.dataset_name:
        try:
            from clearml import Dataset
        except ImportError as exc:
            print(f"[scalereasoner] ERROR: clearml not installed: {exc}", file=sys.stderr)
            return 2
        print(f"[scalereasoner] Retrieving ClearML dataset: "
              f"project={args.dataset_project!r} name={args.dataset_name!r}")
        image_dir = Path(
            Dataset.get(
                dataset_project=args.dataset_project, dataset_name=args.dataset_name
            ).get_local_copy()
        )
    else:
        image_dir = Path(args.image_dir or args.image_root)

    # find_images lives in the repo's src/; add repo root to path when running
    # from the agent checkout.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.image_utils import find_images

    image_paths = find_images(image_dir, args.max_images)
    print(f"[scalereasoner] (describe) {len(image_paths)} images from {image_dir}")
    if not image_paths:
        print("[scalereasoner] ERROR: no images found.", file=sys.stderr)
        return 1

    results = run_describe_hf(image_paths, args.max_new_tokens)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(results)
    non_empty = sum(1 for r in results if r["description"])
    print(f"[scalereasoner] wrote {out_path}")
    print(f"[scalereasoner] {n} tiles, {non_empty} non-empty descriptions "
          f"({non_empty / n:.0%})")

    task = getattr(args, "_clearml_task", None)
    if task is not None:
        try:
            task.upload_artifact("scalereasoner_describe_jsonl", artifact_object=str(out_path))
            print("[scalereasoner] uploaded artifact scalereasoner_describe_jsonl")
        except Exception as exc:  # noqa: BLE001
            print(f"[scalereasoner] WARNING: artifact upload failed: {exc}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ScaleReasoner-R1 (MCQ or off-contract describe).")
    ap.add_argument("--mode", choices=("mcq", "describe"), default="mcq",
                    help="mcq = trained 3-image multiple-choice contract; "
                         "describe = OFF-CONTRACT single-tile free-text (qualitative only).")
    ap.add_argument("--input", help="JSONL of Scale-VQA-style MCQs (mode=mcq).")
    ap.add_argument("--image_root", default=".", help="Root dir for relative image paths (mcq).")
    ap.add_argument("--image_dir", help="Folder of single images to describe (mode=describe, local).")
    ap.add_argument("--output", default="scalereasoner_outputs.jsonl", help="Output JSONL.")
    ap.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--base_url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible endpoint (backend=vllm, mode=mcq).")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_images", type=int, default=100,
                    help="Cap the number of items (smoke test first, per project policy).")
    # ClearML remote dispatch (mirrors scripts/run_remote_vlm.py).
    ap.add_argument("--run_remote", action="store_true",
                    help="Register a ClearML task and dispatch to --queue_name, then exit.")
    ap.add_argument("--queue_name", default="default", help="ClearML queue for remote execution.")
    ap.add_argument("--project_name", default="Pathology/VLM", help="ClearML project.")
    ap.add_argument("--task_name", default="scalereasoner_describe", help="ClearML task name.")
    ap.add_argument("--dataset_project", default="Pathology/VLM", help="ClearML dataset project.")
    ap.add_argument("--dataset_name", help="ClearML dataset name (describe input on the agent).")
    args = ap.parse_args()

    # --- Optional ClearML task init + remote dispatch --------------------
    if args.run_remote or (args.dataset_name and args.mode == "describe"):
        try:
            from clearml import Task
        except ImportError as exc:
            print(f"[scalereasoner] ERROR: clearml not installed: {exc}", file=sys.stderr)
            return 2
        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            reuse_last_task_id=False,
            output_uri=False,
        )
        task.connect(vars(args))
        try:
            # HF-only remote deps: the full requirements.txt pins vllm, which on
            # a CUDA-12 agent pulls a conflicting CUDA-13 tree and fails pip
            # install. The remote describe run uses the HF backend only.
            req_dir = Path(__file__).resolve().parent
            req_path = req_dir / "requirements-remote.txt"
            if not req_path.is_file():
                req_path = req_dir / "requirements.txt"
            if req_path.is_file():
                task.set_packages(str(req_path))
                print(f"[scalereasoner] Pinned remote packages from: {req_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[scalereasoner] WARNING: could not set packages: {exc}")
        if args.run_remote:
            print(f"[scalereasoner] Dispatching task to queue: {args.queue_name!r}")
            task.execute_remotely(queue_name=args.queue_name, exit_process=True)
        args._clearml_task = task

    if args.mode == "describe":
        return _describe_main(args)

    # --- MCQ mode (trained contract, unchanged) --------------------------
    if not args.input:
        print("[scalereasoner] ERROR: --input is required for mode=mcq.", file=sys.stderr)
        return 1
    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[scalereasoner] ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    mcqs = load_mcqs(in_path)
    if args.max_images and args.max_images > 0:
        mcqs = mcqs[: args.max_images]
    if not mcqs:
        print("[scalereasoner] ERROR: no MCQs to run.", file=sys.stderr)
        return 1

    image_root = Path(args.image_root)
    print(f"[scalereasoner] {len(mcqs)} MCQs, backend={args.backend}, "
          f"image_root={image_root}")

    if args.backend == "hf":
        results = run_hf(mcqs, image_root, args.max_new_tokens)
    else:
        results = run_vllm(mcqs, image_root, args.max_new_tokens, args.base_url)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    valid = sum(1 for r in results if r["answer_valid"])
    graded = [r for r in results if r.get("gold") and r["answer"]]
    correct = sum(1 for r in graded if r["answer"] == str(r["gold"]).upper())
    print(f"[scalereasoner] wrote {out_path}")
    print(f"[scalereasoner] {n} MCQs, {valid} parseable <answer> "
          f"({valid / n:.0%})")
    if graded:
        print(f"[scalereasoner] accuracy on {len(graded)} graded: "
              f"{correct}/{len(graded)} ({correct / len(graded):.0%})")
    else:
        print("[scalereasoner] no gold answers present -> accuracy not computed "
              "(smoke test: only checking output format).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
