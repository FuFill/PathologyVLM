# ScaleReasoner-R1 runner

Self-contained folder for running **ScaleReasoner-R1**
([`ChiPhan1110/ScaleReasoner-R1`](https://huggingface.co/ChiPhan1110/ScaleReasoner-R1)),
a Qwen2.5-VL-7B pathology VLM trained with GRPO for **cross-scale multi-image
multiple-choice** VQA (MICCAI 2026).

## Why this is separate from the Quilt-LLaVA pipeline

This is a **different task shape** from the rest of this repo:

| | Input | Output |
|---|---|---|
| Quilt-LLaVA (main repo) | 1 patch (256px) | structured JSON description |
| ScaleReasoner-R1 (here) | 3 images: 10x / 40x / 200x + options A–D | `<think>…</think> <answer>X</answer>` |

Because the input/output contracts differ, **no "which model is better"
comparison is made here.** The model card reports ScaleReasoner-R1 beating
Quilt-LLaVA on its *own* cross-scale MCQ benchmark, but that is not the same
task as our single-patch C16 baseline. Any comparison on identical patches is
**deferred to discussion** — this folder only provides a technical smoke path
(model loads, produces a parseable `<answer>`).

## Layout

```
scalereasoner/
├── run_scalereasoner.py   # runner: HF Transformers or vLLM backend
├── requirements.txt       # multi-image VLM deps (separate from repo root)
└── README.md              # this file
```

## Install

Requires a CUDA 12.1+ GPU host with PyTorch 2.3+. Use a fresh environment so
the Qwen2.5-VL deps don't collide with the Quilt-LLaVA (`llava`, torch 2.0.1)
stack used elsewhere in this repo.

```bash
python -m venv .venv-scalereasoner && source .venv-scalereasoner/bin/activate
pip install -r scalereasoner/requirements.txt
```

## Input format

One MCQ per line (JSONL), matching the Scale-VQA schema:

```json
{"question": "What is the most likely diagnosis?",
 "options": {"A": "Benign", "B": "DCIS", "C": "Invasive carcinoma", "D": "Normal"},
 "answer": "C",
 "image_path": {"low_mag": "wsi1/10_a.jpg", "mid_mag": "wsi1/40_a.jpg", "high_mag": "wsi1/200_a.jpg"}}
```

- `answer` is optional. If present, accuracy is computed; if absent, only the
  output-format (`<answer>` parseable) rate is reported — which is all a smoke
  test needs.
- Relative `image_path` values are resolved against `--image_root`.

## Run (smoke test first — project policy)

Per project policy, **always start with a capped run** (`--max_images 100`) to
inspect output quality before scaling up.

**Backend 1 — HF Transformers (single GPU, simplest):**

```bash
python scalereasoner/run_scalereasoner.py \
    --input path/to/scale_vqa_test.jsonl \
    --image_root path/to/scale_vqa_images \
    --backend hf \
    --max_images 100 \
    --output scalereasoner/outputs_smoke_100.jsonl
```

**Backend 2 — vLLM server (batched eval):**

```bash
# Terminal 1: serve the model
vllm serve ChiPhan1110/ScaleReasoner-R1 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --limit-mm-per-prompt.image 5

# Terminal 2: query it
python scalereasoner/run_scalereasoner.py \
    --input path/to/scale_vqa_test.jsonl \
    --image_root path/to/scale_vqa_images \
    --backend vllm --base_url http://localhost:8000/v1 \
    --max_images 100 \
    --output scalereasoner/outputs_smoke_100.jsonl
```

The system prompt is baked into `run_scalereasoner.py` verbatim from the model
card — it is part of the model's trained I/O contract; do not paraphrase it.

## Interpreting output

Each output row has `answer` (parsed A–D or `null`), `answer_valid`, `think`,
and `raw`. The runner prints the parseable-`<answer>` rate and, if gold answers
are present, accuracy. Once the smoke run looks good, drop `--max_images` (or
raise it) to run the full set.

## Getting the Scale-VQA data

The MCQ benchmark is on HuggingFace:
[`iMVR-PL/Scale-VQA`](https://huggingface.co/datasets/iMVR-PL/Scale-VQA).
This repo does not vendor it; download it separately and point `--input` /
`--image_root` at the downloaded splits.
