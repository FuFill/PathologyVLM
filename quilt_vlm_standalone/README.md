# Quilt-LLaVA Lymph-Node Describer — Standalone

A self-contained package that runs the pathology vision-language model
[`wisdomik/Quilt-Llava-v1.5-7b`](https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b)
(a LLaVA-1.5 fork) on H&E histology tiles and produces a **structured JSON**
morphological micro-description for each image.

This folder is portable: copy it anywhere, `pip install -r requirements.txt`,
and run. It does **not** depend on the parent research repository.

> **Research prototype — not a diagnostic tool.** This is one experimental stage
> of a larger pipeline, provided for research and reproducibility. It has **not**
> been clinically validated. Do not use its output for patient care.
>
> **Safety / scope.** The prompt *instructs* the model to describe visual
> morphology only and to avoid naming a clinical diagnosis, and `should_abstain`
> is meant to flag insufficient evidence. These are prompt-level constraints, not
> guarantees — the model can still produce wrong or off-spec text, so **always
> inspect `raw_response`**. Do not weaken the prompt to raise valid-JSON rates.
>
> **Known bias.** On this checkpoint the model **over-calls
> `tumor_suspicious=yes` on benign lymphoid tissue**. On the reference
> CAMELYON16 subset the morphology call was near chance-level against the tile
> mask (high sensitivity, low specificity). Treat `tumor_suspicious` as a soft
> triage signal only, never as ground truth.

---

## 1. What's in this folder

| File | Role |
|---|---|
| `run.py` | Entry point. Discovers images (folder or manifest), loads the model, runs inference, streams JSONL. |
| `model.py` | `llava` bootstrap, model loading, single-image generation, seeding. |
| `prompt.py` | The two JSON prompt variants (`neutral`, `safe`) + version tags. |
| `json_utils.py` | Defensive JSON parsing/normalization + provenance flags so a valid row is always produced. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

---

## 2. Requirements

- **Python 3.10+**
- A **CUDA GPU** is strongly recommended. The 7B model needs roughly **~14 GB
  VRAM** in full fp16, or **~7 GB** with `--load_4bit`. CPU will technically
  load but is impractically slow.
- **`git` on your PATH** — at first run the code pip-installs the upstream
  `llava` package straight from GitHub (see "How it works" below).
- Network access on first run (to download the model weights from Hugging Face,
  ~14 GB, cached afterwards, and to fetch the `llava` package).

Install:

```bash
pip install -r requirements.txt
```

If your CUDA version isn't 12.9, install the matching `torch` / `torchvision`
build first, then `pip install -r requirements.txt` for the rest.

---

## 3. Usage

Two input modes (mutually exclusive):

**Folder mode** — describe every tile under a folder (recursive). No patch
provenance is carried:

```bash
python run.py --image_dir path/to/tiles --output results.jsonl
```

**Manifest mode** — describe the tiles named in a CSV and carry each row's
provenance (source, label, mask, coordinates, slide id, score) straight into the
output. This is the mode for real MIL/ABMIL-stage integration:

```bash
python run.py --manifest patches.csv --image_root /data --output results.jsonl
```

Quick smoke test — first 10 images, 4-bit, deterministic:

```bash
python run.py --image_dir path/to/tiles --output out.jsonl \
    --max_images 10 --load_4bit --temperature 0 --seed 0
```

Supported image extensions: `.jpg .jpeg .png .webp .tif .tiff`.

### The manifest CSV

The CSV must contain a **path column** — the first found of `patch_path`,
`image_path`, `path`, `minio_path`, `filepath`. Relative paths are resolved
against `--image_root`. Any of these **provenance columns**, when present, are
copied verbatim onto the output row:

```
source  label  prediction  confidence  x  y  slide_id  patch_id
attention_score  attention_rank  tile_size  tile_in_mask  dataset  split  fold
dataset_id  dataset_version
```

Rows with an empty path are skipped.

### Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--image_dir` | *(one of two required)* | Folder of H&E tiles, searched recursively. |
| `--manifest` | *(one of two required)* | CSV naming tiles + carrying provenance. |
| `--image_root` | `""` | Root prefix for resolving relative manifest paths. |
| `--output` | `quilt_vlm_results.jsonl` | Output **JSONL** file (one result object per line). |
| `--model_name` | `wisdomik/Quilt-Llava-v1.5-7b` | Hugging Face model id. Must contain `llava`. |
| `--model_revision` | `""` | Pin the HF weights commit/tag (see reproducibility below). Empty = repo default. |
| `--prompt_variant` | `neutral` | `neutral` (open observation) or `safe` (older two-regime prompt). |
| `--max_images` | `0` | Cap on number of images. `0` = process all. |
| `--temperature` | `0.4` | Sampling temperature. `0` = greedy/deterministic (use this for a control run). |
| `--repetition_penalty` | `1.08` | Penalizes token repetition (`>1.0` to apply). |
| `--top_p` | `0.95` | Nucleus sampling cutoff; only used when sampling (`temperature > 0`). |
| `--max_new_tokens` | `768` | Maximum generated tokens per image. |
| `--seed` | `0` | RNG seed applied before each image. Makes a *sampled* run repeatable; does **not** make `temperature>0` behave like greedy. |
| `--load_4bit` | *(off)* | bitsandbytes 4-bit quantization (~half the VRAM). Needs a GPU. |

**On temperature.** `temperature=0.4` produces varied wording but is
**stochastic** — repeated runs at 0.4 will disagree on some tiles even with a
fixed seed across different hardware/kernels. For a reproducible control, use
`--temperature 0`. Lower non-zero temperatures were observed to collapse toward a
single canned answer.

### Prompt variants

- **`neutral`** (default, `neutral_v6_open_observation`) — names no findings and
  forces no benign/malignant choice. The model describes only what it
  independently observes and may answer `uncertain` / abstain.
- **`safe`** (`safe_v4_neutral_contrastive`) — the older prompt that forces a
  two-regime benign-vs-atypical choice and pre-lists specific findings. The model
  tends to parrot those phrases back, which contributes to the yes-bias. Kept for
  reproducing earlier runs.

---

## 4. Output format

The output is a **JSONL file** (one JSON object per line), **written and flushed
after every image** so a long run's progress survives interruption. A sidecar
`<output>.run_manifest.json` records the full reproducibility context (see §5).

Each line contains four groups of keys:

### 4a. Model description (the schema — always present)

| Field | Type | Values / meaning |
|---|---|---|
| `schema_version` | string | `vlm_schema_v1`. |
| `tissue_organ` | string | Constrained vocabulary; lymph-node context. |
| `tissue_description` | string | One-sentence synthesis of observed morphology. |
| `cellularity` | string | e.g. `low` / `moderate` / `high`. |
| `architecture` | string | e.g. `preserved` / `mildly distorted` / `severely distorted`. |
| `visible_abnormalities` | list | Features the model reports seeing (or empty). |
| `tumor_suspicious` | string | **`yes` / `no` / `uncertain`** — the headline call. See the yes-bias warning above. |
| `evidence` | list | Concrete visual features cited to justify the call. |
| `artifacts` | list | e.g. `blur`, `fold`, `none`. |
| `limitations` | list | Short caveats the model raises. |
| `visual_description_confidence` | string | `low` / `medium` / `high` — confidence in the *description*. |
| `conclusion_confidence` | string | `low` / `medium` / `high` — confidence in the *call*. |
| `should_abstain` | bool | `true` = evidence insufficient. |

### 4b. Per-image provenance + JSON accounting

| Field | Meaning |
|---|---|
| `image_id` | File stem — a stable join key. |
| `image_path` | Full path to the source tile. |
| `raw_response` | The model's exact decoded text **before** parsing. Kept for audit. |
| `strict_json_valid` | `true` only if `raw_response` was valid JSON with **no** repair at all. |
| `parse_valid` | `true` if a JSON object was recoverable by any means (fence-strip, `\_` repair, or brace-salvage). |
| `schema_valid` | `true` if the parsed object already had every required key with in-range enums (before normalization). |
| `repair_stage` | Which stage recovered it: `strict` / `fence` / `escape` / `brace` / `none`. |
| `json_valid` | **Deprecated** alias of `parse_valid`; kept for compatibility. |
| `error` | Empty on success; `ExceptionType: message` if that image failed (a row is still written). |
| *(manifest columns)* | In manifest mode, every carried provenance column (`source`, `label`, `tile_in_mask`, `x`, `y`, `slide_id`, `patch_id`, ...). |

> **strict vs. repaired JSON.** A high `parse_valid` rate can hide the fact that
> almost none of the output was *strict* JSON — this checkpoint frequently emits
> illegal `\_` escapes that only parse after repair. Report `strict_json_valid`
> and `schema_valid` alongside `parse_valid`; do not quote `parse_valid` alone as
> "JSON-valid rate".

### 4c. Run parameters (echoed on every row)

`model_name`, `model_revision`, `llava_git`, `standalone_commit`,
`prompt_variant`, `prompt_version`, `schema_version`, `temperature`,
`repetition_penalty`, `top_p`, `max_new_tokens`, `seed`, `load_4bit` — so any
single result is fully reproducible.

---

## 5. Reproducibility

For a run you can reproduce exactly, pin **all** of:

- **Model weights** — `--model_revision <sha>` pins the Hugging Face commit.
  (The upstream loader doesn't accept a `revision` arg, so the code enforces the
  pin by pre-downloading that revision via `snapshot_download`; if that fails it
  warns, falls back to the default, and still records the requested value so the
  discrepancy is auditable. To hard-pin without the CLI flag, set
  `MODEL_REVISION` in `model.py`.)
- **`llava` loader** — already pinned to a specific commit in `model.py`
  (`QUILT_LLAVA_GIT`).
- **This package's commit** — recorded per-row as `standalone_commit`
  (best-effort `git rev-parse`). When sharing a run, link the **specific commit**
  of this folder, not `main`.
- **Seed + generation params** — `--seed`, `--temperature`, `--repetition_penalty`,
  `--top_p`, `--max_new_tokens`, all echoed on every row and in the run manifest.

The `<output>.run_manifest.json` sidecar collects all of the above plus the input
mode, input path, `image_root`, and image count.

---

## 6. How it works (internals)

1. **Work list** — folder mode (`model.find_images`, recursive, sorted) or
   manifest mode (`run._load_manifest`, one row per named tile with provenance).

2. **`llava` bootstrap** (`model.bootstrap_llava`) — the non-obvious part. The
   Quilt-LLaVA checkpoint uses the *original* LLaVA weight-naming scheme, which
   stock `transformers` cannot load, so the upstream `llava` package is
   required. It is **not** a normal dependency (its `setup.py` pins an
   incompatible `torch`), so at first run the code:
   - `pip install --no-deps` the pinned Quilt-LLaVA commit from GitHub,
   - frees the `transformers` auto-mapping `llava` slot (else it shadows the
     package),
   - stubs out `llava_mpt` (incompatible with `transformers>=4.36`),
   then imports `llava`. Idempotent — subsequent runs skip the install.

3. **Model load** (`model.load_model`) — loads weights via the upstream
   `load_pretrained_model` (auto device map, optional 4-bit), optionally
   pre-pinning `--model_revision`.

4. **Generation** (`model.generate_answer`) — builds the `llava_v1` conversation
   with the `<image>` token, **prefills the assistant turn with `{`** to force
   JSON-only output, applies a **safe repetition-penalty processor** that clamps
   LLaVA's negative image-token index (`-200`) before gathering (the stock
   processor crashes with a CUDA assert on it), and seeds the RNGs immediately
   before `generate`.

5. **Parse + normalize** (`json_utils`) — `parse_with_provenance` records how the
   text parsed (strict / fence / escape / brace) and whether it was
   schema-complete, then `normalize_json` fills missing fields, coerces types,
   and constrains enums — so **a complete schema row is written for every image,
   even on malformed model output**.

---

## 7. Troubleshooting

- **`pip install of llava failed`** — you need `git` on PATH and network access
  on first run. Behind a proxy/offline, pre-install the pinned commit yourself:
  `pip install --no-deps git+https://github.com/aldraus/quilt-llava@7e70fc39f792ac55de010eb37bff0a6d6f491c13`
- **CUDA out of memory** — add `--load_4bit` (halves VRAM). Full fp16 needs
  ~14 GB.
- **Everything says `tumor_suspicious: yes`** — this is the known yes-bias of
  this checkpoint (see the warning at the top). Treat `tumor_suspicious` as a
  soft triage signal, not ground truth, and always inspect `evidence` /
  `raw_response`. Try `--prompt_variant neutral` (the default) rather than `safe`.
- **"JSON-valid rate" looks great but output is messy** — check
  `strict_json_valid`, not `parse_valid`. Most rows parse only after `\_` escape
  repair, so `strict_json_valid` is often near zero even when `parse_valid` is
  ~99%.
- **Empty output / `error` populated** — check the `error` field on that row;
  the run continues past per-image failures, and completed rows are already
  flushed to the JSONL, so one bad tile won't abort or lose the job.
