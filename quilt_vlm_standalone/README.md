# Quilt-LLaVA Lymph-Node Describer — Standalone

A self-contained package that runs the pathology vision-language model
[`wisdomik/Quilt-Llava-v1.5-7b`](https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b)
(a LLaVA-1.5 fork) on H&E histology tiles and produces a **structured JSON**
morphological micro-description for each image.

This folder is portable: copy it anywhere, `pip install -r requirements.txt`,
and run. It does **not** depend on the parent research repository. It bakes in
the production **"best config"** (the `safe` prompt at temperature `0.4`,
repetition penalty `1.08`, `top_p 0.95`, `max_new_tokens 768`).

> **Safety / scope.** The model describes visual morphology only. By design it
> **never emits a clinical diagnosis name**, and the `should_abstain` flag marks
> insufficient evidence. This is a description/triage aid, **not** a diagnostic
> device. Do not weaken the prompt to raise valid-JSON rates.

---

## 1. What's in this folder

| File | Role |
|---|---|
| `run.py` | Entry point. Discovers images, loads the model, runs inference, writes the JSON. |
| `model.py` | `llava` bootstrap, model loading, and single-image generation. |
| `prompt.py` | The fixed best-config (`safe`) JSON prompt + its version tag. |
| `json_utils.py` | Defensive JSON parsing/normalization so a valid row is always produced. |
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

Describe every tile in a folder (recursive), full precision:

```bash
python run.py --image_dir path/to/tiles --output results.json
```

Quick smoke test — first 5 images, 4-bit to save GPU memory:

```bash
python run.py --image_dir path/to/tiles --output out.json --max_images 5 --load_4bit
```

Supported image extensions: `.jpg .jpeg .png .webp .tif .tiff` (searched
recursively under `--image_dir`).

### Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--image_dir` | *(required)* | Folder of H&E tiles, searched recursively. |
| `--output` | `quilt_vlm_results.json` | Output JSON file (a list of per-image results). |
| `--model_name` | `wisdomik/Quilt-Llava-v1.5-7b` | Hugging Face model id. Must contain `llava`. |
| `--max_images` | `0` | Cap on number of images. `0` = process all. |
| `--temperature` | `0.4` | Sampling temperature. **This is the best-config value.** `0` = greedy/deterministic. Higher = more varied wording (and more drift). |
| `--repetition_penalty` | `1.08` | Penalizes token repetition (`>1.0` to apply). Reduces the model looping on a phrase. |
| `--top_p` | `0.95` | Nucleus sampling cutoff; only used when sampling (`temperature > 0`). |
| `--max_new_tokens` | `768` | Maximum generated tokens per image. The JSON object fits comfortably under this. |
| `--load_4bit` | *(off)* | Use bitsandbytes 4-bit quantization (~half the VRAM, minor quality cost). Needs a GPU. Omit for full fp16. |

**Why these defaults.** Temperature `0.4` + `top_p 0.95` + repetition penalty
`1.08` was the configuration that produced fluent, varied (non-repeating)
descriptions with ~99% valid JSON on the reference benchmark. Lowering the
temperature to ~0.2 was observed to make the model collapse toward a single
canned answer, so `0.4` is the recommended operating point.

---

## 4. Output format

The output file is a **JSON array**; each element is one image's result. Every
element contains three groups of keys:

### 4a. Model description (the schema — always present)

| Field | Type | Values / meaning |
|---|---|---|
| `schema_version` | string | `vlm_schema_v1`. |
| `tissue_organ` | string | Constrained vocabulary; forced to `lymph_node` by the prompt. |
| `tissue_description` | string | One-sentence synthesis of observed morphology. |
| `cellularity` | string | e.g. `low` / `moderate` / `high`. |
| `architecture` | string | e.g. `preserved` / `mildly distorted` / `severely distorted`. |
| `visible_abnormalities` | list | Features the model reports seeing (or `["none"]`). |
| `tumor_suspicious` | string | **`yes` / `no` / `uncertain`** — the headline morphological call. |
| `evidence` | list | Concrete visual features cited to justify the call. |
| `artifacts` | list | e.g. `blur`, `fold`, `none`. |
| `limitations` | list | Short caveats the model raises. |
| `visual_description_confidence` | string | `low` / `medium` / `high` — confidence in the *description*. |
| `conclusion_confidence` | string | `low` / `medium` / `high` — confidence in the *tumor_suspicious call*. |
| `should_abstain` | bool | `true` = evidence insufficient to commit to a regime. |

### 4b. Per-image provenance

| Field | Meaning |
|---|---|
| `image_id` | File stem (filename without extension) — a stable join key. |
| `image_path` | Full path to the source tile. |
| `raw_response` | The model's exact decoded text **before** parsing. Kept for audit/debugging. |
| `parse_valid` | `true` if a JSON object was recoverable from `raw_response`. |
| `json_valid` | Alias of `parse_valid` (compatibility). |
| `error` | Empty string on success; `ExceptionType: message` if that image failed. A row is written even on failure. |

### 4c. Run parameters (echoed on every row)

`model_name`, `prompt_version`, `temperature`, `repetition_penalty`, `top_p`,
`max_new_tokens`, `load_4bit` — so any single result is fully reproducible.

### Example element

```json
{
  "image_id": "tile_007_042",
  "image_path": "tiles/tile_007_042.png",
  "model_name": "wisdomik/Quilt-Llava-v1.5-7b",
  "prompt_version": "safe_v4_neutral_contrastive",
  "schema_version": "vlm_schema_v1",
  "temperature": 0.4,
  "repetition_penalty": 1.08,
  "top_p": 0.95,
  "max_new_tokens": 768,
  "load_4bit": false,
  "raw_response": "{\"tissue_organ\": \"lymph_node\", ... }",
  "parse_valid": true,
  "json_valid": true,
  "tissue_organ": "lymph_node",
  "cellularity": "moderate",
  "architecture": "mildly distorted",
  "visible_abnormalities": ["foreign epithelial nests"],
  "evidence": ["nuclear atypia"],
  "tumor_suspicious": "yes",
  "tissue_description": "A lymph node with foreign epithelial nests and nuclear atypia, suggesting a possible malignancy.",
  "artifacts": ["blur"],
  "limitations": [],
  "visual_description_confidence": "high",
  "conclusion_confidence": "high",
  "should_abstain": false,
  "error": ""
}
```

---

## 5. How it works (internals)

1. **Image discovery** (`model.find_images`) — recursively collects supported
   image files under `--image_dir`, sorted for stable ordering.

2. **`llava` bootstrap** (`model.bootstrap_llava`) — the non-obvious part. The
   Quilt-LLaVA checkpoint uses the *original* LLaVA weight-naming scheme, which
   stock `transformers` cannot load, so the upstream `llava` package is
   required. It is **not** a normal dependency (its `setup.py` pins an
   incompatible `torch`), so at first run the code:
   - `pip install --no-deps` the pinned Quilt-LLaVA commit from GitHub,
   - frees the `transformers` auto-mapping `llava` slot (else it shadows the
     package),
   - stubs out `llava_mpt` (incompatible with `transformers>=4.36`),
   then imports `llava`. This is idempotent — subsequent runs skip the install.

3. **Model load** (`model.load_model`) — loads weights via the upstream
   `load_pretrained_model` (auto device map, optional 4-bit).

4. **Generation** (`model.generate_answer`) — builds the `llava_v1` conversation
   with the `<image>` token, **prefills the assistant turn with `{`** to force
   JSON-only output, and applies a **safe repetition-penalty processor** that
   clamps LLaVA's negative image-token index (`-200`) before gathering (the
   stock processor crashes with a CUDA assert on it).

5. **Parse + normalize** (`json_utils`) — strips markdown fences, repairs the
   illegal `\_` escapes LLaVA tends to emit, and falls back to the
   first-`{`..last-`}` substring. `normalize_json` then fills any missing
   fields, coerces types, and constrains the enums — so **a complete schema row
   is written for every image, even on malformed model output**.

---

## 6. Troubleshooting

- **`pip install of llava failed`** — you need `git` on PATH and network access
  on first run. Behind a proxy/offline, pre-install the pinned commit yourself:
  `pip install --no-deps git+https://github.com/aldraus/quilt-llava@7e70fc39f792ac55de010eb37bff0a6d6f491c13`
- **CUDA out of memory** — add `--load_4bit` (halves VRAM). Full fp16 needs
  ~14 GB.
- **Everything says `tumor_suspicious: yes`** — this is a known bias of this
  checkpoint at temperature 0.4 on some benign lymphoid tissue; the model tends
  to over-call malignancy. Treat `tumor_suspicious` as a soft triage signal, not
  ground truth, and always inspect `evidence` / `raw_response`.
- **Empty output / `error` populated** — check the `error` field on that row;
  the run continues past per-image failures so one bad tile won't abort the job.
