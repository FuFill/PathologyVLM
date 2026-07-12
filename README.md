# vlm-pathology-baseline

A minimal, reproducible baseline for running a pathology vision-language
model (VLM) on H&E histology images via a remote **ClearML** GPU agent.

This is the first experiment in a larger pipeline. Future versions will
plug in a WSI / MIL stage in front of the VLM:

```
WSI -> MIL/attention -> top-k patches -> VLM -> structured explanation
```

For now this module covers only the VLM step:

```
H&E patch / set of patches -> structured microdescription / explanation / uncertainty
```

---

## 1. Project overview

* **Model (default):** `wisdomik/Quilt-Llava-v1.5-7b`
* **Dataset (default):** `data/quilt-1m` locally, or a ClearML Dataset copy of Quilt-1M
* **Compute:** ClearML GPU agent (remote)
* **Prompt:** fixed, requires JSON-only output
* **Prompt variants:** `standard` (structured) and `safe` (abstain-first)
* **Outputs:** `outputs/vlm_outputs.jsonl`, `outputs/vlm_outputs.csv`
* **ClearML:** input images, scalar metrics, and output files are logged
  and uploaded as task artifacts.

## 2. What this module does

* Loads a small subset of H&E histology images.
* Sends each image to a LLaVA-style VLM with a fixed prompt.
* Forces a structured JSON answer with: tissue description, cellularity,
  architecture, visible abnormalities, suspicion flag, evidence,
  artifacts, limitations, confidence, and an abstain flag.
* Normalizes and persists results as JSONL + CSV.
* Logs everything to ClearML.

## 3. What this module does NOT do

* It **does not** provide a final diagnosis.
* It **is not** a medical decision system.
* It is **only a baseline experiment**, not a clinical tool.
* Outputs must be interpreted by qualified medical professionals.

The prompt instructs the model to describe only visible morphological
features and to set `should_abstain=true` when evidence is insufficient.

## 4. Project structure and what each part does

```
Medlab/
  README.md
  CLAUDE.md                    # guidance for AI agents working in this repo
  requirements.txt
  clearml.conf                 # ClearML creds (gitignored — never commit)
  configs/
    default.yaml
  src/                         # importable library code
    json_utils.py
    image_utils.py
    vlm_inference.py
    prompt_templates.py
  scripts/                     # runnable entrypoints
    prepare_pathgen_subset.py
    upload_clearml_dataset.py
    run_remote_vlm.py
    inspect_dataset.py
    build_c16_subset.py
    report_c16_groups.py
    build_quilt1m_report.py
    generate_wsi_explanation_report.py
  scalereasoner/               # standalone ScaleReasoner-R1 runner (separate task)
    run_scalereasoner.py
    requirements.txt
    README.md
  data/                        # datasets (gitignored)
  outputs/                     # run outputs (gitignored)
```

### `src/` — library modules

| Module | Responsibility |
|---|---|
| `json_utils.py` | The **output schema contract**. Defines `DEFAULT_SCHEMA`, `SCHEMA_VERSION`, and the defensive parser/normalizer that turns noisy VLM text into a stable dict. `tissue_organ` supports `uncertain`; both `visual_description_confidence` and `conclusion_confidence` are always emitted. Every output row is built from this schema, so schema changes flow everywhere automatically. |
| `vlm_inference.py` | Loads Quilt-LLaVA via the upstream `llava` loader (the HF checkpoint uses original-LLaVA class names) and runs single-image generation. Prefills the assistant turn with `{` to force JSON, and uses a CUDA-safe repetition-penalty processor that tolerates LLaVA's `IMAGE_TOKEN_INDEX (-200)`. |
| `image_utils.py` | Safe RGB image opening used by inference. |
| `prompt_templates.py` | The fixed prompts (`standard`, `safe`) and `get_prompt_version()`. The `safe` prompt forces `tissue_organ=lymph_node` — correct for CAMELYON16 (breast metastasis to lymph node); do not change it. |

### `scripts/` — entrypoints

| Script | Responsibility |
|---|---|
| `run_remote_vlm.py` | Main orchestrator. Runs the VLM locally or dispatches to a ClearML GPU agent, then writes `vlm_outputs.jsonl` / `.csv` / `vlm_metadata.csv`. Each row now carries **run provenance** — `prompt_version`, `schema_version`, `temperature`, `repetition_penalty`, `max_new_tokens`, `git_commit`, plus `parse_valid` / `json_valid` — so any ClearML run can be traced back to the exact code and settings. |
| `build_c16_subset.py` | Builds a **stratified** subset of the 4656-patch CAMELYON16 ABMIL export, stratified by `(source, slide_label, tile_in_mask)` with deterministic (no-RNG) selection and a per-stratum floor so small clinically-important groups aren't starved. Copies images into a `slide_id/source/patch.png` layout + `subset_metadata.csv`. |
| `report_c16_groups.py` | Builds a **grouped** evaluation table (not one aggregate number) from a run's `vlm_outputs.jsonl`: per `(source, label, tile_in_mask)` group it reports n, JSON-valid %, parse-valid %, `unique_raw` (distinct raw responses — flags mode collapse), tumor yes/no/uncertain, and abstain %. |
| `upload_clearml_dataset.py` | Uploads a local folder as a ClearML Dataset so the remote agent can fetch it. |
| `prepare_pathgen_subset.py` | Builds a PathGen-1.6M patch subset by joining the gated JSON metadata with TCGA slides fetched via `gdc-client`. |
| `inspect_dataset.py` / `inspect_quilt_1m_image.py` | Sanity-check a dataset JSON / one image's model output. |
| `build_quilt1m_report.py` / `generate_wsi_explanation_report.py` | Proxy-metric report for Quilt-1M outputs / per-slide WSI explanation report. |

### `scalereasoner/` — separate model, separate task

Standalone runner for **ScaleReasoner-R1**, a Qwen2.5-VL-7B cross-scale
multi-image MCQ model. This is a **different task shape** from the
Quilt-LLaVA single-patch JSON describer (3 magnifications + A–D options →
`<think>/<answer>`), so it lives in its own folder with its own
`requirements.txt`, and **no cross-model comparison** is made until both are
run on identical inputs. See `scalereasoner/README.md`.

### Reproducibility note

Outputs are self-describing: every row records the git commit, prompt version,
schema version, and generation parameters. To reproduce a ClearML result,
check out the row's `git_commit` and re-run with the recorded params.

## 5. Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Notes:

* `bitsandbytes` 4-bit quantization requires a CUDA GPU. On a CPU-only
  machine you can still install it, but you must launch inference without
  `--load_4bit`.
* `openslide-python` is a thin wrapper around the native **OpenSlide**
  library and *will not import* until that library is installed and
  visible on the dynamic-loader path.
  * **Windows:** download the binaries from
    <https://openslide.org/download/>, unzip them, and add the `bin/`
    directory to `PATH` (or call `os.add_dll_directory(...)` before
    importing). PowerShell example:
    `$env:Path = "C:\openslide\bin;$env:Path"`.
  * **Linux:** `sudo apt install openslide-tools libopenslide0` (Debian /
    Ubuntu) or use your distro's equivalent.
  * **macOS:** `brew install openslide`.
* The PathGen dataset preparation step also needs the **GDC Data
  Transfer Tool** (`gdc-client`). Download it from
  <https://gdc.cancer.gov/access-data/gdc-data-transfer-tool> and place
  the binary on your `PATH`. You can verify with `gdc-client --version`.

## 6. ClearML credentials setup

You need ClearML credentials to talk to the ClearML server. Pick **one**
of the methods below. Never commit credentials.

### Option A: `clearml-init` (interactive)

```bash
clearml-init
```

This writes credentials to `~/clearml.conf`.

### Option B: local `clearml.conf` in the project folder

1. Create a file named `clearml.conf` in the project root.
2. Paste the configuration block shown in the ClearML web UI under
   **Settings -> Workspace -> Create new credentials**.
3. Tell the ClearML SDK where to find it:

**Linux / macOS:**

```bash
export CLEARML_CONFIG_FILE=./clearml.conf
```

**Windows PowerShell:**

```powershell
$env:CLEARML_CONFIG_FILE = ".\clearml.conf"
```

### Security warning

`clearml.conf`, `*.conf`, `.env`, and `.env.*` are listed in
`.gitignore`. **Do not commit credentials.** If credentials leak, rotate
them immediately from the ClearML web UI.

## 7. Prepare 10-image subset (PathGen-1.6M)

PathGen-1.6M (`jamessyx/PathGen`) is distributed in **two pieces**:

1. A gated metadata file `PathGen-1.6M.json` hosted on Hugging Face.
   Each row has the shape:

   ```json
   {
     "wsi_id":   "TCGA-AA-3844-01Z-00-DX1.<uuid>",
     "position": ["35136", "33344"],
     "caption":  "...",
     "file_id":  "<gdc-file-uuid>"
   }
   ```

2. The actual TCGA whole-slide images (`.svs`), hosted by the
   [GDC](https://portal.gdc.cancer.gov/). They are downloaded with the
   `gdc-client` tool, one slide per `file_id`.

`scripts/prepare_pathgen_subset.py` glues both pieces together: it picks
the first `--max_images` rows with *distinct* `file_id` values
(minimizing the number of slides you need to download), runs
`gdc-client download <file_id>` for any missing slide, opens each slide
with OpenSlide, reads a `--patch_size`×`--patch_size` patch at
`(x, y)` on level 0, and saves it as a JPEG plus a manifest row.

### Step 7a — Get `PathGen-1.6M.json`

The dataset is **gated**. You must:

1. Open <https://huggingface.co/datasets/jamessyx/PathGen> in a browser
   while logged into Hugging Face and accept the dataset terms.
2. Log in on the command line: `huggingface-cli login` (paste a token
   with read access).
3. Download the JSON manually, e.g.:

   ```bash
   hf download jamessyx/PathGen PathGen-1.6M.json \
     --repo-type=dataset --local-dir .
   ```

Place the file anywhere; pass its path via `--pathgen_json`.

### Step 7b — Inspect the JSON (optional)

```bash
python scripts/inspect_dataset.py --pathgen_json ./PathGen-1.6M.json
```

Prints the row count, the keys observed in the first 50 rows, a safe
preview of the first record, and how many distinct `file_id` values are
in that head — useful for sanity checking the download.

### Step 7c — Extract the 10-image subset

The script will call `gdc-client download <file_id>` for each slide it
cannot find under `--wsi_dir`. Slides are large (often >1 GB each); the
first run will spend most of its time downloading.

**Bash:**

```bash
python scripts/prepare_pathgen_subset.py \
  --max_images 10 \
  --out_dir data/he_test_10 \
  --pathgen_json ./PathGen-1.6M.json \
  --wsi_dir data/wsi \
  --patch_size 672
```

**Windows PowerShell:**

```powershell
python scripts/prepare_pathgen_subset.py `
  --max_images 10 `
  --out_dir data/he_test_10 `
  --pathgen_json .\PathGen-1.6M.json `
  --wsi_dir data\wsi `
  --patch_size 672
```

Useful flags:

* `--no_auto_download` — never invoke `gdc-client`; only use slides
  already present in `--wsi_dir`. The script prints which slides it
  needs so you can fetch them on another machine.
* `--gdc_token_file path/to/token.txt` — required for controlled-access
  TCGA slides.
* `--allow_repeat_slides` — drop the distinct-`file_id` constraint
  (useful if you specifically want N patches from the same slide).
* `--gdc_client /path/to/gdc-client` — point at a non-default binary.

The script writes:

* `data/he_test_10/pathgen_0000.jpg` ... `pathgen_0009.jpg`
* `data/he_test_10/manifest.jsonl` with one row per saved patch
  carrying `image_id`, `image_path`, `dataset`, `source_index`,
  `wsi_id`, `file_id`, `x`, `y`, `patch_size`, `magnification`,
  `source`, `question_or_instruction`, and `reference_answer`
  (the PathGen caption).

### Layouts recognized for `--wsi_dir`

Both layouts work; the script will discover either:

```
data/wsi/<file_id>/<wsi_id>.svs        # produced by `gdc-client download <file_id>`
data/wsi/<wsi_id>.svs                  # flat layout, e.g. moved by hand
```

## 8. Upload ClearML Dataset

The remote GPU agent cannot read your local Windows filesystem. Upload
the subset as a ClearML Dataset:

**Bash:**

```bash
python scripts/upload_clearml_dataset.py \
  --dataset_project Pathology/VLM \
  --dataset_name quilt-1m_test_40 \
  --folder data/quilt-1m
```

**Windows PowerShell:**

```powershell
python scripts/upload_clearml_dataset.py `
  --dataset_project Pathology/VLM `
  --dataset_name quilt-1m_test_40 `
  --folder data\quilt-1m
```

The script prints the dataset project, name, ID, and number of files.

## 9. Run inference

To run directly on the local Quilt-1M folder:

**Bash:**

```bash
python scripts/run_remote_vlm.py \
  --image_dir data/quilt-1m \
  --model_name wisdomik/Quilt-Llava-v1.5-7b \
  --prompt_variant standard \
  --max_images 10 \
  --load_4bit
```

**Windows PowerShell:**

```powershell
python scripts/run_remote_vlm.py `
  --image_dir .\data\quilt-1m `
  --model_name wisdomik/Quilt-Llava-v1.5-7b `
  --prompt_variant standard `
  --max_images 10 `
  --load_4bit
```

This local path does not require ClearML credentials.

For a safer prompt, replace `standard` with `safe`.

If you want ClearML remote execution, upload the folder as a ClearML
Dataset and run that dataset through a GPU agent:

Make sure a ClearML agent is running on a GPU machine and listening to
the queue you target (`gpu` by default).

**Bash:**

```bash
python scripts/run_remote_vlm.py \
  --run_remote \
  --queue_name gpu \
  --dataset_project Pathology/VLM \
  --dataset_name quilt-1m_test_40 \
  --model_name wisdomik/Quilt-Llava-v1.5-7b \
  --prompt_variant safe \
  --max_images 10 \
  --load_4bit
```

**Windows PowerShell:**

```powershell
python scripts/run_remote_vlm.py `
  --run_remote `
  --queue_name gpu `
  --dataset_project Pathology/VLM `
  --dataset_name quilt-1m_test_40 `
  --model_name wisdomik/Quilt-Llava-v1.5-7b `
  --prompt_variant safe `
  --max_images 10 `
  --load_4bit
```

When `--run_remote` is set, the script registers the task with ClearML
and immediately exits the local process. The agent then pulls the task,
installs dependencies, and runs inference on the GPU. Do not combine
`--run_remote` with `--image_dir`.

## 9b. CAMELYON16 stratified baseline

The `data/vlm_patches_wsi_abmil/` export has 4656 ABMIL patches. For a clean
baseline, do **not** run all of them at once — build a stratified subset,
run it, then report by group.

**1. Build the subset** (deterministic, ~400 patches by default):

```bash
python scripts/build_c16_subset.py --target_n 400 --floor 20
```

This writes `data/c16_stratified_subset/` (images in `slide_id/source/` layout
+ `subset_metadata.csv`) and prints the realized per-group counts.

**2. Upload it as a ClearML Dataset:**

```bash
python scripts/upload_clearml_dataset.py \
  --dataset_project Pathology/VLM \
  --dataset_name c16_stratified_subset \
  --folder data/c16_stratified_subset
```

**3. Run the VLM** on the subset (smoke-test with `--max_images 100` first),
then **4. build the grouped report:**

```bash
python scripts/report_c16_groups.py --input outputs/<run_dir>/vlm_outputs.jsonl
```

The report is intentionally **not** a single number. Watch `unique_raw` per
group: if it is far below `n`, the model is repeating itself (mode collapse)
rather than reading each patch — a failure a single aggregate would hide.

## 11. Inspect one image

To see exactly what the model printed and all CSV fields for one image:

**Bash:**

```bash
python scripts/inspect_quilt_1m_image.py \
  --image 04dfb3d9-d5fe-4948-bac9-2a950476ca1d_1.jpg
```

**Windows PowerShell:**

```powershell
python scripts/inspect_quilt_1m_image.py `
  --image 04dfb3d9-d5fe-4948-bac9-2a950476ca1d_1.jpg
```

You can pass `--image` as a basename, `image_id`, or any unique path fragment.

To run **locally** instead (e.g., for debugging on a small GPU), drop
`--run_remote`:

```bash
python scripts/run_remote_vlm.py \
  --image_dir data/quilt-1m \
  --model_name wisdomik/Quilt-Llava-v1.5-7b \
  --prompt_variant standard \
  --max_images 2
```

## 10. Expected outputs

After the remote task completes you will find on the agent (and as
ClearML artifacts):

* `outputs/vlm_outputs.jsonl` — one JSON object per image with the raw
  response and the normalized fields.
* `outputs/vlm_outputs.csv` — same data, flattened for spreadsheets.

Each row contains:

```
image_id, image_path, model_name, raw_response, json_valid,
tissue_description, cellularity, architecture, visible_abnormalities,
tumor_suspicious, evidence, artifacts, limitations,
visual_description_confidence, conclusion_confidence,
should_abstain, error
```

### Output JSON schema

```json
{
  "tissue_description": "",
  "cellularity": "",
  "architecture": "",
  "visible_abnormalities": [],
  "tumor_suspicious": "yes/no/uncertain",
  "evidence": [],
  "artifacts": [],
  "limitations": [],
  "visual_description_confidence": "low/medium/high",
  "conclusion_confidence": "low/medium/high",
  "should_abstain": true
}
```

Missing fields are filled with safe defaults so a row is always written
even if the model's reply is malformed.

### Future input metadata contract

When this module is wired to the full WSI / MIL pipeline, each input
will carry richer metadata. The code is prepared for it:

```json
{
  "case_id": "",
  "slide_id": "",
  "patch_path": "",
  "x": 0,
  "y": 0,
  "magnification": "",
  "source": "random/MIL_topk/oracle_mask/thumbnail",
  "model_score": null,
  "true_label": null,
  "mask_overlap": null
}
```

For this first test, only `image_id`, `image_path`, and dataset
metadata are required.

## 11. ClearML artifacts and logs

In the ClearML web UI, open the task and check:

* **Console** — live agent logs.
* **Scalars**
  * `progress / processed_images`
  * `quality / json_valid_rate`
* **Plots / Debug Samples** — `input_images` debug images.
* **Single values** — `num_images`, `valid_json_rate`.
* **Artifacts** — `vlm_outputs_jsonl`, `vlm_outputs_csv`.
* **Configuration / Args** — every CLI argument used.

## 12. Troubleshooting

**Wrong ClearML queue name.**
The task is registered but never picked up by an agent. Verify the
queue name in the ClearML UI under *Workers and Queues* and pass the
exact same value to `--queue_name`.

**ClearML agent is not running.**
The task stays in *Pending* forever. On the GPU machine run:
`clearml-agent daemon --queue gpu --gpus 0 --foreground`.

**Credentials are not found.**
Symptom: `Failed to connect to ClearML server` or `ClearML.conf was not
found`. Either run `clearml-init` or set `CLEARML_CONFIG_FILE` (see
section 6).

**`CLEARML_CONFIG_FILE` is not set.**
The SDK falls back to `~/clearml.conf`. If you keep credentials only in
the project folder, you must export the variable in the same shell that
runs the script.

**Remote GPU cannot see local Windows paths.**
Never pass a `C:\Users\...` path to the remote task. Always upload the
data as a ClearML Dataset first (section 8). On the agent, the dataset
is fetched via `Dataset.get(...).get_local_copy()`.

**No images found in the ClearML Dataset.**
Either the dataset is empty or the file extensions are not in the
supported list (`.jpg .jpeg .png .webp .tif .tiff`). Re-run
`upload_clearml_dataset.py` for `data/quilt-1m/` and confirm the
folder contains JPEGs before uploading.

**PathGen JSON download fails with 401 / "gated dataset".**
Open <https://huggingface.co/datasets/jamessyx/PathGen> while logged in,
accept the dataset terms, then `huggingface-cli login` on the command
line and re-run `hf download jamessyx/PathGen PathGen-1.6M.json
--repo-type=dataset --local-dir .`.

**`gdc-client: command not found` / `prepare_pathgen_subset.py` cannot download slides.**
Install the GDC Data Transfer Tool from
<https://gdc.cancer.gov/access-data/gdc-data-transfer-tool> and add the
binary to your `PATH`. Verify with `gdc-client --version`. As a last
resort, run with `--no_auto_download` after manually downloading the
required slides into `--wsi_dir`.

**`gdc-client` exits non-zero (403 / access denied).**
The `file_id` points to a controlled-access TCGA slide. Request access
through dbGaP, obtain a user token from the GDC portal, and re-run with
`--gdc_token_file path/to/token.txt`. Or skip the entry by selecting
different rows (PathGen-1.6M has 1.6M of them).

**`prepare_pathgen_subset.py` cannot import openslide.**
`openslide-python` requires the native OpenSlide library on the
dynamic-loader path. See section 5 — on Windows, prepend the OpenSlide
`bin/` directory to `PATH` before launching Python; on Linux install
`libopenslide0`; on macOS `brew install openslide`.

**`OpenSlide` reads a black or mostly white patch.**
The `(x, y)` in PathGen-1.6M is a level-0 pixel coordinate. If the slide
has been re-processed (re-mosaicked, lossy reconverted, or replaced by
a different vendor scan), positions can fall outside tissue. Re-inspect
the slide thumbnail and either pick a different PathGen row for that
`file_id` or accept it as a known limitation.

**Disk fills up before all 10 slides finish downloading.**
TCGA `.svs` files routinely exceed 1 GB. Plan for ~20 GB of free space
for a 10-slide subset, mount a larger volume at `--wsi_dir`, or
process slides one by one with `--max_images 1` and delete each slide
after its patch is extracted.

**Remote machine cannot download the Hugging Face model.**
The agent host needs outbound HTTPS to `huggingface.co`. If it is
firewalled, log in on the agent (`huggingface-cli login`), pre-cache
the model under `~/.cache/huggingface`, or set `HF_HOME` and
`HF_HUB_OFFLINE=1` on the agent.

**CUDA is not available.**
The script logs `CUDA available: False` and falls back to CPU. The
QUILT-LLaVA 7B model is not realistically runnable on CPU. Confirm the
agent was started with `--gpus`, that NVIDIA drivers are visible
(`nvidia-smi`), and that PyTorch was installed with a matching CUDA
build.

**GPU out of memory.**
Use `--load_4bit`. If that is not enough, lower `--max_new_tokens`,
process fewer images per task, or switch to a larger GPU.

**bitsandbytes error.**
`bitsandbytes` needs a CUDA build that matches the GPU drivers and a
recent enough `accelerate`. Re-install with
`pip install -U bitsandbytes accelerate`. On CPU-only hosts, simply
drop `--load_4bit`.

**Model class mismatch for QUILT-LLaVA.**
Some forks ship custom classes (e.g., `LlavaLlamaForCausalLM`) that
`LlavaForConditionalGeneration.from_pretrained` cannot load. The
script raises a clear error in that case. Workarounds: pin a
transformers version compatible with the checkpoint, or replace
`LlavaForConditionalGeneration` in `src/vlm_inference.py` with the
class indicated in the model card.

**Invalid JSON output.**
`json_valid=false` rows preserve the raw response in `raw_response`
and still fill the schema with defaults. The `json_valid_rate` scalar
in ClearML tracks this over time.

**Model generates a diagnosis despite the prompt.**
The fixed prompt explicitly forbids it, but VLMs can ignore
instructions. Flag such rows in post-processing (e.g., scan
`raw_response` for diagnostic terms) and treat them as policy
violations. Do **not** weaken the prompt.

## 13. Next steps

* Add a second VLM and compare structured outputs.
* Plug in the WSI / MIL stage and start consuming the full input
  metadata contract.
* Add automatic validation that the model is not producing diagnostic
  language.
* Add slide-level aggregation across multiple patches.

---

## Quick reference: final commands

### Local setup

```bash
pip install -r requirements.txt
```

### Use local ClearML config

Linux / macOS:

```bash
export CLEARML_CONFIG_FILE=./clearml.conf
```

Windows PowerShell:

```powershell
$env:CLEARML_CONFIG_FILE = ".\clearml.conf"
```

### Inspect dataset

Bash:

```bash
python scripts/inspect_dataset.py --pathgen_json ./PathGen-1.6M.json
```

Windows PowerShell:

```powershell
python scripts/inspect_dataset.py --pathgen_json .\PathGen-1.6M.json
```

### Run local inference

Bash:

```bash
python scripts/upload_clearml_dataset.py \
  --dataset_project Pathology/VLM \
  --dataset_name quilt-1m_test_40 \
  --folder data/quilt-1m
```

Windows PowerShell:

```powershell
python scripts/upload_clearml_dataset.py `
  --dataset_project Pathology/VLM `
  --dataset_name quilt-1m_test_40 `
  --folder .\data\quilt-1m
```

### Upload dataset

Bash:

```bash
python scripts/run_remote_vlm.py --image_dir data/quilt-1m --model_name wisdomik/Quilt-Llava-v1.5-7b --prompt_variant standard --max_images 10 --load_4bit
```

Windows PowerShell:

```powershell
python scripts/run_remote_vlm.py --image_dir .\data\quilt-1m --model_name wisdomik/Quilt-Llava-v1.5-7b --prompt_variant standard --max_images 10 --load_4bit
```

### Run remote inference

Bash:

```bash
python scripts/run_remote_vlm.py --run_remote --queue_name gpu --dataset_project Pathology/VLM --dataset_name quilt-1m_test_40 --model_name wisdomik/Quilt-Llava-v1.5-7b --prompt_variant safe --max_images 10 --load_4bit
```

Windows PowerShell:

```powershell
python scripts/run_remote_vlm.py --run_remote --queue_name gpu --dataset_project Pathology/VLM --dataset_name quilt-1m_test_40 --model_name wisdomik/Quilt-Llava-v1.5-7b --prompt_variant safe --max_images 10 --load_4bit
```

> Tip: ClearML reads `clearml.conf` from your home dir by default. To use
> the repo-local one for the current PowerShell session run
> `$env:CLEARML_CONFIG_FILE = (Resolve-Path .\clearml.conf).Path` before
> the commands above. Add `$env:PYTHONUNBUFFERED = "1"` to see live
> upload progress.

```
python scripts\run_remote_vlm.py `
  --run_remote `
  --queue_name d33b6fa94d02482c818d4e0d45ae31cb `
  --project_name Pathology/VLM `
  --task_name test_100_safe_v4_temp04 `
  --dataset_project Pathology/VLM `
  --dataset_name vlm_patches_wsi_abmil `
  --metadata_csv vlm_patch_metadata.csv `
  --prompt_variant safe `
  --max_images 100 `
  --max_new_tokens 768 `
  --temperature 0.4 `
  --repetition_penalty 1.08 `
  --output_dir outputs\test_100_safe_v4_temp04
  ```
