"""Prompts for the Quilt-LLaVA lymph-node describer (self-contained).

Two variants are available (select with ``--prompt_variant``):

* ``safe`` — the original prompt. It forces a two-regime benign-vs-atypical
  choice and pre-lists specific findings, which the model tends to parrot back
  regardless of the tile (a known source of the yes-bias). Kept for
  reproducing the earlier runs.
* ``neutral`` — the current default. Names NO findings and does NOT force a
  benign/malignant choice; the model describes only what it independently
  observes and may answer ``uncertain`` / abstain.

Both — by design — NEVER emit a clinical diagnosis name and force a single JSON
object. Do not weaken these rules to raise valid-JSON rates. Output is not
guaranteed; always inspect ``raw_response``.
"""

from __future__ import annotations

# Order of fields the model is asked to emit (documentation / reference).
PROMPT_FIELDS = [
    "tissue_organ",
    "tissue_description",
    "cellularity",
    "architecture",
    "visible_abnormalities",
    "tumor_suspicious",
    "evidence",
    "artifacts",
    "limitations",
    "visual_description_confidence",
    "conclusion_confidence",
    "should_abstain",
]

SAFE_PROMPT_VERSION = "safe_v4_neutral_contrastive"

SAFE_PROMPT = """You are a pathology image assistant analyzing an H&E histology tile from a Lymph Node biopsy.

Important rules:
1. Do not provide a final clinical diagnosis name. Describe visual morphology only.
2. Set tissue_organ="lymph_node". Do not guess colon or GI origins.
3. Classify the tile into exactly ONE of two morphological regimes:
   - Regime A (Benign Lymphoid): Uniform lymphocytes, preserved follicular organization, absence of foreign epithelial nests or nuclear atypia. -> tumor_suspicious="no".
   - Regime B (Atypical / Infiltrating): Infiltrating foreign cell populations, severe nuclear pleomorphism, disrupted stroma, or frequent mitotic activity. -> tumor_suspicious="yes".
4. In visible_abnormalities and evidence, write ONLY what you independently observe in the image. If normal lymphoid tissue is preserved, write "none".
5. Return ONLY a single JSON object. No prose before or after. No markdown fences. No backslash escapes inside field names.

tissue_organ must be exactly ONE of:
  "lymph_node", "colon", "rectum", "lung", "breast", "kidney", "prostate", "brain",
  "liver", "stomach", "pancreas", "skin", "bone_marrow", "soft_tissue", "other", "uncertain"

tumor_suspicious must be exactly ONE of: "yes", "no", "uncertain".
visual_description_confidence must be exactly ONE of: "low", "medium", "high".
conclusion_confidence must be exactly ONE of: "low", "medium", "high".
should_abstain must be a JSON boolean (true or false).

Return JSON with exactly these fields, in this order:
{
  "tissue_organ": "lymph_node",
  "cellularity": "<low | moderate | high>",
  "architecture": "<preserved | mildly distorted | severely distorted>",
  "visible_abnormalities": ["<list visible abnormalities observed independently, or write 'none'>"],
  "evidence": ["<list concrete visual evidence justifying your regime choice, or write 'none'>"],
  "tumor_suspicious": "yes/no/uncertain",
  "tissue_description": "<one concise synthesis sentence summarizing the observed features>",
  "artifacts": ["<blur | fold | none>"],
  "limitations": ["<short phrase>"],
  "visual_description_confidence": "low/medium/high",
  "conclusion_confidence": "low/medium/high",
  "should_abstain": false
}"""

# Neutral variant: names NO findings and forces NO benign/malignant choice.
NEUTRAL_PROMPT_VERSION = "neutral_v6_open_observation"

NEUTRAL_PROMPT = """You are a pathology image assistant analyzing an H&E histology tile from a lymph node biopsy.

Rules:
1. Do not provide a clinical diagnosis name. Describe only the visual morphology you observe.
2. Describe ONLY features you can independently see in THIS tile. Do not list findings you cannot point to in the image. If you observe no abnormality, leave the list empty ([]).
3. You are NOT required to decide between "benign" and "malignant". If the tile is bland, ambiguous, or the evidence is insufficient, set tumor_suspicious="uncertain" and should_abstain=true. Abstaining is a valid, correct outcome.
4. Set confidence honestly: "high" only when the features you name are unambiguous and span the tile; "medium" when suggestive but partial; "low" when the tile is bland or ambiguous. When in doubt choose "low".
5. Return ONLY a single JSON object. No prose before or after. No markdown fences. No backslash escapes inside field names.

tissue_organ must be exactly ONE of:
  "lymph_node", "colon", "rectum", "lung", "breast", "kidney", "prostate", "brain",
  "liver", "stomach", "pancreas", "skin", "bone_marrow", "soft_tissue", "other", "uncertain"

tumor_suspicious must be exactly ONE of: "yes", "no", "uncertain".
visual_description_confidence must be exactly ONE of: "low", "medium", "high".
conclusion_confidence must be exactly ONE of: "low", "medium", "high".
should_abstain must be a JSON boolean (true or false).

Return JSON with exactly these fields, in this order:
{
  "tissue_organ": "<one value from the list above>",
  "cellularity": "<low | moderate | high>",
  "architecture": "<preserved | mildly distorted | severely distorted>",
  "visible_abnormalities": ["<only what you independently observe, or leave empty>"],
  "evidence": ["<concrete visual features you can point to in this tile, or leave empty>"],
  "tumor_suspicious": "yes/no/uncertain",
  "tissue_description": "<one concise sentence describing the observed features>",
  "artifacts": ["<blur | fold | none>"],
  "limitations": ["<short phrase>"],
  "visual_description_confidence": "low/medium/high",
  "conclusion_confidence": "low/medium/high",
  "should_abstain": false
}"""

_PROMPTS = {"safe": SAFE_PROMPT, "neutral": NEUTRAL_PROMPT}
_VERSIONS = {"safe": SAFE_PROMPT_VERSION, "neutral": NEUTRAL_PROMPT_VERSION}

PROMPT_VARIANTS = tuple(_PROMPTS.keys())
DEFAULT_PROMPT_VARIANT = "neutral"

# Back-compat module-level aliases (default variant).
PROMPT = NEUTRAL_PROMPT
PROMPT_VERSION = NEUTRAL_PROMPT_VERSION


def get_prompt(variant: str) -> str:
    try:
        return _PROMPTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant!r}") from exc


def get_prompt_version(variant: str) -> str:
    try:
        return _VERSIONS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant!r}") from exc
