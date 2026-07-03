"""Prompt templates for Quilt-1M / Quilt-LLaVA inference."""

from __future__ import annotations

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

_BASE_RULES = """You are a pathology image assistant. Analyze the provided H&E histology image.

Important rules:
1. Do not provide a final clinical diagnosis.
2. Describe only morphological features that are visible in this image.
3. Do not guess the tissue organ if it is not visible with enough confidence.
   Use tissue_organ="uncertain" instead.
4. Return ONLY a single JSON object. No prose before or after. No markdown
   fences. No backslash escapes inside field names.
5. If the image clearly shows a tissue outside the allowed list, use "other".

tissue_organ must be exactly ONE of:
  "colon", "rectum", "lung", "breast", "kidney", "prostate", "brain",
  "liver", "stomach", "pancreas", "lymph_node", "skin", "bone_marrow",
  "soft_tissue", "other", "uncertain"

tumor_suspicious must be exactly ONE of: "yes", "no", "uncertain".
visual_description_confidence must be exactly ONE of: "low", "medium", "high".
conclusion_confidence must be exactly ONE of: "low", "medium", "high".
should_abstain must be a JSON boolean (true or false).

Return JSON with exactly these fields, in this order:
{
  "tissue_organ": "<one of the allowed values>",
  "tissue_description": "<one short sentence on the visible tissue>",
  "cellularity": "<low | moderate | high, plus one short justification>",
  "architecture": "<preserved | mildly distorted | severely distorted, plus one short justification>",
  "visible_abnormalities": ["<short phrase>", "..."],
  "tumor_suspicious": "yes/no/uncertain",
  "evidence": ["<short morphological feature supporting tumor_suspicious>", "..."],
  "artifacts": ["<short phrase>", "..."],
  "limitations": ["<short phrase>", "..."],
  "visual_description_confidence": "low/medium/high",
  "conclusion_confidence": "low/medium/high",
  "should_abstain": true
}"""

STANDARD_PROMPT = """You are an expert pathology AI assistant analyzing an H&E histology tile.
Anatomical Context: Known Lymph Node section (evaluating for metastatic adenocarcinoma).

Important instructions:
1. Do not provide a final clinical diagnosis name. Describe visual morphology only.
2. Evaluate cellularity and architecture specifically relative to lymphoid tissue.
3. Identify foreign cell populations (e.g., metastatic epithelial sheets, glandular nests, nuclear pleomorphism) infiltrating the lymphoid stroma.
4. Return ONLY a single valid JSON object. No prose, no markdown fences, no backslashes.

Return JSON with exactly these fields:
{
  "tissue_organ": "lymph_node",
  "tissue_description": "<one concise sentence on predominant tissue and cell structures>",
  "predominant_cell_type": "<lymphoid | epithelial | stromal | necrotic>",
  "cellularity": "<low | moderate | high>",
  "architecture": "<preserved lymphoid | mildly distorted | severely effaced/distorted>",
  "visible_abnormalities": ["<specific visual feature e.g. enlarged pleomorphic nuclei>", "..."],
  "nuclear_atypia": "<absent | mild | severe>",
  "mitotic_activity": "<absent | low | high>",
  "tumor_suspicious": "<yes | no | uncertain>",
  "evidence": ["<concrete morphological feature justifying suspicion>", "..."],
  "artifacts": ["<blur | fold | none>"],
  "visual_description_confidence": "<low | medium | high>",
  "should_abstain": false
}"""

SAFE_PROMPT = """You are a pathology image assistant. Analyze the provided H&E histology image.

Important rules:
1. Do not provide a final clinical diagnosis.
2. Describe only morphological features that are clearly visible in this exact image.
3. If the tissue origin is not clearly supported by explicit visible morphology,
   you MUST use tissue_organ="uncertain". Never guess the organ.
4. If the image is blurry, cropped, heavily artifacted, non-diagnostic, or the
   evidence for any conclusion is weak or contradictory, you MUST set
   should_abstain=true.
5. You are NOT allowed to output a diagnosis name (for example: carcinoma,
   adenocarcinoma, lymphoma, melanoma, metastasis, benign/malignant diagnosis).
   Use only descriptive morphology.
6. tumor_suspicious can be "yes" or "no" ONLY when you provide explicit visual
   evidence in the evidence list from this image. If explicit evidence is absent,
   set tumor_suspicious="uncertain" and should_abstain=true.
7. Evidence must be concrete and visual (e.g. gland crowding, nuclear pleomorphism,
   mitotic figures, necrosis, keratinization, mucin, stromal reaction). Do not use
   generic claims like "looks abnormal" or "possible neoplastic process" without
   specific morphology.
8. If should_abstain=true:
   - tissue_organ must be "uncertain" unless organ-defining structures are explicit.
   - tumor_suspicious should usually be "uncertain".
   - keep evidence minimal, factual, and image-grounded.
9. Return ONLY a single JSON object. No prose before or after. No markdown
   fences. No backslash escapes inside field names.

tissue_organ must be exactly ONE of:
  "colon", "rectum", "lung", "breast", "kidney", "prostate", "brain",
  "liver", "stomach", "pancreas", "lymph_node", "skin", "bone_marrow",
  "soft_tissue", "other", "uncertain"

tumor_suspicious must be exactly ONE of: "yes", "no", "uncertain".
visual_description_confidence must be exactly ONE of: "low", "medium", "high".
conclusion_confidence must be exactly ONE of: "low", "medium", "high".
should_abstain must be a JSON boolean (true or false).

Return JSON with exactly these fields, in this order:
{
  "tissue_organ": "<one of the allowed values>",
  "tissue_description": "<one short sentence on the visible tissue>",
  "cellularity": "<low | moderate | high, plus one short justification>",
  "architecture": "<preserved | mildly distorted | severely distorted, plus one short justification>",
  "visible_abnormalities": ["<short phrase>", "..."],
  "tumor_suspicious": "yes/no/uncertain",
  "evidence": ["<short morphological feature supporting tumor_suspicious>", "..."],
  "artifacts": ["<short phrase>", "..."],
  "limitations": ["<short phrase>", "..."],
  "visual_description_confidence": "low/medium/high",
  "conclusion_confidence": "low/medium/high",
  "should_abstain": true
}"""

PROMPTS = {
    "standard": STANDARD_PROMPT,
    "safe": SAFE_PROMPT,
}

PROMPT_VERSIONS = {
    "standard": "standard_v1",
    "safe": "safe_v2_evidence_gated",
}


def get_prompt(variant: str) -> str:
    try:
        return PROMPTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant!r}") from exc


def get_prompt_version(variant: str) -> str:
    try:
        return PROMPT_VERSIONS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt variant: {variant!r}") from exc
