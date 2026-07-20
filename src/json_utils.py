"""JSON parsing and normalization helpers for VLM responses.

These helpers are designed to be defensive: VLM outputs are often noisy and
may contain markdown fences, extra prose, or partial JSON. We always try to
return a stable schema so that downstream code (CSV export, ClearML logging)
never breaks on missing fields.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# Version of the normalized output schema. Bump whenever the fields, their
# names, or their allowed values change, so outputs remain traceable.
SCHEMA_VERSION = "vlm_schema_v1"

# Stable output schema. Keep in sync with the fixed prompt in run_remote_vlm.py.
DEFAULT_SCHEMA: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "tissue_organ": "uncertain",
    "tissue_description": "",
    "cellularity": "",
    "architecture": "",
    "visible_abnormalities": [],
    "tumor_suspicious": "uncertain",
    "evidence": [],
    "artifacts": [],
    "limitations": [],
    "visual_description_confidence": "low",
    "conclusion_confidence": "low",
    "should_abstain": True,
}

_ALLOWED_TUMOR_SUSPICIOUS = {"yes", "no", "uncertain"}
_ALLOWED_CONFIDENCE = {"low", "medium", "high"}
# Closed organ vocabulary. Keep in sync with the prompt in run_remote_vlm.py.
ALLOWED_TISSUE_ORGAN = {
    "colon",
    "rectum",
    "lung",
    "breast",
    "kidney",
    "prostate",
    "brain",
    "liver",
    "stomach",
    "pancreas",
    "lymph_node",
    "skin",
    "bone_marrow",
    "soft_tissue",
    "other",
    "uncertain",
}
# Common free-text aliases the model is likely to emit, mapped to the canonical
# vocabulary. Matching is case-insensitive on a normalized form.
_TISSUE_ORGAN_ALIASES: dict[str, str] = {
    "colorectal": "colon",
    "colorectum": "colon",
    "large_intestine": "colon",
    "large intestine": "colon",
    "intestine": "colon",
    "intestines": "colon",
    "small_intestine": "colon",
    "small intestine": "colon",
    "gastrointestinal": "stomach",
    "gastrointestinal_tract": "stomach",
    "gi_tract": "stomach",
    "gi": "stomach",
    "renal": "kidney",
    "mammary": "breast",
    "cerebral": "brain",
    "cerebrum": "brain",
    "cns": "brain",
    "hepatic": "liver",
    "lymphnode": "lymph_node",
    "lymph node": "lymph_node",
    "bonemarrow": "bone_marrow",
    "bone marrow": "bone_marrow",
    "softtissue": "soft_tissue",
    "soft tissue": "soft_tissue",
    "connective tissue": "soft_tissue",
    "connective_tissue": "soft_tissue",
    "tendon": "soft_tissue",
    "tendon_sheath": "soft_tissue",
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

# Markdown-style escapes Quilt-LLaVA / LLaVA frequently emits inside JSON keys
# and values (e.g. "tissue\_description"). These are invalid per JSON spec.
# Replace each illegal "\<char>" with just "<char>".
_INVALID_BACKSLASH_RE = re.compile(r'\\([_*~`#+\-.!])')


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` style fences from a string."""
    return _FENCE_RE.sub("", text).strip()


def _repair_json_text(text: str) -> str:
    """Repair common LLM JSON quirks that break ``json.loads``.

    Currently:
      * Strip illegal markdown-style backslash escapes like ``\\_`` -> ``_``.
        JSON only permits ``\\"  \\\\  \\/  \\b  \\f  \\n  \\r  \\t  \\uXXXX``.
    """
    return _INVALID_BACKSLASH_RE.sub(r"\1", text)


def _try_loads(candidate: str) -> Optional[dict]:
    """Try json.loads, then try again after repairing common LLM quirks."""
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Second chance: repair markdown-style escapes and retry.
    repaired = _repair_json_text(candidate)
    if repaired != candidate:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw VLM text output.

    Strategy:
      1. Strip markdown fences (```json ... ```).
      2. Try direct ``json.loads``; on failure repair markdown escapes and retry.
      3. Fall back to taking the substring between the first ``{`` and the
         last ``}`` and parsing that (with the same repair retry).
      4. Return ``None`` if everything fails.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = _strip_markdown_fences(text)

    # 1. Direct parse (+ repair retry).
    parsed = _try_loads(cleaned)
    if parsed is not None:
        return parsed

    # 2. Substring between first '{' and last '}'.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = cleaned[first : last + 1]
        parsed = _try_loads(candidate)
        if parsed is not None:
            return parsed

    return None


# Required output keys for a schema-complete object (everything the model is
# asked to emit; schema_version is added by us, not the model).
REQUIRED_SCHEMA_KEYS = tuple(k for k in DEFAULT_SCHEMA if k != "schema_version")


def _schema_valid(parsed: Optional[dict]) -> bool:
    """True only if ``parsed`` already contains every required key with an
    in-range value — checked BEFORE ``normalize_json`` fills/coerces anything.

    This distinguishes a genuinely schema-complete response from an empty ``{}``
    (or partial object) that ``normalize_json`` would silently backfill.
    """
    if not isinstance(parsed, dict):
        return False
    for key in REQUIRED_SCHEMA_KEYS:
        if key not in parsed:
            return False
    if _coerce_choice(parsed.get("tumor_suspicious"), _ALLOWED_TUMOR_SUSPICIOUS, "") == "":
        return False
    for key in ("visual_description_confidence", "conclusion_confidence"):
        if _coerce_choice(parsed.get(key), _ALLOWED_CONFIDENCE, "") == "":
            return False
    if not isinstance(parsed.get("should_abstain"), bool):
        return False
    return True


def parse_with_provenance(text: str) -> tuple[Optional[dict], dict]:
    """Parse VLM output and report HOW it parsed, not just whether it did.

    Returns ``(parsed, provenance)`` where ``parsed`` is the recovered dict (or
    ``None``) and ``provenance`` is::

        {
          "strict_json_valid": bool,  # raw text was valid JSON with NO repair
          "parse_valid":       bool,  # a dict was recovered by any means
          "repair_stage":      str,   # strict | fence | escape | brace | none
          "schema_valid":      bool,  # parsed dict already schema-complete
        }

    The point: ``json_valid`` in older runs was just ``parse_valid`` — true even
    when ``\\_`` escapes had to be repaired. ``strict_json_valid`` is the honest
    "was the model's raw output actually valid JSON" signal.
    """
    prov = {
        "strict_json_valid": False,
        "parse_valid": False,
        "repair_stage": "none",
        "schema_valid": False,
    }
    if not text or not isinstance(text, str):
        return None, prov

    # Stage 0: strict — raw text parses as a dict with no cleaning at all.
    try:
        strict = json.loads(text)
        if isinstance(strict, dict):
            prov["strict_json_valid"] = True
            prov["parse_valid"] = True
            prov["repair_stage"] = "strict"
            prov["schema_valid"] = _schema_valid(strict)
            return strict, prov
    except (json.JSONDecodeError, ValueError):
        pass

    cleaned = _strip_markdown_fences(text)

    # Stage 1: parsed after fence-strip, no escape repair needed.
    try:
        fenced = json.loads(cleaned)
        if isinstance(fenced, dict):
            prov["parse_valid"] = True
            prov["repair_stage"] = "fence"
            prov["schema_valid"] = _schema_valid(fenced)
            return fenced, prov
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 2: parsed after escape repair (the common \_ case).
    repaired = _repair_json_text(cleaned)
    if repaired != cleaned:
        try:
            esc = json.loads(repaired)
            if isinstance(esc, dict):
                prov["parse_valid"] = True
                prov["repair_stage"] = "escape"
                prov["schema_valid"] = _schema_valid(esc)
                return esc, prov
        except (json.JSONDecodeError, ValueError):
            pass

    # Stage 3: brace-salvage (first '{' .. last '}', with repair retry).
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        salvaged = _try_loads(cleaned[first : last + 1])
        if salvaged is not None:
            prov["parse_valid"] = True
            prov["repair_stage"] = "brace"
            prov["schema_valid"] = _schema_valid(salvaged)
            return salvaged, prov

    return None, prov


def _coerce_list(value: Any) -> list:
    """Coerce a value into a list when reasonable."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        # Split comma-separated strings into a list, but only if it looks
        # like a list (contains a comma). Otherwise wrap as single element.
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if value.strip():
            return [value.strip()]
        return []
    # Numbers, dicts, etc. -> wrap as single-element list.
    return [value]


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    return default


def _coerce_choice(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in allowed:
            return v
    return default


def _coerce_tissue_organ(value: Any) -> str:
    """Map a free-text organ string to the canonical ALLOWED_TISSUE_ORGAN set.

    Lower-cases, trims, replaces spaces/dashes with underscores, then:
      1. accepts if already in ALLOWED_TISSUE_ORGAN
      2. accepts via _TISSUE_ORGAN_ALIASES lookup
      3. accepts if any allowed token appears as a substring (e.g.
         "lung adenocarcinoma" -> "lung", "renal cell carcinoma" -> "kidney"
         via the 'renal' alias)
      4. otherwise returns "uncertain"
    """
    if not isinstance(value, str):
        return "uncertain"
    raw = value.strip().lower()
    if not raw:
        return "uncertain"
    norm = raw.replace("-", "_")
    norm_spaced = norm.replace("_", " ")

    if norm in ALLOWED_TISSUE_ORGAN:
        return norm
    if norm in _TISSUE_ORGAN_ALIASES:
        return _TISSUE_ORGAN_ALIASES[norm]
    if norm_spaced in _TISSUE_ORGAN_ALIASES:
        return _TISSUE_ORGAN_ALIASES[norm_spaced]

    # Substring fallback: scan canonical first, then aliases.
    for token in ALLOWED_TISSUE_ORGAN:
        if token == "other":
            continue
        token_spaced = token.replace("_", " ")
        if token in norm or token_spaced in norm_spaced:
            return token
    for alias, canonical in _TISSUE_ORGAN_ALIASES.items():
        if alias in norm or alias in norm_spaced:
            return canonical
    return "uncertain"


def normalize_json(parsed: Optional[dict]) -> dict:
    """Return a dict that always follows DEFAULT_SCHEMA.

    Missing fields are filled with defaults. Lists are coerced into Python
    lists when possible. ``tumor_suspicious`` and the confidence fields are
    constrained to their allowed enumerations. ``should_abstain`` is forced
    to a boolean.
    """
    out: dict[str, Any] = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_SCHEMA.items()}

    if not isinstance(parsed, dict):
        return out

    # String fields.
    for key in ("tissue_description", "cellularity", "architecture"):
        if key in parsed:
            out[key] = _coerce_str(parsed[key])

    # Tissue organ (constrained vocabulary).
    if "tissue_organ" in parsed:
        out["tissue_organ"] = _coerce_tissue_organ(parsed["tissue_organ"])
    elif "tissue_description" in parsed:
        # Best-effort fallback: try to extract organ from a free-text description.
        out["tissue_organ"] = _coerce_tissue_organ(parsed["tissue_description"])

    # List fields.
    for key in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
        if key in parsed:
            out[key] = _coerce_list(parsed[key])

    # Enum-like fields.
    if "tumor_suspicious" in parsed:
        out["tumor_suspicious"] = _coerce_choice(
            parsed["tumor_suspicious"], _ALLOWED_TUMOR_SUSPICIOUS, "uncertain"
        )
    if "visual_description_confidence" in parsed:
        out["visual_description_confidence"] = _coerce_choice(
            parsed["visual_description_confidence"], _ALLOWED_CONFIDENCE, "low"
        )
    if "conclusion_confidence" in parsed:
        out["conclusion_confidence"] = _coerce_choice(
            parsed["conclusion_confidence"], _ALLOWED_CONFIDENCE, "low"
        )
    if "confidence" in parsed:
        legacy = _coerce_choice(parsed["confidence"], _ALLOWED_CONFIDENCE, "low")
        out["visual_description_confidence"] = legacy
        out["conclusion_confidence"] = legacy

    # Boolean.
    if "should_abstain" in parsed:
        out["should_abstain"] = _coerce_bool(parsed["should_abstain"], default=True)

    return out
