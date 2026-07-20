"""Defensive JSON parsing / normalization for Quilt-LLaVA output (self-contained).

VLM output is noisy: markdown fences, illegal ``\\_`` escapes, trailing prose,
or truncated objects. These helpers ALWAYS return a stable schema so a row is
written for every patch even on malformed output. ``parse_valid`` records
whether a JSON object was actually recoverable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

SCHEMA_VERSION = "vlm_schema_v1"

# Stable output schema. Every output row follows this shape.
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
ALLOWED_TISSUE_ORGAN = {
    "colon", "rectum", "lung", "breast", "kidney", "prostate", "brain",
    "liver", "stomach", "pancreas", "lymph_node", "skin", "bone_marrow",
    "soft_tissue", "other", "uncertain",
}
_TISSUE_ORGAN_ALIASES: dict[str, str] = {
    "colorectal": "colon", "colorectum": "colon", "large_intestine": "colon",
    "large intestine": "colon", "intestine": "colon", "intestines": "colon",
    "small_intestine": "colon", "small intestine": "colon",
    "gastrointestinal": "stomach", "gastrointestinal_tract": "stomach",
    "gi_tract": "stomach", "gi": "stomach", "renal": "kidney",
    "mammary": "breast", "cerebral": "brain", "cerebrum": "brain",
    "cns": "brain", "hepatic": "liver", "lymphnode": "lymph_node",
    "lymph node": "lymph_node", "bonemarrow": "bone_marrow",
    "bone marrow": "bone_marrow", "softtissue": "soft_tissue",
    "soft tissue": "soft_tissue", "connective tissue": "soft_tissue",
    "connective_tissue": "soft_tissue", "tendon": "soft_tissue",
    "tendon_sheath": "soft_tissue",
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
# LLaVA frequently emits illegal markdown escapes like "tissue\_organ".
_INVALID_BACKSLASH_RE = re.compile(r'\\([_*~`#+\-.!])')


def _strip_markdown_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _repair_json_text(text: str) -> str:
    return _INVALID_BACKSLASH_RE.sub(r"\1", text)


def _try_loads(candidate: str) -> Optional[dict]:
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
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
    """Best-effort extraction of a JSON object from raw VLM text.

    1. strip markdown fences; 2. direct json.loads (+ escape repair retry);
    3. fall back to the first-``{`` .. last-``}`` substring; 4. else None.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = _strip_markdown_fences(text)
    parsed = _try_loads(cleaned)
    if parsed is not None:
        return parsed
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        parsed = _try_loads(cleaned[first : last + 1])
        if parsed is not None:
            return parsed
    return None


# Required output keys for a schema-complete object (everything the model is
# asked to emit; schema_version is added by us, not the model).
REQUIRED_SCHEMA_KEYS = tuple(k for k in DEFAULT_SCHEMA if k != "schema_version")


def _schema_valid(parsed: Optional[dict]) -> bool:
    """True only if ``parsed`` already contains every required key with an
    in-range value — checked BEFORE ``normalize_json`` fills/coerces anything.
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

    Returns ``(parsed, provenance)`` where provenance is::

        {"strict_json_valid": bool,  # raw text valid JSON with NO repair
         "parse_valid":       bool,  # a dict recovered by any means
         "repair_stage":      str,   # strict | fence | escape | brace | none
         "schema_valid":      bool}  # parsed dict already schema-complete

    Mirrors ``src/json_utils.parse_with_provenance`` in the parent repo.
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

    # Stage 2: parsed after escape repair (the common \\_ case).
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
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if value.strip():
            return [value.strip()]
        return []
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
    """Return a dict that always follows DEFAULT_SCHEMA (fills + coerces + enums)."""
    out: dict[str, Any] = {
        k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_SCHEMA.items()
    }
    if not isinstance(parsed, dict):
        return out

    for key in ("tissue_description", "cellularity", "architecture"):
        if key in parsed:
            out[key] = _coerce_str(parsed[key])

    if "tissue_organ" in parsed:
        out["tissue_organ"] = _coerce_tissue_organ(parsed["tissue_organ"])
    elif "tissue_description" in parsed:
        out["tissue_organ"] = _coerce_tissue_organ(parsed["tissue_description"])

    for key in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
        if key in parsed:
            out[key] = _coerce_list(parsed[key])

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
    if "confidence" in parsed:  # legacy single-confidence field
        legacy = _coerce_choice(parsed["confidence"], _ALLOWED_CONFIDENCE, "low")
        out["visual_description_confidence"] = legacy
        out["conclusion_confidence"] = legacy

    if "should_abstain" in parsed:
        out["should_abstain"] = _coerce_bool(parsed["should_abstain"], default=True)

    return out
