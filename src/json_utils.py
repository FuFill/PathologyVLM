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

# Stable output schema. Keep in sync with the fixed prompt in run_remote_vlm.py.
DEFAULT_SCHEMA: dict[str, Any] = {
    "tissue_description": "",
    "cellularity": "",
    "architecture": "",
    "visible_abnormalities": [],
    "tumor_suspicious": "uncertain",
    "evidence": [],
    "artifacts": [],
    "limitations": [],
    "confidence": "low",
    "should_abstain": True,
}

_ALLOWED_TUMOR_SUSPICIOUS = {"yes", "no", "uncertain"}
_ALLOWED_CONFIDENCE = {"low", "medium", "high"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` style fences from a string."""
    return _FENCE_RE.sub("", text).strip()


def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from raw VLM text output.

    Strategy:
      1. Strip markdown fences (```json ... ```).
      2. Try direct ``json.loads``.
      3. Fall back to taking the substring between the first ``{`` and the
         last ``}`` and parsing that.
      4. Return ``None`` if everything fails.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = _strip_markdown_fences(text)

    # 1. Direct parse.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Substring between first '{' and last '}'.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = cleaned[first : last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return None

    return None


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


def normalize_json(parsed: Optional[dict]) -> dict:
    """Return a dict that always follows DEFAULT_SCHEMA.

    Missing fields are filled with defaults. Lists are coerced into Python
    lists when possible. ``tumor_suspicious`` and ``confidence`` are
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

    # List fields.
    for key in ("visible_abnormalities", "evidence", "artifacts", "limitations"):
        if key in parsed:
            out[key] = _coerce_list(parsed[key])

    # Enum-like fields.
    if "tumor_suspicious" in parsed:
        out["tumor_suspicious"] = _coerce_choice(
            parsed["tumor_suspicious"], _ALLOWED_TUMOR_SUSPICIOUS, "uncertain"
        )
    if "confidence" in parsed:
        out["confidence"] = _coerce_choice(parsed["confidence"], _ALLOWED_CONFIDENCE, "low")

    # Boolean.
    if "should_abstain" in parsed:
        out["should_abstain"] = _coerce_bool(parsed["should_abstain"], default=True)

    return out
