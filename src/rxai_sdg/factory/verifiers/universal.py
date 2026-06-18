"""Language-agnostic (universal) constraint checkers (spec §9).

These checkers do not depend on the natural language of the answer: structural
format validity (JSON/YAML/markdown), token-count length, bullet counts and line
counts. They register under the wildcard language ``"*"``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import checker

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


def strip_code_fences(text: str) -> str:
    """Remove a single surrounding ```lang ... ``` markdown code fence."""
    t = text.strip()
    if t.startswith("```"):
        # drop first fence line and trailing fence
        lines = t.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return t


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'-]*\b", text))


_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


def count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if _BULLET_RE.match(line))


def count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# format checkers
# ---------------------------------------------------------------------------

@checker("json_valid")
def json_valid(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    candidate = strip_code_fences(answer)
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError) as exc:
        return False, f"invalid JSON: {exc}"
    want = params.get("top_type")
    if want == "object" and not isinstance(parsed, dict):
        return False, "JSON parsed but top-level is not an object"
    if want == "array" and not isinstance(parsed, list):
        return False, "JSON parsed but top-level is not an array"
    return True, "valid JSON"


@checker("yaml_valid")
def yaml_valid(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    try:
        import yaml  # optional dependency
    except ImportError:  # pragma: no cover - environment dependent
        return False, "yaml package not available to verify"
    candidate = strip_code_fences(answer)
    try:
        parsed = yaml.safe_load(candidate)
    except Exception as exc:  # yaml.YAMLError and friends
        return False, f"invalid YAML: {exc}"
    if parsed is None:
        return False, "YAML parsed to None (empty document)"
    # Reject the degenerate case where a plain sentence parses as a scalar string.
    if isinstance(parsed, str):
        return False, "YAML parsed to a bare scalar string, not a structure"
    return True, "valid YAML"


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


@checker("markdown_table")
def markdown_table(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    lines = answer.splitlines()
    has_row = any(line.count("|") >= 2 for line in lines)
    has_sep = any(_TABLE_SEP_RE.match(line) for line in lines)
    if has_row and has_sep:
        return True, "markdown table detected"
    return False, "no markdown table (need a header row and a |---|---| separator)"


@checker("markdown_format")
def markdown_format(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    has_heading = bool(re.search(r"^#{1,6}\s+\S", answer, re.MULTILINE))
    has_bullet = count_bullets(answer) > 0
    has_emphasis = bool(re.search(r"(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`)", answer))
    if has_heading or has_bullet or has_emphasis:
        return True, "markdown formatting present"
    return False, "no markdown structure detected"


# ---------------------------------------------------------------------------
# length / structure checkers
# ---------------------------------------------------------------------------

@checker("length_tokens")
def length_tokens(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    n = word_count(answer)
    max_words = params.get("max_words")
    min_words = params.get("min_words")
    if max_words is not None and n > max_words:
        return False, f"too long: {n} words > max {max_words}"
    if min_words is not None and n < min_words:
        return False, f"too short: {n} words < min {min_words}"
    return True, f"length OK ({n} words)"


@checker("n_bullets")
def n_bullets(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    count = count_bullets(answer)
    n = params.get("n")
    if n is not None and count != n:
        return False, f"expected {n} bullets, found {count}"
    lo, hi = params.get("min"), params.get("max")
    if lo is not None and count < lo:
        return False, f"too few bullets: {count} < {lo}"
    if hi is not None and count > hi:
        return False, f"too many bullets: {count} > {hi}"
    if n is None and lo is None and hi is None and count == 0:
        return False, "no bullet points found"
    return True, f"bullet count OK ({count})"


@checker("n_lines")
def n_lines(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    count = count_nonempty_lines(answer)
    n = params.get("n")
    if n is not None and count != n:
        return False, f"expected {n} non-empty lines, found {count}"
    return True, f"line count OK ({count})"


# ---------------------------------------------------------------------------
# fact recall / update (value-presence) checkers
# ---------------------------------------------------------------------------
#
# These are language-agnostic: the expected value is carried in
# ``params["value"]`` by the simulator (drawn from the FactLedger) so the
# verifier never needs the ledger object itself.

def _normalise(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


# A recall that merely contains the value inside a refusal / "I can't access ..."
# disclaimer is NOT a real recall (fix H, doc-10 Tier-3-in-a-heading-while-refusing).
_RECALL_REFUSAL_RE = re.compile(
    r"\b(?:i (?:can'?t|cannot|don'?t|do not) (?:access|recall|remember|retain|store)|"
    r"i (?:have no|don'?t have) (?:access|memory|record)|"
    r"as an ai|i'?m (?:sorry|unable))\b",
    re.IGNORECASE,
)


def _confirms_value(answer: str, value: Any, match: str) -> bool:
    """The value is present AND the answer is not a refusal/can't-access disclaimer."""
    if _RECALL_REFUSAL_RE.search(answer or ""):
        return False
    return _value_present(answer, value, match)


def _value_present(answer: str, value: Any, match: str) -> bool:
    target = _normalise(str(value))
    hay = _normalise(answer)
    if not target:
        return False
    if match == "fuzzy":
        target_tokens = set(target.split())
        hay_tokens = set(hay.split())
        if not target_tokens:
            return False
        overlap = len(target_tokens & hay_tokens) / len(target_tokens)
        return overlap >= 0.5
    # exact (normalised substring)
    return target in hay


@checker("fact_recall")
def fact_recall(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    value = params.get("value")
    if value is None:
        return False, "no expected value in params"
    match = params.get("match", "exact")
    if _RECALL_REFUSAL_RE.search(answer or ""):
        return False, "value not confirmed: answer refuses / disclaims memory"
    if _value_present(answer, value, match):
        return True, f"recalled value present ({match})"
    return False, f"expected value {value!r} not found in answer ({match} match)"


@checker("fact_update")
def fact_update(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    """Latest value must be present and (for exact match) stale values absent."""
    value = params.get("value")
    if value is None:
        return False, "no current value in params"
    match = params.get("match", "exact")
    if _RECALL_REFUSAL_RE.search(answer or ""):
        return False, "value not confirmed: answer refuses / disclaims memory"
    if not _value_present(answer, value, match):
        return False, f"current value {value!r} not found in answer"
    stale = params.get("stale_values") or []
    if match == "exact":
        for old in stale:
            if _value_present(answer, old, "exact"):
                return False, f"stale value {old!r} still present"
    return True, "current value present, no stale values"
