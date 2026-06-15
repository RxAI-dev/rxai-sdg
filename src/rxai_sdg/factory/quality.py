"""General quality filters that always apply, independent of constraints (spec §3).

These cheap checks (refusal, length, repetition) run on every answer regardless
of whether the turn carries a machine-checkable ``constraint_spec``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REFUSAL_PATTERNS = [
    r"\bi can'?t (?:help|assist|do that|comply)\b",
    r"\bi'?m (?:sorry,? )?(?:but )?i (?:can'?t|cannot)\b",
    r"\bi am unable to\b",
    r"\bas an ai language model\b",
    r"\bi cannot (?:help|assist|provide)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass
class QualityConfig:
    min_chars: int = 8
    min_words: int = 2
    max_repeat_ratio: float = 0.5  # max fraction of repeated trigrams
    allow_refusals: bool = False


def _repeat_ratio(text: str) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 6:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    if not trigrams:
        return 0.0
    unique = len(set(trigrams))
    return 1.0 - unique / len(trigrams)


def check_quality(answer: str, config: QualityConfig | None = None) -> tuple[bool, str]:
    """Return ``(ok, detail)`` for the general-quality gate."""
    config = config or QualityConfig()
    text = (answer or "").strip()
    if len(text) < config.min_chars:
        return False, f"too short ({len(text)} chars)"
    if len(re.findall(r"\b\w+\b", text)) < config.min_words:
        return False, "too few words"
    if not config.allow_refusals and _REFUSAL_RE.search(text):
        return False, "looks like a refusal"
    rr = _repeat_ratio(text)
    if rr > config.max_repeat_ratio:
        return False, f"excessive repetition (ratio {rr:.2f})"
    return True, "quality OK"
