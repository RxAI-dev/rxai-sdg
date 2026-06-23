"""English-specific constraint checkers (spec §9).

These checkers depend on English sentence segmentation, the Latin alphabet, or
the English pronoun set, so they register under ``lang="en"``. The
sentence-segmentation here is a deliberately small regex splitter (no ``nltk``
dependency); swap in a stronger segmenter per language when extending.
"""

from __future__ import annotations

import re
from typing import Any

from .base import checker, register_language_stubs

# Constraint types that are English/Latin/language-specific and therefore need a
# native implementation per locale. Exposed so other locales can register stubs.
LANGUAGE_SPECIFIC_TYPES = [
    "first_letter",
    "max_words_per_sentence",
    "forbidden_token",
    "no_gendered_pronouns",
    "alphabetical_sentence_starts",
    "limerick_structure",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Structured spans that are NOT prose: a lexical sentence-start / word-count rule
# applies to the PROSE only, never to code, LaTeX/math or table syntax. Stripping
# them here means (a) a clean answer that keeps its math/code intact still PASSES the
# constraint, and (b) the model is not pushed to corrupt a LaTeX matrix by prefixing
# every line with the target letter (the "S\;G=\begin{pmatrix} S\;1&1..." defect).
_STRUCTURED_SPANS = [
    re.compile(r"```.*?```", re.S),                 # fenced code blocks
    re.compile(r"~~~.*?~~~", re.S),                  # fenced code blocks (tildes)
    re.compile(r"\\\[.*?\\\]", re.S),               # LaTeX display math \[ ... \]
    re.compile(r"\$\$.*?\$\$", re.S),               # $$ ... $$
    re.compile(r"\\begin\{.*?\}.*?\\end\{.*?\}", re.S),  # LaTeX environments
    re.compile(r"\\\(.*?\\\)", re.S),               # inline math \( ... \)
    re.compile(r"`[^`\n]+`"),                        # inline code spans
    re.compile(r"(?m)^[ \t]*\|.*\|[ \t]*$"),        # markdown table rows
]


def _strip_structured(text: str) -> str:
    """Remove code / LaTeX-math / table spans, leaving prose for lexical checks."""
    out = text
    for rx in _STRUCTURED_SPANS:
        out = rx.sub(" ", out)
    return out


def split_sentences(text: str, prose_only: bool = False) -> list[str]:
    """Split ``text`` into sentences (simple English heuristic).

    With ``prose_only=True`` code/LaTeX/table spans are stripped first, so a
    sentence-start / word-count constraint is checked against prose only (and a
    clean answer keeps its math/code intact). Falls back to the full text if
    stripping leaves nothing (a pure-code answer), so behaviour is unchanged there.
    """
    src = text
    if prose_only:
        stripped = _strip_structured(text)
        if stripped.strip():
            src = stripped
    cleaned = " ".join(src.replace("\n", " ").split())
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def _first_alpha(s: str) -> str:
    for ch in s:
        if ch.isalpha():
            return ch
    return ""


# ---------------------------------------------------------------------------
# checkers
# ---------------------------------------------------------------------------

@checker("first_letter", lang="en")
def first_letter(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    letter = str(params.get("letter", "")).lower()
    if not letter:
        return False, "no target letter in params"
    sentences = split_sentences(answer, prose_only=True)
    if not sentences:
        return False, "no sentences found"
    offenders = [
        s for s in sentences if _first_alpha(s).lower() != letter
    ]
    if offenders:
        sample = offenders[0][:40]
        return False, (
            f"{len(offenders)}/{len(sentences)} sentences do not start with "
            f"{letter!r} (e.g. {sample!r})")
    return True, f"all {len(sentences)} sentences start with {letter!r}"


@checker("max_words_per_sentence", lang="en")
def max_words_per_sentence(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    max_words = params.get("max_words")
    if max_words is None:
        return False, "no max_words in params"
    sentences = split_sentences(answer, prose_only=True)
    if not sentences:
        return False, "no sentences found"
    for s in sentences:
        wc = len(re.findall(r"\b\w[\w'-]*\b", s))
        if wc > max_words:
            return False, f"sentence has {wc} words > max {max_words}: {s[:40]!r}"
    return True, f"all {len(sentences)} sentences within {max_words} words"


@checker("forbidden_token", lang="en")
def forbidden_token(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    token = str(params.get("token", "")).strip()
    if not token:
        return False, "no forbidden token in params"
    pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    if pattern.search(answer):
        return False, f"forbidden token {token!r} present"
    return True, f"forbidden token {token!r} absent"


_GENDERED_PRONOUNS = {"he", "him", "his", "she", "her", "hers", "himself", "herself"}


@checker("no_gendered_pronouns", lang="en")
def no_gendered_pronouns(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    tokens = re.findall(r"\b[\w']+\b", answer.lower())
    found = sorted({t for t in tokens if t in _GENDERED_PRONOUNS})
    if found:
        return False, f"gendered pronouns present: {found}"
    return True, "no gendered pronouns"


@checker("alphabetical_sentence_starts", lang="en")
def alphabetical_sentence_starts(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    """First letters of consecutive sentences must be non-decreasing (A, B, C ...).

    ``strict`` (default ``False``): if true, require a strict A, B, C, ...
    increasing sequence rather than merely non-decreasing.
    """
    strict = bool(params.get("strict", False))
    sentences = split_sentences(answer, prose_only=True)
    if len(sentences) < 2:
        return False, "need at least two sentences"
    letters = [_first_alpha(s).lower() for s in sentences]
    if any(not c for c in letters):
        return False, "a sentence has no leading letter"
    for prev, cur in zip(letters, letters[1:]):
        if strict and not cur > prev:
            return False, f"not strictly increasing at {prev!r}->{cur!r}"
        if not strict and cur < prev:
            return False, f"out of order at {prev!r}->{cur!r}"
    return True, f"sentence starts in alphabetical order: {''.join(letters)}"


@checker("limerick_structure", lang="en")
def limerick_structure(answer: str, params: dict, conversation: Any) -> tuple[bool, str]:
    """Cheap structural check for a limerick: exactly five non-empty lines."""
    lines = [ln for ln in answer.splitlines() if ln.strip()]
    if len(lines) != 5:
        return False, f"limerick needs 5 lines, found {len(lines)}"
    return True, "five-line structure OK"


# Make sure all English checkers are registered on import.
def _ensure_registered() -> None:  # pragma: no cover - import side effect guard
    pass
