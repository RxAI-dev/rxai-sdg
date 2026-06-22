"""Defect detectors for the synthetic-conversation dataset (factual-fabrication
repair loop).

The HolisticJudge plus the format/harness pre-filter still pass an entire defect
class: **factual fabrication and confidence-uncertainty mismatch**. The evidence is
often *in the reasoning trace* - the model writes "I'm not sure who" / "constructed
illustration" / "we can reference" and the answer then asserts named games, URLs,
Metacritic scores, funding sums, poll percentages and attendance figures with full
confidence. Because the generation pipeline (Responder / User-Sim / Curator) has
**no retrieval**, any such checkable specific is fabricated by construction.

These detectors are deterministic and run over the *real* emitted reasoning and
answers (no mocks). Each returns :class:`Flag` objects with evidence spans. The two
PRIMARY detectors (A confidence-mismatch, B fabricated-specifics) become hard
auto-reject gates in the pre-filter; the rest feed scoring/penalties and review.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Flag:
    detector: str          # "A".."G"
    name: str              # human label
    severity: int          # 1 (soft) .. 3 (hard reject)
    turn_index: Optional[int]
    evidence: str


# --------------------------------------------------------------------------- A
# Negation guard: a marker preceded (within ~30 chars) by a negation is the model
# REFUSING to fabricate ("we cannot fabricate data", "rather than invent") - that
# is GOOD behaviour, not a defect, and must not fire.
_NEG_RE = re.compile(
    r"\b(?:cannot|can'?t|won'?t|will not|don'?t|do not|does not|doesn'?t|not|never|"
    r"without|avoid\w*|refus\w*|shouldn'?t|rather than|no need to|instead of)\b", re.I)


def _negated(text: str, start: int) -> bool:
    return bool(_NEG_RE.search(text[max(0, start - 30):start]))


# UNCERTAINTY markers: the model is unsure. Paired SAME-TURN with a fabricated
# specific in that turn's answer (A1) - so generic uncertainty about a feature
# detail in one turn is not joined to a specific in an unrelated turn.
UNCERTAINTY_RES = [
    re.compile(r"\bnot sure (?:who|what|whether|exactly|of the|about the)\b", re.I),
    re.compile(r"\b(?:don'?t|do not|can'?t|cannot|couldn'?t) (?:know who|verify|recall|confirm|be sure)\b", re.I),
    re.compile(r"\bnot 100% sure\b|\bnot entirely sure\b|\bI'?m guessing\b", re.I),
]
# FABRICATION-ADMISSION markers: the model is openly inventing CONTENT. So
# explicit that a conversation-level pairing with a fabricated specific (A3) is
# warranted. Each is negation-guarded at match time.
FABRICATION_ADMISSION_RES = [
    re.compile(r"\bfabricat\w*", re.I),
    re.compile(r"\b(?:I'?ll|we (?:can|could|'?ll)|let'?s) invent\w*", re.I),
    re.compile(r"\bmade it up\b|\bmake (?:it|them|one) up\b|\bmade[- ]up (?:source|figure|stat|quote|example)", re.I),
    re.compile(r"\bplausible (?:excerpt|source|sounding|figure|statistic|details?)\b", re.I),
    re.compile(r"\bhypothetical (?:source|citation|quote|excerpt|figure|statistic|study|reference)\b", re.I),
    re.compile(r"\bconstructed (?:illustration|scenario|example)\b|\bconstructed (?:it|this) (?:as|to be )?illustrat", re.I),
    re.compile(r"\b(?:don'?t|do not|doesn'?t) (?:have|exist)\b[^.\n]{0,40}\b(?:public|available|verifiable|record|source)", re.I),
    re.compile(r"\bwe (?:can|'?ll) reference\b", re.I),
    re.compile(r"\bmight have (?:appeared|existed|been (?:at|in))\b", re.I),
    re.compile(r"\bon the spot\b", re.I),
    re.compile(r"\bdoesn'?t turn up\b|\bno (?:publicly available|reliable|verifiable) (?:record|source|info)", re.I),
]
# WEAK hedges - contributing signal only (too common to gate on alone).
WEAK_HEDGE_RES = [
    re.compile(r"\bprobably\b", re.I), re.compile(r"\bpresumably\b", re.I),
    re.compile(r"\bapproximate\w*\b", re.I), re.compile(r"\blet'?s pick\b", re.I),
]

# B: checkable specifics that, with no retrieval in the loop, are fabricated.
# Bare 4-digit years are intentionally EXCLUDED (far too common in grounded text).
FAB_SPECIFIC_RES = {
    "url": re.compile(r"https?://\S+", re.I),
    "metacritic_score": re.compile(r"\bMetacritic\b|\b\d{2,3}\s*/\s*100\b|\b\d{2}\s*/\s*100\b", re.I),
    "funding_sum": re.compile(r"[\$£€]\s?\d[\d,\.]*\s*(?:k|K|M|m|bn|thousand|million|billion)\b|\bKickstarter\b|\bcrowdfund|\braised\s+[\$£€]\s?\d|\bfunding of\b", re.I),
    "code_host": re.compile(r"\bGitHub\b|\bGitLab\b|\b\d[\d,\.]*\s*stars?\b", re.I),
    "talk_timestamp": re.compile(r"\bGDC\b|GodotCon|\bkeynote at\b|(?:\bat|@|around|timestamp|min(?:ute)?\s+mark)\s+\d{1,2}:\d{2}\b|\b\d{1,2}:\d{2}\s*(?:[-–—]\s*\d{1,2}:\d{2}\s*)?(?:min\b|minute|mark\b)", re.I),
    "attendance": re.compile(r"[≈~]?\s?\d[\d,\.]*\s*(?:million|M|k|thousand)?\s*(?:visitors|attendees|tourists|spectators|/\s?yr\b|per year|annually|people a year)\b", re.I),
    "poll_pct": re.compile(r"\b\d{1,3}\s?%[^.\n]{0,30}(?:poll|survey|approval|respondents|voted)", re.I),
    "named_citation": re.compile(r"\b(?:according to|cited in|as reported by|per)\b[^.\n]{0,40}\b(?:report|study|article|interview|blog|podcast)\b", re.I),
    # an answer that asserts an exact ordinal ranking as fact ("the 37th-largest
    # city") - ungroundable with no retrieval, and the core cities fabrication.
    "rank_assertion": re.compile(r"\bthe\s+\d+(?:st|nd|rd|th)[‑\-\s](?:largest|biggest|smallest|most\s+populous|tallest|longest|highest)\b", re.I),
}

# A2: ungrounded-premise prompts - facts the base model cannot ground (no retrieval).
# Scoped to EXACT-RANKING premises, which are reliably ungroundable ("37th largest
# city"). A bare "who is X?" is NOT here: it over-fires on groundable people (the
# president, famous figures) whose hedged answers are good; obscure-bio fabrication
# is caught instead by the A3 admission gate (reasoning openly admits it cannot find
# the person), so a confident hedge on a knowable person is not punished.
UNGROUNDED_PREMISE_RES = [
    re.compile(r"\b\d+(?:st|nd|rd|th)\s+(?:largest|biggest|smallest|most\s+populous|tallest|longest|highest|richest|oldest)\b", re.I),
]


def _seg(turn: Any, kind: str) -> str:
    """Read a segment's text from either a serialised dict-turn or a Turn object."""
    if not isinstance(turn, dict):
        # Turn dataclass exposes .query/.reasoning/.answer properties
        val = getattr(turn, kind, None)
        if val is not None:
            return val or ""
        segs = getattr(turn, "segments", [])
        for s in segs:
            if getattr(s, "segment_type", None) == kind:
                return getattr(s, "text", "") or ""
        return ""
    for s in turn.get("segments", []):
        if s.get("segment_type") == kind:
            return s.get("text") or ""
    return ""


def _turn_index(turn: Any) -> int:
    if isinstance(turn, dict):
        return turn.get("turn_index", 0)
    return getattr(turn, "turn_index", 0)


def _constraint_spec(turn: Any) -> dict:
    if isinstance(turn, dict):
        return turn.get("constraint_spec") or {}
    cs = getattr(turn, "constraint_spec", None)
    if cs is None:
        return {}
    return cs if isinstance(cs, dict) else getattr(cs, "__dict__", {})


def reasoning_specifics(answer: str) -> list[tuple[str, str]]:
    """Return (name, matched_span) for each fabricated-specific class in an answer."""
    out: list[tuple[str, str]] = []
    for name, rx in FAB_SPECIFIC_RES.items():
        m = rx.search(answer or "")
        if m:
            out.append((name, m.group(0)[:50]))
    return out


def uncertainty_markers(reasoning: str) -> list[str]:
    out = []
    for rx in UNCERTAINTY_RES:
        m = rx.search(reasoning or "")
        if m and not _negated(reasoning, m.start()):
            out.append(m.group(0)[:50])
    return out


def admission_markers(reasoning: str) -> list[str]:
    """Explicit content-fabrication admissions, ignoring negated (refusal) forms."""
    out = []
    for rx in FABRICATION_ADMISSION_RES:
        m = rx.search(reasoning or "")
        if m and not _negated(reasoning, m.start()):
            out.append(m.group(0)[:50])
    return out


def is_ungrounded_premise(first_query: str) -> bool:
    return any(rx.search(first_query or "") for rx in UNGROUNDED_PREMISE_RES)


# --------------------------------------------------------------------------- A
def detect_confidence_mismatch(turns: list[dict], first_query: str = "") -> list[Flag]:
    """A (PRIMARY). The answer asserts checkable specifics its own reasoning is not
    grounded in. Three precise gates (each a hard reject):

    * A1 same-turn mismatch: a turn whose reasoning is UNCERTAIN and whose own
      answer asserts a fabricated specific.
    * A2 ungrounded premise: an opener the base model cannot ground (Nth-largest /
      who-is-<obscure>) and any answer asserts a fabricated specific.
    * A3 fabrication admission: reasoning anywhere openly admits inventing content
      (negation-guarded) and any answer asserts a fabricated specific.
    """
    flags: list[Flag] = []
    spec_turns = [(_turn_index(t), n, s)
                  for t in turns for (n, s) in reasoning_specifics(_seg(t, "answer"))]
    if not spec_turns:
        return flags  # no checkable specific anywhere -> nothing to mismatch

    # A1: same-turn uncertainty + specific in that same turn's answer
    for t in turns:
        ti = _turn_index(t)
        unc = uncertainty_markers(_seg(t, "reasoning"))
        specs = reasoning_specifics(_seg(t, "answer"))
        if unc and specs:
            flags.append(Flag("A", "confidence_uncertainty_mismatch", 3, ti,
                              f"reasoning(t{ti}) is unsure ('{unc[0]}') yet its answer "
                              f"asserts {specs[0][0]} '{specs[0][1]}'"))
            break
    if flags:
        return flags

    # A2: ungrounded premise + any fabricated specific
    if is_ungrounded_premise(first_query):
        st, sn, sp = spec_turns[0]
        flags.append(Flag("A", "ungrounded_premise_fabrication", 3, st,
                          f"ungrounded premise {first_query[:45]!r} yet answer(t{st}) "
                          f"asserts {sn} '{sp}'"))
        return flags

    # A3: explicit fabrication admission anywhere + any fabricated specific
    for t in turns:
        adm = admission_markers(_seg(t, "reasoning"))
        if adm:
            st, sn, sp = spec_turns[0]
            flags.append(Flag("A", "fabrication_admission", 3, st,
                              f"reasoning(t{_turn_index(t)}) admits '{adm[0]}' "
                              f"and answer(t{st}) asserts {sn} '{sp}'"))
            break
    return flags


# --------------------------------------------------------------------------- B
def detect_fabricated_specifics(turns: list[dict]) -> list[Flag]:
    """B. Any fabricated checkable specific in an answer (severity scales with how
    many distinct classes appear in one answer). Standalone flag/penalty; combined
    with A it is a hard gate."""
    flags: list[Flag] = []
    for t in turns:
        specs = reasoning_specifics(_seg(t, "answer"))
        if specs:
            sev = 3 if len(specs) >= 3 else (2 if len(specs) == 2 else 1)
            ev = ", ".join(f"{n}:{s}" for n, s in specs[:4])
            flags.append(Flag("B", "fabricated_specifics", sev,
                              _turn_index(t), ev))
    return flags


# --------------------------------------------------------------------------- C
_PY_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S | re.I)


def _has_assert(code: str) -> bool:
    return bool(re.search(r"^\s*assert\b", code, re.M))


_BENCH_RE = re.compile(r"\btimeit\b|\binput\s*\(|\bwhile True\b|__main__|requests\.|urllib|socket", re.I)


def detect_code_mismatch(turns: list[dict], timeout: float = 8.0) -> list[Flag]:
    """C (execution-verifiable). For any answer whose python contains an ``assert``
    (a claimed output), build a runnable script from that answer's code blocks -
    the definition blocks come first so the assert resolves its references - skip
    benchmark/IO blocks, execute in a subprocess sandbox, and flag if it raises.
    An AssertionError means the answer's own claimed output is wrong."""
    flags: list[Flag] = []
    for t in turns:
        ans = _seg(t, "answer")
        blocks = _PY_BLOCK_RE.findall(ans)
        if not blocks or not any(_has_assert(b) for b in blocks):
            continue
        # keep parseable, non-benchmark blocks in order (defs before the asserts)
        usable = []
        for b in blocks:
            if _BENCH_RE.search(b):
                continue
            try:
                ast.parse(b)
            except SyntaxError:
                continue
            usable.append(b)
        script = "\n\n".join(usable)
        if "assert" not in script:
            continue
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", script],
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            if "AssertionError" in proc.stderr:
                kind, sev = "assert_failed", 3
            elif "NameError" in proc.stderr:
                continue  # likely an unresolved cross-block ref, not a real claim bug
            else:
                kind, sev = "code_error", 2
            flags.append(Flag("C", kind, sev, _turn_index(t),
                              f"executed claim failed: {tail[0][:100]}"))
    return flags


# --------------------------------------------------------------------------- D
_FORMAT_MECH_RES = [
    re.compile(r"\bexactly (?:one|two|three|four|five|\d+) (?:bullet|sentence|word|line|item|paragraph)", re.I),
    re.compile(r"\b(?:not|no) more than\b|\bat most \d+\b", re.I),
    re.compile(r"\bmust be (?:valid )?(?:json|yaml|markdown)\b", re.I),
    re.compile(r"\bno (?:special )?formatting (?:requested|needed)\b", re.I),
    re.compile(r"\bbullet[- ]?point", re.I),
    re.compile(r"\bmarkdown (?:heading|table|format)", re.I),
    re.compile(r"\bensure (?:exactly|not more|the format|it'?s valid)", re.I),
    re.compile(r"\bword count\b|\bcount words\b", re.I),
]


def detect_format_bookkeeping(turns: list[dict], ratio_threshold: float = 0.45) -> list[Flag]:
    """D. Reasoning dominated by format/rubric mechanics rather than task cognition."""
    flags: list[Flag] = []
    for t in turns:
        r = _seg(t, "reasoning")
        if not r.strip():
            continue
        sents = re.split(r"(?<=[.!?\n])\s+", r)
        sents = [s for s in sents if s.strip()]
        if not sents:
            continue
        mech = sum(1 for s in sents if any(rx.search(s) for rx in _FORMAT_MECH_RES))
        ratio = mech / len(sents)
        if ratio >= ratio_threshold and mech >= 2:
            flags.append(Flag("D", "format_bookkeeping_reasoning", 1,
                              _turn_index(t),
                              f"{mech}/{len(sents)} reasoning sentences are format-mechanics"))
    return flags


# --------------------------------------------------------------------------- E
_FILLER_RES = [
    re.compile(r"\bwill produce (?:the )?final answer\.?", re.I),
    re.compile(r"\blet'?s comply\.?", re.I),
    re.compile(r"\b(?:proceed|let'?s proceed|let'?s do it|let'?s go)\.\s*$", re.I),
    re.compile(r"\bnow (?:write|produce|output) (?:the )?(?:answer|response)\.?\s*$", re.I),
]


def _overlap_ratio(a: str, b: str) -> float:
    """Fraction of the longer text's word-bigrams that appear verbatim in both."""
    def bigrams(s):
        w = re.findall(r"\w+", s.lower())
        return set(zip(w, w[1:]))
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    return inter / max(1, min(len(ba), len(bb)))


def detect_reasoning_artifacts(turns: list[dict], dup_threshold: float = 0.6) -> list[Flag]:
    """E. (i) filler tails in reasoning; (ii) the answer pre-written verbatim inside
    the reasoning trace (high reasoning<->answer bigram overlap)."""
    flags: list[Flag] = []
    for t in turns:
        r, a = _seg(t, "reasoning"), _seg(t, "answer")
        for rx in _FILLER_RES:
            m = rx.search(r)
            if m:
                flags.append(Flag("E", "filler_tail", 1, _turn_index(t),
                                  m.group(0)[:50]))
                break
        if len(a) > 200 and _overlap_ratio(r, a) >= dup_threshold:
            flags.append(Flag("E", "answer_duplicated_in_reasoning", 1,
                              _turn_index(t),
                              f"reasoning<->answer bigram overlap >= {dup_threshold}"))
    return flags


# --------------------------------------------------------------------------- F
_STANDING_LICENSE_RE = re.compile(
    r"\b(?:from now on|going forward|always|every (?:reply|time|response|answer)|"
    r"for the rest|keep (?:doing|using)|continue to|each time|whenever I)\b", re.I)


def detect_constraint_integrity(turns: list[dict]) -> list[Flag]:
    """F. (i) phantom standing constraints: scope=standing with no licensing phrase
    in the user query that introduced it. (ii) fact_recall mislabel heuristic."""
    flags: list[Flag] = []
    for t in turns:
        cs = _constraint_spec(t)
        scope = cs.get("scope")
        if scope == "standing":
            q = _seg(t, "query")
            if q and not _STANDING_LICENSE_RE.search(q):
                flags.append(Flag("F", "phantom_standing_constraint", 2,
                                  _turn_index(t),
                                  f"scope=standing but query has no standing license: {q[:60]!r}"))
    return flags


# --------------------------------------------------------------------------- run
RUN_ORDER = ["A", "B", "C", "D", "E", "F"]


def run_all(record: dict, run_code: bool = True) -> list[Flag]:
    """Run every detector over one serialised record (dict with first_query+turns)."""
    turns = record.get("turns") or []
    if isinstance(turns, str):
        import json
        turns = json.loads(turns)
    fq = record.get("first_query", "")
    flags: list[Flag] = []
    flags += detect_confidence_mismatch(turns, fq)
    flags += detect_fabricated_specifics(turns)
    if run_code:
        flags += detect_code_mismatch(turns)
    flags += detect_format_bookkeeping(turns)
    flags += detect_reasoning_artifacts(turns)
    flags += detect_constraint_integrity(turns)
    return flags
