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


# ------------------------------------------------------------------------- A-cite
# Fabricated SCHOLARLY CITATIONS with specifics. With no retrieval in the loop, an
# answer that attributes a concrete figure to a named study/article/report, or
# cites a dated publication venue, is fabricated by construction - and the LLM judge
# is empirically blind to it (it scores such answers factual_grounding 10 because
# the prose is fluent and the topic is real). These three high-precision patterns
# each require a publication noun CO-OCCURRING with a checkable specific (a number,
# a percentage, or a year+venue), so a vague "studies suggest ..." is NOT flagged.
_CITATION_FIGURE_RE = re.compile(
    r"\b(?:study|studies|paper|papers|article|report|survey|analysis|meta-?analysis|"
    r"researchers?|scientists?|authors?|historians?|economists?)\b"
    r"[^.\n]{0,70}\b(?:estimat\w+|found|reported|concluded|showed|calculated|"
    r"determined|measured|put (?:it|the (?:number|figure|total|count|rate)) at)\b"
    r"[^.\n]{0,30}[\d≈~]", re.I)
_CITATION_VENUE_YEAR_RE = re.compile(
    r"\b(?:a|an|the)\s+\d{4}\s+(?:article|study|paper|report|survey|analysis|review|"
    r"meta-?analysis|editorial|essay)\b[^.\n]{0,15}\b(?:in|by|from|published)\b", re.I)
_CITATION_ACCORDING_RE = re.compile(
    r"\baccording to\b[^.\n]{0,55}\b(?:study|studies|report|survey|poll|census|article|"
    r"paper|index|database|dataset)\b[^.\n]{0,40}[\d%]", re.I)
_CITATION_RES = {
    "cited_figure": _CITATION_FIGURE_RE,
    "dated_venue": _CITATION_VENUE_YEAR_RE,
    "according_to_stat": _CITATION_ACCORDING_RE,
}


def fabricated_citations(answer: str) -> list[tuple[str, str]]:
    """Return (pattern, matched_span) for each fabricated-citation class in an answer."""
    out: list[tuple[str, str]] = []
    for name, rx in _CITATION_RES.items():
        m = rx.search(answer or "")
        if m:
            out.append((name, m.group(0)[:70]))
    return out


def detect_fabricated_citation(turns: list[dict]) -> list[Flag]:
    """A-cite (PRIMARY, hard reject). An answer attributes a concrete figure to a
    named study/report, or cites a dated publication venue. With no retrieval such a
    citation is invented; severity 3 because a confident fake citation is poison the
    LLM judge cannot see."""
    flags: list[Flag] = []
    for t in turns:
        hits = fabricated_citations(_seg(t, "answer"))
        if hits:
            ev = "; ".join(f"{n}:{s}" for n, s in hits[:3])
            flags.append(Flag("A", "fabricated_citation", 3, _turn_index(t), ev))
    return flags


# ------------------------------------------------------------------- A-disclaim
# The reasoning-vs-answer contradiction that the reasoning-visibility fix was meant
# to expose: the REASONING explicitly admits it cannot ground a specific (no
# documented case, can't fabricate one), yet the ANSWER asserts a concrete empirical
# FINDING with a figure ("field studies ... found volumes of 30-40 ft³"). Visible
# only by cross-checking reasoning against answer; the LLM judge passes it. Tightened
# to a CLAIMED finding-with-number (not a recommendation to run a study, not a real
# documented event) so it is false-positive-free.
_DISCLAIM_RE = re.compile(
    r"can'?t fabricate(?: a| any)?(?: specific| real| actual)?(?: documented| published)?\s*"
    r"(?:case|study|studies|data|source|measurement|figure|example)"
    r"|cannot fabricate(?: a| any)?(?: specific| documented)?\s*(?:case|study|data|source|measurement)"
    r"|no (?:known |real |actual |specific )?(?:documented|published) (?:case|study|studies|data|record|measurement)"
    r"|(?:can'?t|cannot|couldn'?t) (?:find|locate|document) (?:a |any )?(?:specific|documented|real|published) "
    r"(?:case|study|data|source|measurement)"
    r"|there (?:is|are) no (?:known |documented |published |real )(?:study|studies|data|case|measurement)",
    re.IGNORECASE)
_FINDING_RE = re.compile(
    r"\b(?:field stud(?:y|ies)|stud(?:y|ies)|researchers?|surveys?|measurements?)\b"
    r"[^.\n]{0,90}\b(?:found|measured|recorded|documented|reported|showed|observed|estimated)\b"
    r"[^.\n]{0,45}[\d]", re.IGNORECASE)
_FINDING_REC_RE = re.compile(
    r"\b(?:run|conduct|perform|do|carry out|consider|use|via|through|validat\w+|recommend\w+|future)\s+"
    r"(?:[a-z]+\s+){0,3}(?:field stud|stud|survey|measurement)", re.IGNORECASE)


def detect_disclaimer_then_finding(turns: list[dict]) -> list[Flag]:
    """A-disclaim (PRIMARY, hard reject). Any turn whose REASONING admits it cannot
    document/fabricate a specific, while its ANSWER asserts a quantified empirical
    finding (a study that found/measured a number) - a reasoning<->answer
    confidence-uncertainty contradiction the LLM judge does not catch."""
    flags: list[Flag] = []
    for t in turns:
        if not _DISCLAIM_RE.search(_seg(t, "reasoning")):
            continue
        answer = _seg(t, "answer")
        m = _FINDING_RE.search(answer)
        if not m:
            continue
        if _FINDING_REC_RE.search(answer[max(0, m.start() - 25):m.end()]):
            continue  # it's a recommendation to run a study, not a claimed finding
        flags.append(Flag("A", "disclaimer_then_finding", 3, _turn_index(t),
                          f"reasoning can't-document but answer claims: {m.group(0)[:60]!r}"))
    return flags


# ----------------------------------------------------------------------- G-corrupt
# A lexical sentence-start / first-letter constraint applied DESTRUCTIVELY to
# structured content: the model prefixes the target letter onto every line of a
# LaTeX matrix or code block ("S\;G=\begin{pmatrix} S\;1&1&0&0 ..."), corrupting the
# math/code. The signature is the SAME single letter glued to a math token (\; \, \[
# \begin) many times - legitimate dense LaTeX uses these too, but never one specific
# letter glued repeatedly. Also catches a fenced code block whose lines are all
# prefixed with the same single letter + space.
_CORRUPT_GLUE_RE = re.compile(r"\b([A-Za-z])\\[;,](?=[A-Za-z0-9\\])")
_CORRUPT_DELIM_RE = re.compile(r"\b([A-Za-z])\s{0,2}\\(?:\[|begin\b)")
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def _constraint_corruption(answer: str, glue_threshold: int = 5) -> Optional[str]:
    from collections import Counter
    c: Counter = Counter()
    for m in _CORRUPT_GLUE_RE.finditer(answer or ""):
        c[m.group(1)] += 1
    for m in _CORRUPT_DELIM_RE.finditer(answer or ""):
        c[m.group(1)] += 1
    if c:
        letter, n = c.most_common(1)[0]
        if n >= glue_threshold:
            return f"letter {letter!r} glued into LaTeX/math {n} times"
    for block in _FENCE_RE.findall(answer or ""):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) >= 4:
            prefixes = {ln.lstrip()[:2] for ln in lines}
            if len(prefixes) == 1:
                p = next(iter(prefixes))
                if len(p) == 2 and p[0].isalpha() and p[1] == " ":
                    return f"every code line prefixed with {p[0]!r}"
    return None


def detect_constraint_corruption(turns: list[dict]) -> list[Flag]:
    """G (hard reject). A lexical constraint mechanically corrupting a LaTeX/code
    block (the target letter spliced into every formula/code line). The output is
    garbled training data even though the constraint verifier may report 'satisfied'."""
    flags: list[Flag] = []
    for t in turns:
        ev = _constraint_corruption(_seg(t, "answer"))
        if ev:
            flags.append(Flag("G", "constraint_corruption", 3, _turn_index(t), ev))
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


def detect_reasoning_artifacts(turns: list[dict], dup_threshold: float = 0.82) -> list[Flag]:
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
    flags += detect_fabricated_citation(turns)
    flags += detect_constraint_corruption(turns)
    flags += detect_fabricated_specifics(turns)
    if run_code:
        flags += detect_code_mismatch(turns)
    flags += detect_format_bookkeeping(turns)
    flags += detect_reasoning_artifacts(turns)
    flags += detect_constraint_integrity(turns)
    return flags
