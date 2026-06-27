"""Focused factuality gate (problem 2: fabrication of named specifics).

The holistic judge scores a whole conversation on a 1-10 ``factual_grounding``
axis and is BLIND to a single confident-but-wrong specific buried in an otherwise
good answer (validated: it gave ``factual_grounding=9`` to an answer stating Van
Gogh was played by "Thomas Murray" - it was Tony Curran). A *decomposed* check -
extract atomic checkable claims, verify each - catches these where holistic
scoring does not (validated on the same answer: the claim-level pass returns the
correct "Tony Curran" correction, and also caught a wrong air-date the holistic
pass and a human reviewer both missed), while leaving advice/opinion/estimate
content unflagged.

This is a SEPARATE LLM call from the rubric judge, deliberately narrow: it only
adjudicates verifiable named-entity / date / attribution / quantitative claims.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .clients import LLMClient
from .schemas import Turn

_FACTCHECK_SYSTEM = (
    "You are a meticulous fact-checker. You are given assistant ANSWERS from a "
    "conversation. Extract every CHECKABLE factual claim about named people, "
    "places, works, organisations, dates, attributions (who said/wrote what), and "
    "specific quantities. IGNORE opinions, advice, hypotheticals, and anything the "
    "answer explicitly labels an estimate, example, or sample. Judge each claim "
    "TRUE / FALSE / UNVERIFIABLE using only your own knowledge. Mark a claim FALSE "
    "ONLY when you are confident it is wrong (e.g. a plausible but incorrect actor "
    "name, a wrong date, a misattributed quote). When unsure, use UNVERIFIABLE - "
    "never guess FALSE. Output ONLY a JSON object, no prose:\n"
    '{"false_claims":[{"claim":"<verbatim claim>","correction":"<the truth>"}],'
    '"any_false":true|false}'
)


def _answers_blob(turns: list[Turn], per_turn_cap: int = 1500) -> str:
    parts: list[str] = []
    for t in turns:
        for seg in (t.segments or []):
            if getattr(seg, "segment_type", None) == "answer":
                txt = (seg.text or "").strip()
                if txt:
                    parts.append(txt[:per_turn_cap])
    return "\n---\n".join(parts)


def _last_json_object(text: Optional[str]) -> Optional[dict]:
    """Return the LAST top-level ``{...}`` in ``text`` that parses as a JSON object.

    The verification model may emit reasoning (which can itself contain braces)
    before the JSON, so a greedy regex is wrong. This scans for balanced top-level
    objects, tracking string context so braces inside strings do not miscount.
    """
    if not text:
        return None
    results: list[dict] = []
    depth = 0
    start: Optional[int] = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
    return results[-1] if results else None


@dataclass
class FactCheckResult:
    passed: bool
    false_claims: list[dict] = field(default_factory=list)
    available: bool = True  # False => the call failed / unparseable (do NOT gate)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "false_claims": self.false_claims[:10],
            "available": self.available,
        }


def _tok(s: str) -> set:
    return set(re.findall(r"\w+", (s or "").lower()))


@dataclass
class AnswerRepairer:
    """Repair-then-recheck (problem 2, yield lever). The factuality check already
    returns each false claim WITH its correction; rather than just reject, feed
    those corrections back and apply them to the offending answer, then let the
    caller re-check. Validated: a salvageable error (a wrong sum) is fixed and the
    re-check passes; an unsalvageable one (an answer built on a false premise) is
    fixed superficially but the re-check still rejects it - so yield rises on
    fixable conversations while cleanliness is guaranteed by the re-check."""

    client: LLMClient
    max_tokens: int = 3000
    enabled: bool = True
    #: a claim is attributed to the answer with the highest token overlap, but only
    #: if the overlap clears this fraction of the claim's tokens (else it is skipped
    #: - we never guess which answer to edit).
    min_overlap: float = 0.5

    def repair(self, turns: list[Turn], false_claims: list[dict]) -> bool:
        """Apply corrections to the matching answers IN PLACE. Returns True if any
        answer was changed."""
        if not false_claims:
            return False
        answer_segs = [seg for t in turns for seg in (t.segments or [])
                       if seg.segment_type == "answer" and (seg.text or "").strip()]
        if not answer_segs:
            return False
        by_seg: dict[int, list] = {}
        for c in false_claims:
            seg = self._best_match(c.get("claim", ""), answer_segs)
            if seg is not None:
                by_seg.setdefault(id(seg), []).append(c)
        seg_by_id = {id(s): s for s in answer_segs}
        changed = False
        for sid, claims in by_seg.items():
            seg = seg_by_id[sid]
            fixed = self._fix(seg.text, claims)
            if fixed and fixed.strip() and fixed != seg.text:
                seg.text = fixed
                changed = True
        return changed

    def _best_match(self, claim: str, segs: list):
        ct = _tok(claim)
        if not ct:
            return None
        best, best_score = None, 0.0
        for seg in segs:
            st = _tok(seg.text)
            score = len(ct & st) / len(ct)
            if score > best_score:
                best, best_score = seg, score
        return best if best_score >= self.min_overlap else None

    def _fix(self, answer: str, claims: list) -> Optional[str]:
        errs = "\n".join(
            f"- WRONG: {c.get('claim','')}\n  FIX: {c.get('correction','')}"
            for c in claims)
        system = (
            "You are given an assistant ANSWER and a list of specific factual/"
            "arithmetic errors, each with its correction. Return the FULL answer "
            "with ONLY those errors fixed, changing the wrong values/words and any "
            "directly dependent figures, and keeping everything else and all "
            "formatting identical. Do not add commentary. Output ONLY the corrected "
            "answer.")
        try:
            resp = self.client.generate(
                f"ERRORS TO FIX:\n{errs}\n\nANSWER:\n{answer[:4000]}",
                system_prompt=system, temperature=0.0, max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001
            return None
        out = (resp.text or "").strip()
        # a localized repair stays close to the original size; reject a runaway
        # rewrite (keeps the original, which the gate then rejects - no yield loss
        # vs not repairing, and no risk of a mangled answer entering the dataset).
        if not out or not (0.5 * len(answer) <= len(out) <= 1.5 * len(answer) + 200):
            return None
        return out


@dataclass
class FactChecker:
    """One decomposed claim-verification call per conversation."""

    client: LLMClient
    max_tokens: int = 12000
    enabled: bool = True

    def check(self, turns: list[Turn]) -> FactCheckResult:
        blob = _answers_blob(turns)
        if not blob.strip():
            return FactCheckResult(passed=True, available=True)
        try:
            resp = self.client.generate(
                "ANSWERS:\n" + blob, system_prompt=_FACTCHECK_SYSTEM,
                temperature=0.0, max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - a failed call must NOT gate (no false reject)
            return FactCheckResult(passed=True, available=False)
        data = _last_json_object(resp.text)
        if data is None:
            # unparseable -> treat as "no signal", do not gate (mirrors the holistic
            # judge's "no rubric -> do not gate" rule; avoids silent over-rejection).
            return FactCheckResult(passed=True, available=False)
        raw = data.get("false_claims") or []
        false_claims = [
            {"claim": str(c.get("claim", ""))[:300],
             "correction": str(c.get("correction", ""))[:300]}
            for c in raw if isinstance(c, dict) and c.get("claim")
        ]
        any_false = bool(data.get("any_false")) or bool(false_claims)
        return FactCheckResult(passed=not any_false, false_claims=false_claims,
                               available=True)
