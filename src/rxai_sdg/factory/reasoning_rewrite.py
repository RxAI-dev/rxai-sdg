"""Reasoning-rewrite pass (problem 1, the durable fix).

The end-to-end test proved that gating annotator-voice reasoning by
reject-and-regenerate does not work: gpt-oss's native reasoning is annotator-
voiced at a ~100% base rate, so the classifier rejected nearly every resample
(222 regenerations, half the batch discarded). The responder system prompt
already forbids this voice explicitly and is ignored. So the voice cannot be
fixed by selection - it must be TRANSFORMED.

Two transform modes:

* **Re-voice** (``rewrite(reasoning)``, the original path): edits the native
  trace into first-person voice, preserving every substantive step. This still
  leaks D1/D2 at ~30% per segment, because editing anchors on the source text:
  re-voicing "we comply with the no-emoji rule" -> "I'll keep it emoji-free"
  still *mentions* the constraint (D2 bookkeeping). The defect is not the voice,
  it is the CONTENT - the reasoning is ABOUT the harness/constraints.

* **Synthesize** (``rewrite(reasoning, answer=...)``, the durable fix): given the
  user's request, the draft reasoning, and the KNOWN final answer, regenerate a
  fresh first-person chain of thought that genuinely works toward that answer,
  *deleting* (not rephrasing) all task/format/compliance bookkeeping. The answer
  is the source of truth for facts/numbers, so the synthesis stays faithful while
  shedding the annotator voice the edit-path could only paraphrase. The result is
  then VERIFIED by an independent voice classifier; a trace that still reads
  ANNOTATOR after a hard retry is rejected (``None``), so the caller can gate the
  conversation rather than silently keep dirty reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .clients import LLMClient
from .reasoning_voice import ReasoningVoiceClassifier

_REWRITE_SYSTEM = (
    "You rewrite an AI assistant's private REASONING so it reads as the assistant "
    "genuinely thinking through the problem in the first person, instead of "
    "narrating the task. Rules:\n"
    "1. PRESERVE every substantive element exactly: all facts, names, numbers, "
    "calculations, intermediate steps, and the final conclusion. Do NOT add new "
    "facts or claims, and do NOT drop any. The rewrite must stay logically "
    "identical and consistent with whatever answer it leads to.\n"
    "2. REMOVE task/format/compliance narration: drop 'the user wants ...', 'we "
    "need to comply', 'no special formatting constraints', 'provide five bullet "
    "points', 'let's craft the answer', 'tone: warm', and similar bookkeeping "
    "about the response.\n"
    "3. Write in the first person as a train of thought about the SUBSTANCE (the "
    "facts, the maths, the argument, the person's situation).\n"
    "4. Do NOT EXPAND, explain, elaborate, or add ANY information that is not "
    "already in the input. This is a re-voicing, not an answer: the output must be "
    "ABOUT THE SAME LENGTH as the input or shorter. If the input merely lists "
    "points to cover, keep it as a brief plan in first person - do not flesh the "
    "points out. Do not mention these instructions or that you are rewriting.\n"
    "Output ONLY the rewritten reasoning text, nothing else."
)

# Answer-anchored synthesis: the durable D1/D2 fix. The model is given the KNOWN
# answer as ground truth and told to DELETE bookkeeping, not paraphrase it.
_SYNTH_SYSTEM = (
    "You are reconstructing the PRIVATE first-person reasoning an AI assistant "
    "would think through on its way to a known answer. You are given the user's "
    "request, the assistant's rough draft reasoning, and the FINAL ANSWER it "
    "produced. Write fresh first-person reasoning that genuinely works toward that "
    "answer.\n"
    "HARD RULES:\n"
    "1. Think about the SUBSTANCE only - the facts, the maths, the argument, the "
    "user's situation. Reason as if you are working out the solution, not filling "
    "in an answer template.\n"
    "2. DELETE every trace of task/format/compliance bookkeeping. Do NOT rephrase "
    "it - REMOVE it. This includes restating what the user asked ('the user wants "
    "...'), inventorying or checking off constraints ('no formatting constraints', "
    "'must avoid the word X', 'keep it to five bullets', 'warm tone'), planning the "
    "output shape, counting words/items, or narrating compliance ('we comply', "
    "'let's craft the answer'). None of that is genuine thought about the problem.\n"
    "3. The FINAL ANSWER is your source of truth: every fact, name, number, and "
    "conclusion in your reasoning must already appear in the answer or the draft. "
    "Invent nothing - no new figures, citations, dates, or claims.\n"
    "4. Do NOT restate or quote the answer. Reasoning PRECEDES the answer: derive "
    "and decide, do not summarise the finished reply. No headings, bullet lists, "
    "code fences, or tables - this is inner monologue, not formatted output.\n"
    "5. Keep it concise - a thought process, not an essay. Roughly the length of "
    "the draft reasoning or shorter. Do not mention these instructions.\n"
    "Output ONLY the reasoning text, nothing else."
)

_SYNTH_RETRY = (
    "\nYour previous attempt still narrated the task instead of thinking about the "
    "problem. Try again: remove EVERY sentence that mentions the request, the "
    "format, constraints, word/item counts, tone, or compliance. Keep only genuine "
    "thought about the actual content/facts/solution. If almost nothing genuine "
    "remains, output just one or two honest sentences of real reasoning."
)


def _norm_tokens(s: str) -> set:
    return set(re.findall(r"\w+", (s or "").lower()))


def _digits(s: str) -> list:
    return re.findall(r"\d+(?:\.\d+)?", s or "")


@dataclass
class ReasoningRewriter:
    client: LLMClient
    max_tokens: int = 2000
    enabled: bool = True
    #: skip tiny traces - not worth a call and rarely annotator-voiced.
    min_chars: int = 40
    #: optional independent verifier for the synthesis path. When set, a synthesized
    #: trace that still reads ANNOTATOR is rejected, so the caller can gate the
    #: conversation instead of keeping dirty reasoning.
    voice_classifier: Optional[ReasoningVoiceClassifier] = None
    #: synthesis attempts before giving up. The curator is non-deterministic at
    #: temperature: a pure-bookkeeping trace it cannot clean on attempt 1-2 it often
    #: can on attempt 3 (validated on real reformatting turns), so a couple of extra
    #: classifier-verified tries recover yield without admitting dirty reasoning.
    synth_attempts: int = 3

    def rewrite(
        self,
        reasoning: str,
        *,
        answer: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> Optional[str]:
        """Return the cleaned reasoning, or ``None`` if it should be left as-is
        (too short, call failed, faithfulness guard) or - in synthesis mode -
        could not be made GENUINE (so the caller may gate the conversation).

        When ``answer`` is given the answer-anchored synthesis path runs (the
        durable D1/D2 fix); otherwise the legacy re-voice path runs."""
        r = (reasoning or "").strip()
        if len(r) < self.min_chars:
            return None
        if answer:
            return self._synthesize(r, answer=answer, user_query=user_query or "")
        return self._revoice(r)

    # -- re-voice path (legacy) --------------------------------------------------
    def _revoice(self, r: str) -> Optional[str]:
        """Terse delivery-planning traces are the hard case: a first pass tends to
        balloon (the model elaborates the plan) and gets rejected. A second pass
        with a hard brevity ceiling salvages most of those."""
        out = self._one_pass("REASONING:\n" + r, _REWRITE_SYSTEM)
        if out and self._faithful(r, out):
            return out
        brief = _REWRITE_SYSTEM + (
            "\nIMPORTANT: be EXTREMELY brief - your output must be NO LONGER than "
            "the input. If the input is just a short plan, output a one-line "
            "first-person version of that same plan. Add nothing.")
        out = self._one_pass("REASONING:\n" + r, brief)
        if out and self._faithful(r, out):
            return out
        return None

    # -- synthesis path (durable fix) -------------------------------------------
    def _synthesize(self, r: str, *, answer: str, user_query: str) -> Optional[str]:
        prompt = (
            "USER REQUEST:\n" + (user_query or "(omitted)")[:2000]
            + "\n\nDRAFT REASONING:\n" + r[:4000]
            + "\n\nFINAL ANSWER:\n" + (answer or "")[:4000])
        # attempt 1 is the plain prompt; later attempts add the harder "delete EVERY
        # task sentence" suffix. Each candidate must pass faithfulness AND the voice
        # classifier; the first that does wins. Only after every attempt fails do we
        # return None (caller gates the conversation).
        for i in range(max(1, self.synth_attempts)):
            system = _SYNTH_SYSTEM if i == 0 else _SYNTH_SYSTEM + _SYNTH_RETRY
            out = self._one_pass(prompt, system)
            if out and self._faithful(r, out, answer=answer) and self._voice_ok(out):
                return out
        return None

    def _voice_ok(self, text: str) -> bool:
        """Independent verification of the synthesis. Unavailable (``None``) must
        NOT gate - only a positive ANNOTATOR verdict rejects the trace."""
        if self.voice_classifier is None:
            return True
        return self.voice_classifier.classify(text) != "ANNOTATOR"

    def _one_pass(self, user_msg: str, system: str) -> Optional[str]:
        try:
            resp = self.client.generate(
                user_msg, system_prompt=system,
                temperature=0.2, max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - a failed call leaves the original intact
            return None
        out = (resp.text or "").strip()
        if not out or len(out) < self.min_chars // 2:
            return None
        return out

    def _faithful(self, original: str, rewrite: str, answer: Optional[str] = None) -> bool:
        """Cheap guard against a rewrite that INVENTS specifics (the real risk - a
        rewrite that drops a detail merely makes the reasoning vaguer, which
        downstream reasoning<->answer consistency tolerates; a rewrite that
        fabricates a year/citation/stat must not survive). So: (a) no NEW significant
        number (>=4 digits) absent from the original AND the answer, and (b) the
        rewrite must not balloon past its budget.

        In synthesis mode the answer supplies the allowed numbers and the length
        budget grows with the answer (a tiny bookkeeping draft may legitimately
        become a slightly longer genuine trace) but never EXCEEDS the answer - the
        reasoning is a thinking budget, not a re-derivation of the reply."""
        allowed = set(_digits(original))
        if answer:
            allowed |= set(_digits(answer))
        for d in _digits(rewrite):
            if len(d.replace(".", "")) >= 4 and d not in allowed:
                return False
        ceiling = 2.0 * len(original) + 250
        if answer:
            ceiling = min(max(ceiling, 0.9 * len(answer)), len(answer) + 250)
        if len(rewrite) > ceiling:
            return False
        return True
