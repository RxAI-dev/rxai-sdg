"""Reasoning-rewrite pass (problem 1, the durable fix).

The end-to-end test proved that gating annotator-voice reasoning by
reject-and-regenerate does not work: gpt-oss's native reasoning is annotator-
voiced at a ~100% base rate, so the classifier rejected nearly every resample
(222 regenerations, half the batch discarded). The responder system prompt
already forbids this voice explicitly and is ignored. So the voice cannot be
fixed by selection - it must be TRANSFORMED.

This pass rewrites a reasoning trace from task/format/compliance narration into
genuine first-person in-character thinking, preserving every substantive step,
claim, number, and conclusion (so reasoning<->answer consistency is untouched).
It only changes VOICE, never content. A rewrite that drifts from the original
(adds/removes facts, changes a conclusion) is rejected and the original kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .clients import LLMClient

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

    def rewrite(self, reasoning: str) -> Optional[str]:
        """Return the rewritten reasoning, or ``None`` if it should be left as-is
        (too short, call failed, or the rewrite failed the faithfulness guard)."""
        r = (reasoning or "").strip()
        if len(r) < self.min_chars:
            return None
        try:
            resp = self.client.generate(
                "REASONING:\n" + r, system_prompt=_REWRITE_SYSTEM,
                temperature=0.2, max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - a failed call leaves the original intact
            return None
        out = (resp.text or "").strip()
        if not out or len(out) < self.min_chars // 2:
            return None
        if not self._faithful(r, out):
            return None
        return out

    def _faithful(self, original: str, rewrite: str) -> bool:
        """Cheap guard against a rewrite that INVENTS specifics (the real risk - a
        voice rewrite that drops a detail merely makes the reasoning vaguer, which
        downstream reasoning<->answer consistency tolerates; a rewrite that fabricates
        a year/citation/stat must not survive). So: (a) no NEW significant number
        (>=4 digits, i.e. years, large figures, citation years) absent from the
        original, and (b) the rewrite must not balloon (a sign of added content).
        Small derived numbers (e.g. an intermediate sum 12.3) are allowed."""
        orig_nums = set(_digits(original))
        for d in _digits(rewrite):
            if len(d.replace(".", "")) >= 4 and d not in orig_nums:
                return False
        if len(rewrite) > 2.0 * len(original) + 250:
            return False
        return True
