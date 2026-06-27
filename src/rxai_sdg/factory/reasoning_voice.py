"""Small LLM classifier gate for reasoning voice (problem 1: D1 residual).

Regex-gating the annotator-voice class does not converge: three rounds of
pattern-broadening, and a fresh run still surfaces new paraphrases ("No extra
constraints", "Let's craft answer", "The user wants ...", "Provide answer.").
A small classifier judges the PROPERTY ("is this the assistant genuinely
thinking about the problem, or narrating the annotation task / planning output
format / referencing compliance?") instead of enumerating surface forms, so it
generalises to paraphrases.

It is a BACKSTOP: the free regex (``has_harness_leak``) runs first and handles
the obvious cases; the classifier is only consulted when regex passes, to catch
what regex missed. A match -> regenerate the turn (same path as the regex gate).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .clients import LLMClient

_VOICE_SYSTEM = (
    "You classify a single REASONING trace written by an AI assistant while it "
    "works on a user's request. Decide whether the reasoning is:\n"
    "  GENUINE  - the assistant actually thinking about the SUBSTANCE of the "
    "problem (the facts, the maths, the argument, the user's situation).\n"
    "  ANNOTATOR - the reasoning instead narrates the TASK itself: inventorying "
    "format/length constraints ('no special formatting constraints'), planning "
    "the output shape ('produce five bullet points'), narrating compliance "
    "('we need to comply', 'let's craft the answer'), or describing the request "
    "in third person ('the user wants ...').\n"
    "Brief task-framing mixed into otherwise substantive reasoning is GENUINE; "
    "only answer ANNOTATOR when the trace is dominated by task/format/compliance "
    "narration rather than thinking about the content.\n"
    "Answer with exactly one word: GENUINE or ANNOTATOR."
)

_LABEL_RE = re.compile(r"\b(ANNOTATOR|GENUINE)\b", re.IGNORECASE)


@dataclass
class ReasoningVoiceClassifier:
    client: LLMClient
    max_tokens: int = 2000   # a reasoning classifier model may think first
    enabled: bool = True
    #: minimum reasoning length (chars) worth a call - tiny traces are cheap to
    #: let the regex handle and not worth an LLM round-trip.
    min_chars: int = 40

    def classify(self, reasoning: str) -> Optional[str]:
        r = (reasoning or "").strip()
        if len(r) < self.min_chars:
            return None
        try:
            resp = self.client.generate(
                "REASONING:\n" + r[:4000], system_prompt=_VOICE_SYSTEM,
                temperature=0.0, max_tokens=self.max_tokens)
        except Exception:  # noqa: BLE001 - a failed call must NOT gate
            return None
        text = resp.text or ""
        # take the LAST label token (a reasoning model states its verdict last).
        labels = _LABEL_RE.findall(text)
        if not labels:
            return None
        return labels[-1].upper()

    def is_annotator(self, reasoning: str) -> bool:
        return self.classify(reasoning) == "ANNOTATOR"
