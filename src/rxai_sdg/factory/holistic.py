"""Optional holistic LLM-judge (spec §6.3).

Off by default. When enabled it is *gated* (only score conversations that pass
programmatic checks) and/or *sampled* (only score a fraction of conversations,
since 25-35 turn conversations are expensive to judge). It emits a **structured
rubric** rather than a single scalar so downstream filtering can target an axis.

IMPORTANT: keep this judge's model/prompt **separate** from the MT-Bench eval
judge, to avoid optimising the dataset toward the same judge we measure with.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Optional

from .clients import LLMClient
from .responder import format_transcript
from .schemas import Turn

RUBRIC_AXES = ["instruction_following", "coherence", "recall_fidelity", "naturalness"]

_JUDGE_SYSTEM = (
    "You are a strict but fair conversation-quality judge for a training dataset. "
    "Score the assistant's performance across the whole multi-turn conversation. "
    "Return ONLY a compact JSON object with integer scores 1-10 for the keys: "
    + ", ".join(RUBRIC_AXES) + "."
)


@dataclass
class HolisticJudge:
    client: LLMClient
    rng: Optional[random.Random] = None
    sample_rate: float = 0.1
    gate_on_programmatic: bool = True
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def should_score(self, programmatic_passed: bool) -> bool:
        if self.gate_on_programmatic and not programmatic_passed:
            return False
        return self.rng.random() < self.sample_rate

    def score(self, turns: list[Turn]) -> Optional[dict[str, int]]:
        transcript = format_transcript(turns)
        prompt = (
            "Evaluate this conversation and return the rubric JSON.\n\n" + transcript)
        resp = self.client.generate(
            prompt, system_prompt=_JUDGE_SYSTEM, temperature=0.0,
            max_tokens=self.max_tokens)
        return self._parse(resp.text)

    @staticmethod
    def _parse(text: str) -> Optional[dict[str, int]]:
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
        out: dict[str, int] = {}
        for axis in RUBRIC_AXES:
            val = data.get(axis)
            if isinstance(val, (int, float)):
                out[axis] = int(max(1, min(10, round(val))))
        return out or None
