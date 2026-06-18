"""Holistic LLM-judge (fix G).

After each conversation is generated the judge is called **once on the whole
conversation** (always-on; cost is acceptable) and a structured rubric is stored
on the record for later filtering. It is also the **coherence gate** that catches
semantic failures the programmatic verifier misses.

The judge must run on a **different model family** from the Responder/Simulator
(the ``JUDGE_MODEL`` is non-Qwen, e.g. ``gpt-oss-120b``) and is kept conceptually
separate from any MT-Bench-style eval judge, so we never optimise the dataset
toward the judge we measure with.
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

#: The six 1-10 rubric axes stored on every record.
RUBRIC_AXES = [
    "instruction_following", "coherence", "naturalness",
    "role_consistency", "recall_fidelity", "appropriateness",
]

_JUDGE_SYSTEM = (
    "You are a strict but fair conversation-quality judge for a training dataset. "
    "You will be given a transcript of a conversation between a USER and an "
    "ASSISTANT, delimited by <conversation> tags. Your ONLY job is to rate the "
    "ASSISTANT's performance across that transcript. The assistant was NOT asked to "
    "produce any rubric, score, or JSON - that JSON is YOUR output, not something "
    "the assistant should have written. Judge the assistant only against what the "
    "USER turns inside the transcript actually request.\n\n"
    "Score each axis from 1 (terrible) to 10 (excellent):\n"
    "- instruction_following: did the assistant do what each user turn in the "
    "transcript asked, including any formatting/lexical constraints (reformat, "
    "bullets, length, etc.)?\n"
    "- coherence: does the conversation hang together; do answers follow from the "
    "real prior content?\n"
    "- naturalness: do the turns read like a real human-assistant conversation?\n"
    "- role_consistency: does each speaker stay in role (the user asks, the "
    "assistant answers; nobody claims the other's output)?\n"
    "- recall_fidelity: when a user refers back to earlier content or a detail they "
    "shared, does the assistant recall it correctly (no false memory, no 'I can't "
    "remember' when it should)?\n"
    "- appropriateness: is the tone appropriate to the topic, especially for "
    "sensitive/distressing topics (supportive, never flippant or trivializing)?\n\n"
    "Think briefly, then output ONLY a JSON object with integer keys "
    + ", ".join(RUBRIC_AXES) +
    ' and a "notes" string of at most one sentence naming the single worst issue '
    "in the assistant's performance (empty string if none). No prose outside the JSON."
)


@dataclass
class HolisticJudge:
    client: LLMClient
    rng: Optional[random.Random] = None
    sample_rate: float = 1.0
    gate_on_programmatic: bool = False
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def should_score(self, programmatic_passed: bool) -> bool:
        """Back-compat gate. The loop now scores every conversation (always-on)."""
        if self.gate_on_programmatic and not programmatic_passed:
            return False
        return self.rng.random() < self.sample_rate

    def score(self, turns: list[Turn]) -> Optional[dict[str, object]]:
        transcript = format_transcript(turns)
        prompt = (
            "Rate the ASSISTANT in the conversation below and return the rubric "
            "JSON (your output only - the assistant was not asked for any JSON).\n\n"
            "<conversation>\n" + transcript + "\n</conversation>")
        try:
            resp = self.client.generate(
                prompt, system_prompt=_JUDGE_SYSTEM, temperature=0.0,
                max_tokens=self.max_tokens)
        except Exception:
            return None
        return self._parse(resp.text)

    @staticmethod
    def _parse(text: Optional[str]) -> Optional[dict[str, object]]:
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
        out: dict[str, object] = {}
        for axis in RUBRIC_AXES:
            val = data.get(axis)
            if isinstance(val, (int, float)):
                out[axis] = int(max(1, min(10, round(val))))
        if not out:
            return None
        notes = data.get("notes")
        out["notes"] = str(notes)[:200] if notes else ""
        return out
