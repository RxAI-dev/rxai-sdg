"""Responder / Teacher component (spec §5.2).

Generates the reference answer for a turn with full-context (stateless)
generation. It must be a strong model that satisfies constraints correctly. The
provider is injected as an :class:`~rxai_sdg.factory.clients.LLMClient`.

All generation happens in *reasoning mode*: the teacher emits a
``<think>...</think>`` block followed by the final answer. The two are parsed
into separate ``reasoning`` and ``answer`` segments. Derived instruct/mixed
variants are produced later by the writer (spec §8).
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from .clients import LLMClient
from .prompts import PromptPack
from .schemas import Segment, Turn

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def format_transcript(turns: list[Turn]) -> str:
    """Render completed turns as a plain chat transcript for full-context prompts."""
    lines: list[str] = []
    for t in turns:
        if t.query:
            lines.append(f"User: {t.query}")
        if t.answer:
            lines.append(f"Assistant: {t.answer}")
    return "\n".join(lines)


def split_reasoning_answer(text: str) -> tuple[str, str]:
    """Split a raw generation into ``(reasoning, answer)``.

    Supports an explicit ``<think>...</think>`` block. When no block is present,
    reasoning is empty and the whole text is the answer.
    """
    if text is None:
        return "", ""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1).strip()
        answer = _THINK_RE.sub("", text, count=1).strip()
        return reasoning, answer
    return "", text.strip()


class Responder:
    def __init__(self, client: LLMClient, capture_logits: bool = False,
                 max_tokens: int = 4096, temperature: float = 0.7):
        self.client = client
        self.capture_logits = capture_logits
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        prior_turns: list[Turn],
        query: str,
        prompt_pack: PromptPack,
        turn_index: int,
        intent: Optional[str] = None,
        policy: Optional[str] = None,
        active_constraints_note: str = "",
    ) -> Turn:
        transcript = format_transcript(prior_turns)
        parts = []
        if transcript:
            parts.append("Conversation so far:\n" + transcript)
        if active_constraints_note:
            parts.append(active_constraints_note)
        parts.append(f"User: {query}")
        parts.append(
            "Respond as the Assistant. Think inside <think>...</think>, then give "
            "the final answer. The final answer must be self-contained.")
        prompt = "\n\n".join(parts)

        resp = self.client.generate(
            prompt,
            system_prompt=prompt_pack.responder_system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            capture_logits=self.capture_logits,
        )
        reasoning, answer = split_reasoning_answer(resp.text)

        segments = [Segment("query", query)]
        if reasoning:
            segments.append(Segment("reasoning", reasoning))
        segments.append(Segment("answer", answer))

        logits_ref = None
        if self.capture_logits and resp.logits is not None:
            logits_ref = f"logits://{uuid.uuid4()}"

        return Turn(
            turn_index=turn_index,
            segments=segments,
            intent=intent,
            policy=policy,
            reasoning_flag=bool(reasoning),
            topk_logits_ref=logits_ref,
        )
