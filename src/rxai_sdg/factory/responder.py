"""Responder / Teacher component (spec §5.2).

Generates the reference answer for a turn with full-context generation. It must
be a strong, **memory-enabled** model that satisfies constraints correctly. The
provider is injected as an :class:`~rxai_sdg.factory.clients.LLMClient`.

Generation happens in *reasoning mode*: the teacher emits a single
``<think>...</think>`` block followed by the final answer. The output contract is
exactly ``<think>\\n{reasoning}\\n</think>\\n{answer}`` and the answer must stand
alone. :func:`parse_response` enforces strict, robust segmentation:

* exactly one well-formed ``<think>...</think>`` block with non-trivial content
  -> ``reasoning`` = block content, ``answer`` = the remainder, ``well_formed`` =
  True;
* otherwise -> the whole (tag-stripped) output is the ``answer``, no reasoning
  segment, ``well_formed`` = False (a *malformed* output).

A ``</think>`` tag (or other chain-of-thought marker) is **never** left inside an
``answer`` segment. The conversation loop counts malformed outputs and treats
memory-disclaimer answers as failures to be regenerated (§4.1, §4.4).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Optional

from .clients import LLMClient
from .prompts import PromptPack
from .schemas import Segment, Turn

# A single, non-greedy think block. DOTALL so reasoning may span lines.
_THINK_BLOCK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
# Any stray opening/closing think tag (used to detect / scrub leakage).
_THINK_TAG_RE = re.compile(r"</?think\s*>", re.IGNORECASE)
# Reasoning content that is effectively empty.
_EMPTY_REASONING = {"", ".", "..", "...", "…"}

# Chain-of-thought / internal-QA markers that must never appear in an answer.
_COT_LEAK_PATTERNS = [
    r"</?think\s*>",
    r"\bdraft\s*\d+\b",
    r"\blet'?s (?:verify|double-check|check)\b",
    r"\bself-?contained\??\b",
    r"\bno reference to (?:the )?reasoning\b",
    r"\bfinal answer self-contained\b",
    r"\bscratchpad\b",
    r"\bchain[- ]of[- ]thought\b",
]
_COT_LEAK_RE = re.compile("|".join(_COT_LEAK_PATTERNS), re.IGNORECASE)

# Memory-disclaimer phrasing. Fatal for a memory-model dataset (§4.1): any answer
# matching these is treated as a failed turn and regenerated.
_MEMORY_DISCLAIMER_PATTERNS = [
    r"\bi (?:can'?t|cannot|don'?t|do not|am not able to|won'?t) "
    r"(?:store|retain|remember|recall|keep|save|hold on to|access) [^.?!\n]*"
    r"(?:between|across|from) (?:[^.?!\n]*?)(?:conversations?|sessions?|chats?|interactions?)",
    r"\bi (?:don'?t|do not) (?:have|retain|keep) (?:any )?memory of "
    r"(?:our )?(?:previous|past|earlier|prior) (?:conversations?|sessions?|chats?|messages?|turns?)",
    r"\beach (?:conversation|session|chat) is (?:independent|separate|isolated)\b",
    r"\bi (?:have no|don'?t have a) memory (?:of|between|across)\b",
    r"\bi (?:start|begin) (?:each|every) (?:conversation|session|chat) (?:fresh|anew|from scratch)\b",
    r"\bas an ai,? i (?:can'?t|cannot|don'?t|do not) (?:store|retain|remember|recall) [^.?!\n]*"
    r"(?:information|details|data)\b",
    r"\bi (?:can'?t|cannot|don'?t|do not) remember (?:anything|information|details) "
    r"(?:from|between|across) (?:previous|past|earlier|prior|separate) ",
]
_MEMORY_DISCLAIMER_RE = re.compile("|".join(_MEMORY_DISCLAIMER_PATTERNS), re.IGNORECASE)


@dataclass
class ParsedResponse:
    """Result of strictly segmenting a raw generation."""

    reasoning: Optional[str]
    answer: str
    well_formed: bool


@dataclass
class ResponderOutput:
    """A generated turn plus the parser's malformed flag (§4.1).

    ``reasoning_missing`` is set when the responder runs in reasoning mode (the
    default) but the response carried **no** reasoning - neither a separate
    ``reasoning_content`` field nor an inline ``<think>`` block. It surfaces
    endpoint misconfiguration (reasoning silently dropped or disabled).
    """

    turn: Turn
    malformed: bool
    reasoning_missing: bool = False


def format_transcript(turns: list[Turn]) -> str:
    """Render completed turns as a plain chat transcript for full-context prompts."""
    lines: list[str] = []
    for t in turns:
        if t.query:
            lines.append(f"User: {t.query}")
        if t.answer:
            lines.append(f"Assistant: {t.answer}")
    return "\n".join(lines)


def _strip_think_tags(text: str) -> str:
    """Remove stray ``<think>`` / ``</think>`` tag markers (content preserved)."""
    return _THINK_TAG_RE.sub("", text).strip()


def parse_response(text: Optional[str]) -> ParsedResponse:
    """Strictly split a raw generation into reasoning / answer segments.

    See the module docstring for the exact contract. The returned ``answer`` is
    guaranteed to contain no ``<think>``/``</think>`` tag.
    """
    if not text:
        return ParsedResponse(reasoning=None, answer="", well_formed=False)

    blocks = list(_THINK_BLOCK_RE.finditer(text))
    if len(blocks) == 1:
        m = blocks[0]
        reasoning = m.group(1).strip()
        remainder = (text[: m.start()] + text[m.end():]).strip()
        answer = _strip_think_tags(remainder)
        if reasoning.lower() not in _EMPTY_REASONING:
            # Well-formed: one block, non-trivial reasoning, tag-free answer.
            return ParsedResponse(reasoning=reasoning, answer=answer, well_formed=True)

    # Malformed: zero / multiple blocks, or an empty reasoning block. Treat the
    # whole (tag-stripped) output as the answer; no reasoning segment.
    return ParsedResponse(reasoning=None, answer=_strip_think_tags(text), well_formed=False)


def split_reasoning_answer(text: str) -> tuple[str, str]:
    """Back-compat helper returning ``(reasoning, answer)`` (reasoning ``""`` if none)."""
    parsed = parse_response(text)
    return (parsed.reasoning or ""), parsed.answer


def is_memory_disclaimer(answer: str) -> bool:
    """True if ``answer`` denies having conversational memory (§4.1)."""
    return bool(_MEMORY_DISCLAIMER_RE.search(answer or ""))


def has_cot_leak(answer: str) -> bool:
    """True if ``answer`` leaks chain-of-thought / internal-QA markers (§4.1)."""
    return bool(_COT_LEAK_RE.search(answer or ""))


def _segment_response(reasoning_field: Optional[str], content: str) -> ParsedResponse:
    """Segment a raw response, preferring a separate ``reasoning_content`` field.

    Reasoning models on OpenAI-compatible endpoints (e.g. Qwen3.5) return the
    chain of thought in ``message.reasoning_content`` rather than inline. When that
    field is populated we use it directly and strip any ``<think>`` block/tag from
    the content to form the answer; otherwise we fall back to inline ``<think>``
    parsing (:func:`parse_response`).
    """
    if reasoning_field is not None and reasoning_field.strip().lower() not in _EMPTY_REASONING:
        reasoning = reasoning_field.strip()
        answer = _strip_think_tags(_THINK_BLOCK_RE.sub("", content or ""))
        return ParsedResponse(reasoning=reasoning, answer=answer, well_formed=True)
    return parse_response(content)


class Responder:
    def __init__(self, client: LLMClient, capture_logits: bool = False,
                 max_tokens: int = 4096, temperature: float = 0.7,
                 reasoning_mode: bool = True):
        self.client = client
        self.capture_logits = capture_logits
        self.max_tokens = max_tokens
        self.temperature = temperature
        #: the responder model reasons by default, so a reasoning segment is
        #: expected on every turn; an empty one is counted as ``reasoning_missing``.
        self.reasoning_mode = reasoning_mode

    def generate(
        self,
        prior_turns: list[Turn],
        query: str,
        prompt_pack: PromptPack,
        turn_index: int,
        intent: Optional[str] = None,
        policy: Optional[str] = None,
        active_constraints_note: str = "",
    ) -> ResponderOutput:
        """Generate one turn. Returns the parsed :class:`Turn` and a malformed flag.

        The prompt carries only the conversation context, any still-active standing
        instructions, and the bare reasoning/answer output contract - never our
        internal QA checklist.
        """
        transcript = format_transcript(prior_turns)
        parts = []
        if transcript:
            parts.append("Conversation so far:\n" + transcript)
        if active_constraints_note:
            parts.append(active_constraints_note)
        parts.append(f"User: {query}")
        parts.append(
            "Respond as the assistant, drawing on the whole conversation above. "
            "Write only the final answer.")
        prompt = "\n\n".join(parts)

        resp = self.client.generate(
            prompt,
            system_prompt=prompt_pack.responder_system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            capture_logits=self.capture_logits,
        )
        parsed = _segment_response(getattr(resp, "reasoning", None), resp.text)

        reasoning_flag = bool(parsed.reasoning and parsed.reasoning.strip())
        reasoning_missing = self.reasoning_mode and not reasoning_flag

        segments = [Segment("query", query)]
        if parsed.reasoning is not None:
            segments.append(Segment("reasoning", parsed.reasoning))
        segments.append(Segment("answer", parsed.answer))

        logits_ref = None
        if self.capture_logits and resp.logits is not None:
            logits_ref = f"logits://{uuid.uuid4()}"

        turn = Turn(
            turn_index=turn_index,
            segments=segments,
            intent=intent,
            policy=policy,
            reasoning_flag=reasoning_flag,
            topk_logits_ref=logits_ref,
        )
        return ResponderOutput(
            turn=turn, malformed=not parsed.well_formed,
            reasoning_missing=reasoning_missing)
