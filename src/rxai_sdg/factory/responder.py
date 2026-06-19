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
    """Render completed turns as a plain chat transcript for full-context prompts.

    NB: user+assistant only, **no turn numbers** - this feeds the
    Responder/Simulator generation context, where a turn index would be poison
    (the inference-time model has no turn numbering). Do not add labels here. The
    holistic judge uses :func:`format_transcript_for_judge` instead.
    """
    lines: list[str] = []
    for t in turns:
        if t.query:
            lines.append(f"User: {t.query}")
        if t.answer:
            lines.append(f"Assistant: {t.answer}")
    return "\n".join(lines)


def format_transcript_for_judge(turns: list[Turn]) -> str:
    """Render turns for the holistic judge: segment-delimited, labeled, reasoning shown.

    Unlike :func:`format_transcript` (user+assistant only, used for generation
    context) this renders **all three segments** of every turn - crucially the
    teacher ``reasoning`` - with explicit ``[Turn i]`` labels so the judge can cite
    which turn failed.

    The ``[Turn i]`` labels here are JUDGE INPUT only; they are never part of the
    emitted training data and must not be confused with turn-index *leakage*
    (failure mode B), which is about turn numbers appearing inside a generated
    ``answer``/``reasoning`` segment.
    """
    lines: list[str] = []
    for t in turns:
        lines.append(f"[Turn {t.turn_index}]")
        lines.append(f"User: {t.query or ''}")
        reasoning = t.reasoning
        lines.append(f"Reasoning: {reasoning if (reasoning and reasoning.strip()) else '(none)'}")
        lines.append(f"Assistant: {t.answer or ''}")
        lines.append("")
    return "\n".join(lines).rstrip()


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


# ---------------------------------------------------------------------------
# Leakage detection + sanitization (failure modes A / B / C)
# ---------------------------------------------------------------------------
#
# These regexes are shared by the generation sanitization pass (here) and the
# judge's deterministic pre-filter (``holistic.deterministic_prefilter``). The
# pre-filter HARD-FAILS on them; the sanitization pass below removes only the
# mechanical, meaning-preserving artifacts (a leading "Thinking Process:" scaffold
# and a glued trailing token) so the primary fix stays the generation prompt, not
# string-stripping (task Phase 3).

# (A) Harness / meta phrases that must never appear in stored *reasoning*. These
# are high-signal: organic, substance-only reasoning does not contain them. They
# are checked against ``reasoning`` only (never answers).
HARNESS_REASONING_RES: list[re.Pattern] = [
    re.compile(r"persistent memory", re.IGNORECASE),
    re.compile(r"you are a (?:helpful |expert |strong |memory-enabled )*assistant", re.IGNORECASE),
    re.compile(r"drawing on the (?:whole|entire) conversation", re.IGNORECASE),
    re.compile(r"the (?:whole |entire )?conversation above", re.IGNORECASE),
    re.compile(r"write only the final answer", re.IGNORECASE),
    re.compile(r"never deny having memory", re.IGNORECASE),
    re.compile(r"make each inferential step explicit and checkable", re.IGNORECASE),
    re.compile(r"\bthinking process\s*:", re.IGNORECASE),
    re.compile(r"contradictory .{0,40}?instructions?", re.IGNORECASE),
    re.compile(r"\bthe system (?:prompt|message|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bmy (?:system )?(?:prompt|instructions?)\b", re.IGNORECASE),
    re.compile(r"\bas an ai language model\b", re.IGNORECASE),
]


def has_harness_leak(reasoning: str) -> Optional[str]:
    """Return the matched harness snippet if ``reasoning`` leaks a meta phrase."""
    for rx in HARNESS_REASONING_RES:
        m = rx.search(reasoning or "")
        if m:
            return m.group(0)
    return None


# (B) Turn-index references. FATAL inside an *answer* (corrupts the training
# target) and illegitimate anywhere. Capital-T "Turn N" / "reference_turn_N" are
# the canonical leaks; lowercase "turn N" requires an explicit positional cue so
# we never flag physical "turn 90 degrees".
_TURN_INDEX_STRICT_RE = re.compile(r"\bTurn\s+\d+\b|\breference_turn_\d+\b")
_TURN_INDEX_CUE_RE = re.compile(
    r"\b(?:in|from|during|at|see|per|back in|back to|discussed in|mentioned in|"
    r"as (?:we|you) (?:discussed|said|noted|mentioned) in)\s+turn\s+\d+\b",
    re.IGNORECASE)


def has_turn_index_leak(text: str) -> bool:
    """True if ``text`` references a turn by index (failure mode B)."""
    t = text or ""
    return bool(_TURN_INDEX_STRICT_RE.search(t) or _TURN_INDEX_CUE_RE.search(t))


# (C) Trailing generation artifact: a corrupted short token glued to the final
# word, e.g. "...output.cw", "...generate.cltr", "...Ready.cw". Detection is
# broad; sanitization only strips the known corruption suffixes so a real filename
# / domain ending (".py", ".com") is never mangled.
_ARTIFACT_KNOWN_RE = re.compile(r"\.(?:cw|cltr|clt|cwt|ctr)\b\s*$|(?:\bcw|\bcltr)\s*$")
_ARTIFACT_GENERIC_RE = re.compile(r"[A-Za-z]{3,}\.([a-z]{1,4})\s*$")
_ARTIFACT_ALLOW = {
    "com", "org", "net", "io", "ai", "co", "gov", "edu", "py", "js", "ts",
    "md", "txt", "csv", "json", "html", "htm", "xml", "yaml", "yml", "pdf",
    "png", "jpg", "jpeg", "gif", "svg", "sh", "go", "rs", "rb", "cpp", "etc",
}
_ARTIFACT_STRIP_RE = re.compile(r"\s*\.(?:cw|cltr|clt|cwt|ctr)\b\s*$")


def has_trailing_artifact(text: str) -> bool:
    """True if ``text`` ends with a corrupted glued token (failure mode C)."""
    t = (text or "").rstrip()
    if not t:
        return False
    if _ARTIFACT_KNOWN_RE.search(t):
        return True
    m = _ARTIFACT_GENERIC_RE.search(t)
    return bool(m and m.group(1).lower() not in _ARTIFACT_ALLOW)


# "standing instruction" is reserved schema vocabulary in this system (it named
# the old raw-spec note). A native-reasoning model occasionally uses it as plain
# English when an active standing constraint exists. That is not a schema leak,
# but it collides with our reserved term, so we normalize the exact phrase to a
# meaning-preserving synonym in the captured reasoning. The genuine schema tokens
# (json_valid, top_type, forbidden_token, constraint_spec) are never English and
# are handled by the loop's spec-leak coherence gate instead.
_RESERVED_PHRASE_RE = re.compile(r"\bstanding (instruction)", re.IGNORECASE)

# A leading scaffold header this native-reasoning model emits on essentially every
# turn ("Thinking Process:\n\n1. **Analyze the Request:**..."). The header itself
# is failure-mode-A scaffolding (it cannot be prompted away - the model echoes any
# instruction about it straight back into the reasoning), so it is stripped here
# as a meaning-preserving sanitization. The reasoning's substance is untouched.
_THINKING_HEADER_RE = re.compile(
    r"\A\s*(?:thinking process|thought process|reasoning process|chain[- ]of[- ]thought|"
    r"here'?s my (?:thinking|reasoning|thought process)|let me think(?: this through)?|"
    r"my (?:reasoning|thought process))\s*:?\s*\n+",
    re.IGNORECASE)


def _normalize_reasoning(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return _RESERVED_PHRASE_RE.sub(r"ongoing \1", text)


def sanitize_reasoning(text: Optional[str]) -> Optional[str]:
    """Mechanical, meaning-preserving clean-up of a generated reasoning segment.

    Strips the leading "Thinking Process:" scaffold, normalizes the reserved
    "standing instruction" phrase, and strips a glued trailing artifact. It does
    **not** attempt to rewrite substantive content - genuine harness/meta leakage
    woven into the reasoning is left for the judge pre-filter to hard-fail.
    """
    if not text:
        return text
    text = _THINKING_HEADER_RE.sub("", text, count=1)
    text = _normalize_reasoning(text) or ""
    text = _ARTIFACT_STRIP_RE.sub(".", text)
    return text.strip()


def sanitize_generated_text(text: Optional[str]) -> Optional[str]:
    """Strip a glued trailing artifact from a generated answer / user query."""
    if not text:
        return text
    return _ARTIFACT_STRIP_RE.sub(".", text).strip()


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
            # Plain chat context, no turn numbers and no "draw on the conversation
            # above" framing - that phrasing was getting echoed verbatim into the
            # native-reasoning model's reasoning trace (failure mode A).
            parts.append(transcript)
        if active_constraints_note:
            parts.append(active_constraints_note)
        parts.append(f"User: {query}")
        prompt = "\n\n".join(parts)

        resp = self.client.generate(
            prompt,
            system_prompt=prompt_pack.responder_system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            capture_logits=self.capture_logits,
        )
        parsed = _segment_response(getattr(resp, "reasoning", None), resp.text)
        # Sanitization pass (safety net for failure modes A/C): strip the leading
        # "Thinking Process:" scaffold and any glued trailing artifact. The primary
        # fix is the harness-free prompt above; this only removes mechanical noise.
        parsed.reasoning = sanitize_reasoning(parsed.reasoning)
        parsed.answer = sanitize_generated_text(parsed.answer) or ""

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
