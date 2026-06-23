"""Responder output-contract tests (spec §4.1, §7).

Covers the robust reasoning/answer parser, the malformed-output counter, the
memory-disclaimer filter and the chain-of-thought leak detector.
"""

import random

from rxai_sdg.factory.clients import MockLLMClient
from rxai_sdg.factory.config import FactoryConfig
from rxai_sdg.factory.factory_runner import DataFactory
from rxai_sdg.factory.prompts import get_prompt_pack
from rxai_sdg.factory.responder import (
    Responder, parse_response, split_reasoning_answer,
    is_memory_disclaimer, has_cot_leak, _segment_response,
)
from rxai_sdg.factory.clients import LLMResponse


# ---- configurable reasoning source (field vs inline <think>) ---------------

def test_reasoning_source_auto_prefers_field():
    # gpt-oss / Qwen3.5: dedicated field present -> used; <think> stripped from answer
    p = _segment_response("the genuine field CoT", "<think>x</think>final", source="auto")
    assert p.reasoning == "the genuine field CoT"
    assert p.answer == "final" and "<think>" not in p.answer


def test_reasoning_source_auto_falls_back_to_inline():
    # Qwen3-32B: empty field -> parse inline <think>
    p = _segment_response(None, "<think>inline CoT</think>the answer", source="auto")
    assert p.reasoning == "inline CoT"
    assert p.answer == "the answer"


def test_reasoning_source_inline_ignores_field():
    p = _segment_response("field CoT", "<think>inline CoT</think>ans", source="inline")
    assert p.reasoning == "inline CoT" and p.answer == "ans"


def test_reasoning_source_field_ignores_inline():
    p = _segment_response("field CoT", "<think>inline CoT</think>ans", source="field")
    assert p.reasoning == "field CoT"
    assert p.answer == "ans" and "<think>" not in p.answer
    # empty field under 'field' mode -> empty reasoning (reasoning_missing upstream)
    p2 = _segment_response(None, "<think>inline</think>ans", source="field")
    assert p2.reasoning == "" and not p2.well_formed


# ---- parser ----------------------------------------------------------------

def test_parse_well_formed_block():
    p = parse_response("<think>a</think>b")
    assert p.reasoning == "a"
    assert p.answer == "b"
    assert p.well_formed is True


def test_parse_well_formed_with_newlines():
    p = parse_response("<think>\nstep one\nstep two\n</think>\nThe final answer.")
    assert p.reasoning == "step one\nstep two"
    assert p.answer == "The final answer."
    assert p.well_formed is True


def test_parse_malformed_no_block_is_whole_answer():
    p = parse_response("just an answer, no think tags")
    assert p.reasoning is None
    assert p.answer == "just an answer, no think tags"
    assert p.well_formed is False


def test_parse_empty_reasoning_is_malformed():
    for empty in ("<think></think>real answer", "<think>...</think>real answer"):
        p = parse_response(empty)
        assert p.well_formed is False
        assert "</think>" not in p.answer
        assert "<think>" not in p.answer


def test_parse_strips_stray_closing_tag_from_answer():
    # reasoning leaked a stray closing tag into the remainder; never keep it
    p = parse_response("<think>reason</think> answer text </think>")
    assert "</think>" not in p.answer
    assert "<think>" not in p.answer


def test_parse_multiple_blocks_is_malformed_and_tag_free():
    p = parse_response("<think>a</think> x <think>b</think> y")
    assert p.well_formed is False
    assert "</think>" not in p.answer and "<think>" not in p.answer


def test_split_reasoning_answer_backcompat():
    r, a = split_reasoning_answer("<think>reason here</think> Final answer.")
    assert r == "reason here"
    assert a == "Final answer."
    r2, a2 = split_reasoning_answer("No think block, just answer.")
    assert r2 == ""
    assert a2 == "No think block, just answer."


def test_answer_segment_never_contains_think_tag():
    samples = [
        "<think>x</think>y",
        "no block",
        "<think>a</think> b </think>",
        "<think></think> c",
        "<think>a</think> m <think>n</think> o",
    ]
    for s in samples:
        assert "</think>" not in parse_response(s).answer
        assert "<think>" not in parse_response(s).answer


# ---- disclaimer / cot detectors --------------------------------------------

def test_memory_disclaimer_detector():
    assert is_memory_disclaimer("I cannot store personal information between conversations.")
    assert is_memory_disclaimer("I don't retain personal information between separate chat sessions.")
    assert is_memory_disclaimer("Each session is independent, so I can't recall earlier turns.")
    assert not is_memory_disclaimer("Your favorite color is teal, as you told me earlier.")


def test_cot_leak_detector():
    assert has_cot_leak("Draft 1: here is the text")
    assert has_cot_leak("Let's verify the result.")
    assert has_cot_leak("Final answer self-contained? Yes.")
    assert has_cot_leak("blah </think> leaked")
    assert not has_cot_leak("The answer is 42 because energy disperses.")


# ---- responder integration -------------------------------------------------

def test_responder_flags_malformed_and_builds_segments():
    good = Responder(MockLLMClient(default="<think>s</think> Final.")).generate(
        [], "q", get_prompt_pack("general"), 0)
    assert good.malformed is False
    assert good.turn.reasoning_flag is True
    assert [s.segment_type for s in good.turn.segments] == ["query", "reasoning", "answer"]

    bad = Responder(MockLLMClient(default="no block here at all")).generate(
        [], "q", get_prompt_pack("general"), 0)
    assert bad.malformed is True
    assert bad.turn.reasoning_flag is False
    assert [s.segment_type for s in bad.turn.segments] == ["query", "answer"]


def test_responder_prompt_has_no_qa_checklist_or_disclaimer_instruction():
    client = MockLLMClient(default="<think>x</think> y")
    pack = get_prompt_pack("general")
    Responder(client).generate([], "q", pack, 0)
    prompt = client.calls[-1]["prompt"].lower()
    sys = client.calls[-1]["system_prompt"].lower()
    # the internal QA checklist must not leak into the generation prompt
    assert "self-contained" not in prompt
    assert "no reference to" not in prompt
    # the system prompt is minimal and free of the echo-bait harness phrases the
    # native-reasoning model would parrot (mode A), including the "ongoing
    # conversation" framing that made it agonize about a turn-0 "contradiction".
    assert "expert" in sys
    for bad in ("persistent memory", "never deny having memory",
                "drawing on the whole conversation", "write only the final answer",
                "ongoing conversation"):
        assert bad not in sys
        assert bad not in prompt


def test_responder_passes_prior_turns_as_real_messages():
    from rxai_sdg.factory.schemas import Segment, Turn
    client = MockLLMClient(default="<think>r</think> a2")
    prior = [Turn(0, [Segment("query", "q1"), Segment("reasoning", "secret"),
                      Segment("answer", "a1")])]
    Responder(client).generate(prior, "q2", get_prompt_pack("general"), 1)
    call = client.calls[-1]
    # prior turn is passed as real role-tagged messages, reasoning excluded
    assert call["messages"] == [{"role": "user", "content": "q1"},
                                {"role": "assistant", "content": "a1"}]
    # the current user message is just the query - no transcript, no "User:" label
    assert call["prompt"] == "q2"


# ---- reasoning capture (reasoning_content field vs inline <think>) ---------

def test_reasoning_captured_from_reasoning_content_field():
    # endpoint returns reasoning in a SEPARATE field (the real Qwen3.5 behaviour)
    client = MockLLMClient(default=LLMResponse(
        text="The final answer stands alone.",
        reasoning="step 1: recall context\nstep 2: apply constraint"))
    out = Responder(client).generate([], "q", get_prompt_pack("general"), 0)
    assert out.turn.reasoning == "step 1: recall context\nstep 2: apply constraint"
    assert out.turn.reasoning_flag is True
    assert out.turn.answer == "The final answer stands alone."
    assert out.malformed is False
    assert out.reasoning_missing is False
    assert [s.segment_type for s in out.turn.segments] == ["query", "reasoning", "answer"]


def test_reasoning_field_strips_any_inline_think_from_answer():
    client = MockLLMClient(default=LLMResponse(
        text="<think>leftover</think>Clean answer.", reasoning="real reasoning"))
    out = Responder(client).generate([], "q", get_prompt_pack("general"), 0)
    assert out.turn.reasoning == "real reasoning"
    assert "<think>" not in out.turn.answer and "</think>" not in out.turn.answer
    assert out.turn.answer == "Clean answer."


def test_reasoning_falls_back_to_inline_think_block():
    # no reasoning_content field -> parse the inline <think> block from content
    client = MockLLMClient(default=LLMResponse(
        text="<think>inline reasoning here</think>The answer.", reasoning=None))
    out = Responder(client).generate([], "q", get_prompt_pack("general"), 0)
    assert out.turn.reasoning == "inline reasoning here"
    assert out.turn.answer == "The answer."
    assert out.turn.reasoning_flag is True
    assert out.reasoning_missing is False


def test_reasoning_missing_flagged_when_expected_but_absent():
    client = MockLLMClient(default="Just an answer with no reasoning at all.")
    out = Responder(client).generate([], "q", get_prompt_pack("general"), 0)
    assert out.turn.reasoning is None
    assert out.turn.reasoning_flag is False
    assert out.reasoning_missing is True  # reasoning mode expected, none produced


def test_reasoning_missing_increments_in_loop_stats():
    import random
    cfg = FactoryConfig(seed=0, concurrency=1, regeneration_limit=1)
    # a well-formed, non-disclaimer answer but with NO reasoning -> reasoning_missing
    client = MockLLMClient(default="A perfectly fine answer about the topic.")
    factory = DataFactory(cfg, client,
                          simulator_client=MockLLMClient(default="Tell me more."),
                          rng=random.Random(0))
    factory.generate(["Explain entropy."], band="short")
    assert factory.stats.loop.reasoning_missing > 0


class _DisclaimerClient:
    """Always answers with a memory disclaimer (well-formed think block)."""

    def generate(self, prompt, *, system_prompt="", temperature=0.7,
                 max_tokens=4096, capture_logits=False, **kw):
        # a clean disclaimer (no refusal phrasing) so the coherence gate - not the
        # quality refusal gate - is what fails the turn.
        return LLMResponse(
            text="<think>recall</think> I don't retain information between our "
                 "previous conversations, so each session is independent for me.")


def test_disclaimer_answer_fails_and_triggers_regeneration():
    cfg = FactoryConfig(seed=0, concurrency=1, regeneration_limit=2)
    factory = DataFactory(cfg, _DisclaimerClient(), rng=random.Random(0))
    records = factory.generate(["Explain entropy."], band="short")
    # an unsalvageable disclaimer first answer -> the conversation is discarded,
    # and the coherence gate fired (counted as regenerations + coherence failures)
    assert records == []
    assert factory.stats.conversations_discarded == 1
    assert factory.stats.loop.coherence_failures > 0
    assert factory.stats.loop.total_regenerations > 0
