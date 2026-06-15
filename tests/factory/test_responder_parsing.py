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
    is_memory_disclaimer, has_cot_leak,
)
from rxai_sdg.factory.clients import LLMResponse


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
    # the system prompt frames memory and forbids disclaimers
    assert "memory" in sys
    assert "never" in sys


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
