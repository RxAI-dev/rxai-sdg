"""Responder parsing, quality filters, holistic judge, clients (spec §5.2, §6.3)."""

import random

from rxai_sdg.factory.clients import MockLLMClient
from rxai_sdg.factory.holistic import HolisticJudge
from rxai_sdg.factory.prompts import get_prompt_pack
from rxai_sdg.factory.quality import check_quality, QualityConfig
from rxai_sdg.factory.responder import Responder, split_reasoning_answer
from rxai_sdg.factory.schemas import Segment, Turn


def test_split_reasoning_answer():
    r, a = split_reasoning_answer("<think>reason here</think> Final answer.")
    assert r == "reason here"
    assert a == "Final answer."
    r2, a2 = split_reasoning_answer("No think block, just answer.")
    assert r2 == ""
    assert a2 == "No think block, just answer."


def test_responder_produces_segments_and_logits_ref():
    client = MockLLMClient(default="<think>step</think> Here is the answer.")
    resp = Responder(client, capture_logits=True)
    out = resp.generate([], "What is 2+2?", get_prompt_pack("math"), turn_index=0)
    turn = out.turn
    types = [s.segment_type for s in turn.segments]
    assert types == ["query", "reasoning", "answer"]
    assert turn.reasoning_flag is True
    assert out.malformed is False
    assert turn.topk_logits_ref is not None  # logits captured


def test_responder_no_logits_without_flag():
    client = MockLLMClient(default="<think>x</think> y")
    out = Responder(client, capture_logits=False).generate(
        [], "q", get_prompt_pack("general"), 0)
    assert out.turn.topk_logits_ref is None


def test_quality_refusal_and_length():
    assert check_quality("Here is a helpful, detailed answer.")[0] is True
    assert check_quality("I can't help with that.")[0] is False
    assert check_quality("ok")[0] is False  # too short
    assert check_quality("word " * 30 + ". " + "word " * 30,
                         QualityConfig(max_repeat_ratio=0.3))[0] is False  # repetition


def test_holistic_judge_parses_rubric():
    client = MockLLMClient(default=(
        '{"instruction_following": 8, "coherence": 9, "naturalness": 8, '
        '"role_consistency": 10, "recall_fidelity": 7, "appropriateness": 9, '
        '"reasoning_quality": 9, "reasoning_answer_consistency": 8, '
        '"sycophancy_resistance": 7, '
        '"flagged_turns": [{"turn_index": 2, "dimension": "reasoning_quality", '
        '"severity": 2, "evidence": "Thinking Process:"}], "notes": "minor nit"}'))
    judge = HolisticJudge(client, rng=random.Random(0))
    score = judge.score([Turn(0, [Segment("query", "q"), Segment("answer", "a")])])
    assert score == {"instruction_following": 8, "coherence": 9, "naturalness": 8,
                     "role_consistency": 10, "recall_fidelity": 7,
                     "appropriateness": 9, "reasoning_quality": 9,
                     "reasoning_answer_consistency": 8, "sycophancy_resistance": 7,
                     "flagged_turns": [{"turn_index": 2, "dimension": "reasoning_quality",
                                        "severity": 2, "evidence": "Thinking Process:"}],
                     "notes": "minor nit"}


def test_holistic_gate_blocks_failed_programmatic():
    judge = HolisticJudge(MockLLMClient(), rng=random.Random(0), sample_rate=1.0,
                          gate_on_programmatic=True)
    assert judge.should_score(programmatic_passed=False) is False
    assert judge.should_score(programmatic_passed=True) is True


def test_mock_client_handler_and_queue():
    q = MockLLMClient(responses=["a", "b"])
    assert q.generate("x").text == "a"
    assert q.generate("x").text == "b"
    assert q.generate("x").text == "b"  # repeats last
    h = MockLLMClient(handler=lambda prompt, **k: prompt.upper())
    assert h.generate("hi").text == "HI"
