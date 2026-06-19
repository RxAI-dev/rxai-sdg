"""Deterministic tests for the judge overhaul (no network): the segment-labeled
judge transcript, the leakage detectors, the deterministic pre-filter, the
sanitization pass, and the config-driven gate.

The real-endpoint behaviour of the LLM judge is validated separately in
``tools/validate_judge.py`` (Phase 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import random  # noqa: E402

from rxai_sdg.factory import (  # noqa: E402
    ConversationLoop, FactoryConfig, IntentPolicySampler, Responder,
    MockLLMClient, Segment, Turn, VerifyResult,
)
from rxai_sdg.factory.holistic import deterministic_prefilter, RUBRIC_AXES  # noqa: E402
from rxai_sdg.factory.responder import (  # noqa: E402
    format_transcript, format_transcript_for_judge, has_harness_leak,
    has_trailing_artifact, has_turn_index_leak, sanitize_reasoning,
    sanitize_generated_text,
)

import judge_fixtures  # noqa: E402


def _turn(idx, query, reasoning, answer, regen=0):
    segs = [Segment("query", query)]
    if reasoning is not None:
        segs.append(Segment("reasoning", reasoning))
    segs.append(Segment("answer", answer))
    return Turn(turn_index=idx, segments=segs,
                verification=VerifyResult(True, "", regen))


# --------------------------------------------------------------- transcript
def test_judge_transcript_includes_reasoning_and_labels():
    turns = [_turn(0, "hi", "because X", "hello"),
             _turn(1, "more?", None, "sure")]
    out = format_transcript_for_judge(turns)
    assert "[Turn 0]" in out and "[Turn 1]" in out
    assert "Reasoning: because X" in out
    assert "Reasoning: (none)" in out  # turn 1 has no reasoning
    assert "User: hi" in out and "Assistant: hello" in out


def test_plain_transcript_unchanged_no_reasoning_no_labels():
    # format_transcript (generation context) must stay user+assistant only.
    turns = [_turn(0, "hi", "secret reasoning", "hello")]
    out = format_transcript(turns)
    assert "secret reasoning" not in out
    assert "Turn" not in out
    assert out == "User: hi\nAssistant: hello"


# --------------------------------------------------------------- detectors
def test_turn_index_leak_detector():
    assert has_turn_index_leak("As we discussed in Turn 6, ...")
    assert has_turn_index_leak("see (Turn 8 JSON)")
    assert has_turn_index_leak('reference_turn_2 holds the value')
    assert has_turn_index_leak("going back to turn 3 for the value")
    # must NOT fire on physical / non-index uses
    assert not has_turn_index_leak("turn 90 degrees to the right")
    assert not has_turn_index_leak("it was a sharp turn 5 km later")


def test_harness_leak_detector():
    assert has_harness_leak("You are a helpful expert assistant with persistent memory")
    assert has_harness_leak("Thinking Process:\n1. Analyze the request")
    assert has_harness_leak("drawing on the whole conversation above")
    assert has_harness_leak("I never deny having memory")
    assert has_harness_leak("the contradictory system instructions confuse me")
    assert not has_harness_leak("Paris is the capital, on the Seine.")


def test_trailing_artifact_detector():
    assert has_trailing_artifact("...low-impact on the joints.cw")
    assert has_trailing_artifact("Ready to generate.cltr")
    assert has_trailing_artifact("final output.cw")
    # legit filenames / domains / sentences must not fire
    assert not has_trailing_artifact("see main.py")
    assert not has_trailing_artifact("visit example.com")
    assert not has_trailing_artifact("It releases oxygen.")


# --------------------------------------------------------------- sanitization
def test_sanitize_strips_thinking_header_and_artifact():
    r = sanitize_reasoning("Thinking Process:\n\n1. Analyze: the sum is 391.cw")
    assert not r.lower().startswith("thinking process")
    assert "391" in r and not r.endswith(".cw")
    # harness phrase NOT at the leading header is left for the pre-filter to fail
    assert "persistent memory" in sanitize_reasoning(
        "1. I have persistent memory here.")


def test_sanitize_answer_strips_artifact_only():
    assert sanitize_generated_text("the joints.cw") == "the joints."
    assert sanitize_generated_text("see main.py") == "see main.py"


# --------------------------------------------------------------- pre-filter
def test_prefilter_hard_fails_turn_index_in_answer():
    turns = [_turn(0, "q", "clean reasoning about the topic",
                   "As we discussed in Turn 6, Paris.")]
    res = deterministic_prefilter(turns)
    assert res.passed is False
    kinds = {h["kind"] for h in res.hard_fails}
    assert "turn_index_in_answer" in kinds


def test_prefilter_hard_fails_harness_in_reasoning():
    turns = [_turn(0, "q",
                   "You are a helpful expert assistant with persistent memory.",
                   "A clean answer.")]
    res = deterministic_prefilter(turns)
    assert res.passed is False
    assert "harness_in_reasoning" in {h["kind"] for h in res.hard_fails}


def test_prefilter_flags_degenerate_and_regen_softly():
    spiral = (". ".join(["avoid water"] * 8)) + ". check no water. check no water."
    turns = [_turn(0, "q", spiral, "The blue sea.", regen=4)]
    res = deterministic_prefilter(turns, regen_threshold=2)
    # soft flags do NOT hard-fail on their own
    assert res.passed is True
    flag_kinds = {f["kind"] for f in res.flags}
    assert "degenerate_reasoning" in flag_kinds
    assert "excess_regenerations" in flag_kinds


def test_prefilter_clean_passes():
    turns = [_turn(0, "How does photosynthesis work?",
                   "Light reactions split water; the Calvin cycle fixes CO2.",
                   "Plants use sunlight to turn CO2 and water into glucose and "
                   "oxygen.")]
    res = deterministic_prefilter(turns)
    assert res.passed is True and not res.hard_fails


# --------------------------------------------------------------- fixtures
def test_fixtures_prefilter_expectations():
    for fx in judge_fixtures.build_fixtures():
        res = deterministic_prefilter(fx.record.turns)
        assert res.passed is (not fx.prefilter_hard_fail), fx.name
        if fx.prefilter_kinds:
            got = {h["kind"] for h in res.hard_fails}
            assert fx.prefilter_kinds <= got, (fx.name, got)
        if fx.prefilter_flag_kinds:
            gotf = {f["kind"] for f in res.flags}
            assert fx.prefilter_flag_kinds <= gotf, (fx.name, gotf)


# --------------------------------------------------------------- gate logic
def _loop():
    cfg = FactoryConfig(seed=0)
    sampler = IntentPolicySampler(cfg.build_taxonomy(), cfg.intent_weights,
                                  cfg.policy_weights, rng=random.Random(0))
    return ConversationLoop(Responder(MockLLMClient()), sampler, cfg,
                            rng=random.Random(0))


def test_gate_rejects_on_prefilter_hard_fail_even_with_good_scores():
    loop = _loop()
    good = {k: 10 for k in RUBRIC_AXES}

    class _PF:  # minimal stand-in
        passed = False
    assert loop._holistic_ok(good, _PF()) is False


def test_gate_rejects_on_low_field():
    loop = _loop()
    score = {k: 10 for k in RUBRIC_AXES}
    score["reasoning_quality"] = 4  # below the configured min of 7

    class _PF:
        passed = True
    assert loop._holistic_ok(score, _PF()) is False


def test_gate_rejects_on_severity3_flag():
    loop = _loop()
    score = {k: 10 for k in RUBRIC_AXES}
    score["flagged_turns"] = [{"turn_index": 1, "dimension": "harness_leak",
                               "severity": 3, "evidence": "x"}]

    class _PF:
        passed = True
    assert loop._holistic_ok(score, _PF()) is False


def test_gate_passes_clean_high_scores():
    loop = _loop()
    score = {k: 9 for k in RUBRIC_AXES}
    score["flagged_turns"] = [{"turn_index": 0, "dimension": "naturalness",
                               "severity": 1, "evidence": "minor"}]

    class _PF:
        passed = True
    assert loop._holistic_ok(score, _PF()) is True


def test_gate_no_rubric_does_not_gate_when_prefilter_passes():
    loop = _loop()

    class _PF:
        passed = True
    assert loop._holistic_ok({"prefilter": {"passed": True}}, _PF()) is True
    # but a prefilter hard-fail still gates with no rubric
    class _PFbad:
        passed = False
    assert loop._holistic_ok({}, _PFbad()) is False
