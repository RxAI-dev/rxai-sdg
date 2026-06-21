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


def test_harness_leak_tone_bookkeeping():
    # "<style-adjective> tone:" mid-sentence output-planning bookkeeping (the form
    # that slipped into emitted data) must flag.
    assert has_harness_leak("Need empathetic tone: acknowledge frustration, give steps")
    assert has_harness_leak("Should be formal tone: subject, greeting, body")
    assert has_harness_leak("Tone: warm and knowledgeable, then list the steps")
    # substantive tone-ANALYSIS uses content adjectives, not delivery-style ones,
    # so it must NOT flag.
    assert not has_harness_leak("The poem's tone: melancholic, with a somber close.")
    assert not has_harness_leak("Her tone shifted from anger to calm across the scene.")


def test_harness_leak_gpt_oss_safety_meta():
    # gpt-oss reasons about its safety-RL harness on sensitive turns; these are
    # leakage (they reference the policy, not the person's situation).
    assert has_harness_leak("Must follow safety guidelines: provide a supportive response")
    assert has_harness_leak("I should give a safe completion here")
    assert has_harness_leak("This request is disallowed content")
    assert has_harness_leak("per openai policy on crisis")
    assert has_harness_leak("must follow policy for self-harm situations")
    assert has_harness_leak("the policy for suicide ideation says to provide resources")
    assert has_harness_leak("I must comply with the guidelines")


def test_harness_leak_topical_policy_not_flagged():
    # The FP guard: a substantive discussion that happens to use "policy" /
    # "guidelines" topically is genuine reasoning and must NOT hard-fail.
    assert not has_harness_leak(
        "The foederati policy of the late Roman Empire let tribes settle inside the limes.")
    assert not has_harness_leak(
        "Carbon sequestration policy differs sharply between the EU and the US.")
    assert not has_harness_leak(
        "Monetary policy tightening raises the cost of borrowing, cooling demand.")
    assert not has_harness_leak(
        "Here are some practical guidelines for composting at home.")
    assert not has_harness_leak(
        "Their privacy policy lets users export data, which is what she asked about.")


def test_trailing_artifact_detector():
    assert has_trailing_artifact("...low-impact on the joints.cw")
    assert has_trailing_artifact("Ready to generate.cltr")
    assert has_trailing_artifact("final output.cw")
    # legit filenames / domains / sentences must not fire
    assert not has_trailing_artifact("see main.py")
    assert not has_trailing_artifact("visit example.com")
    assert not has_trailing_artifact("It releases oxygen.")


# --------------------------------------------------------------- sanitization
def test_sanitize_is_thin_artifact_only():
    # sanitization is now THIN: it strips ONLY the trailing decoding artifact (C).
    # It does NOT scrub the "Thinking Process:" scaffold or harness phrases - those
    # are real defects the pre-filter must HARD-FAIL (the scrubber hid them before).
    r = sanitize_reasoning("Reasoning about the sum is 391.cw")
    assert "391" in r and not r.endswith(".cw")
    assert "Thinking Process:" in sanitize_reasoning("Thinking Process:\n1. step.")
    assert "persistent memory" in sanitize_reasoning("1. I have persistent memory here.")


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


def test_prefilter_hard_fails_degenerate_and_regen():
    spiral = (". ".join(["avoid water"] * 8)) + ". check no water. check no water."
    turns = [_turn(0, "q", spiral, "The blue sea.", regen=4)]
    res = deterministic_prefilter(turns, regen_threshold=2)
    kinds = {h["kind"] for h in res.hard_fails}
    # degenerate-loop reasoning AND excess regenerations are BOTH hard fails now
    assert res.passed is False
    assert "degenerate_reasoning" in kinds
    assert "excess_regenerations" in kinds


def test_prefilter_hard_fails_restart_spiral_and_numbered_flow():
    spiral = " ".join(f"Wait, let me reconsider point {i}." for i in range(8))
    turns = [_turn(0, "q", spiral, "ok.")]
    assert "restart_spiral" in {h["kind"] for h in deterministic_prefilter(turns).hard_fails}
    flow = "1. User: asked about X.\n2. Model: explained X.\n3. User: asked Y."
    turns2 = [_turn(0, "q", flow, "ok.")]
    assert "numbered_flow_in_reasoning" in {h["kind"] for h in deterministic_prefilter(turns2).hard_fails}


def test_prefilter_hard_fails_target_answer_leak():
    turns = [_turn(0, "q",
                   "Final Output Generation: (This matches the provided good response.)",
                   "An answer.")]
    assert "harness_in_reasoning" in {h["kind"] for h in deterministic_prefilter(turns).hard_fails}


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
