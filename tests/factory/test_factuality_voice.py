"""Deterministic tests (no network) for the two new gates:
- FactChecker (problem 2): decomposed claim verification.
- ReasoningVoiceClassifier (problem 1): annotator-voice backstop.

The real-endpoint behaviour is validated separately; here we drive both with a
scripted client so the parsing, gating policy, and fail-open semantics are pinned.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rxai_sdg.factory.clients import LLMResponse  # noqa: E402
from rxai_sdg.factory.factuality import (  # noqa: E402
    AnswerRepairer, FactChecker, _last_json_object,
)
from rxai_sdg.factory.reasoning_voice import ReasoningVoiceClassifier  # noqa: E402
from rxai_sdg.factory.reasoning_rewrite import ReasoningRewriter  # noqa: E402
from rxai_sdg.factory.schemas import Segment, Turn  # noqa: E402


class ScriptedClient:
    """Returns a fixed text (optionally raising) for every generate call."""

    def __init__(self, text="", raise_exc=False):
        self.text = text
        self.raise_exc = raise_exc
        self.calls = 0

    def generate(self, prompt, *, system_prompt="", temperature=0.7,
                 max_tokens=4096, **kwargs):
        self.calls += 1
        if self.raise_exc:
            raise RuntimeError("endpoint down")
        return LLMResponse(text=self.text)


def _turn(idx, answer, reasoning="thinking about it"):
    return Turn(turn_index=idx, segments=[
        Segment("query", "q"), Segment("reasoning", reasoning),
        Segment("answer", answer)])


# --------------------------------------------------------------- FactChecker
def test_factcheck_flags_false_named_claim():
    client = ScriptedClient(
        '{"false_claims":[{"claim":"Van Gogh played by Thomas Murray",'
        '"correction":"Tony Curran"}],"any_false":true}')
    fc = FactChecker(client).check([_turn(0, "Van Gogh was played by Thomas Murray.")])
    assert fc.available is True
    assert fc.passed is False
    assert fc.false_claims and "Tony Curran" in fc.false_claims[0]["correction"]


def test_factcheck_passes_clean():
    client = ScriptedClient('{"false_claims":[],"any_false":false}')
    fc = FactChecker(client).check([_turn(0, "Water boils at 100 C at sea level.")])
    assert fc.available is True and fc.passed is True and fc.false_claims == []


def test_factcheck_reasoning_model_preamble_is_parsed():
    # a reasoning judge emits thinking text before the JSON; take the last object.
    client = ScriptedClient(
        "Let me check each claim...\nThe actor is wrong.\n"
        '{"false_claims":[{"claim":"x","correction":"y"}],"any_false":true}')
    fc = FactChecker(client).check([_turn(0, "some answer")])
    assert fc.passed is False and fc.available is True


def test_factcheck_fails_open_on_error_and_garbage():
    # a failed call or unparseable output must NOT gate (no false reject).
    assert FactChecker(ScriptedClient(raise_exc=True)).check(
        [_turn(0, "a")]).available is False
    res = FactChecker(ScriptedClient("no json here at all")).check([_turn(0, "a")])
    assert res.available is False and res.passed is True


def test_factcheck_empty_answers_pass():
    fc = FactChecker(ScriptedClient("should not be called"))
    res = fc.check([Turn(turn_index=0, segments=[Segment("query", "q")])])
    assert res.passed is True


def test_last_json_object_picks_last():
    assert _last_json_object('{"a":1} ... {"any_false":true}') == {"any_false": True}
    assert _last_json_object("nothing") is None


# ----------------------------------------------------- ReasoningVoiceClassifier
def test_voice_classifier_annotator_and_genuine():
    long = "x" * 60
    assert ReasoningVoiceClassifier(ScriptedClient("ANNOTATOR")).is_annotator(long) is True
    assert ReasoningVoiceClassifier(ScriptedClient("GENUINE")).is_annotator(long) is False


def test_voice_classifier_takes_last_label():
    # a reasoning model may mention both words while thinking; the verdict is last.
    long = "x" * 60
    clf = ReasoningVoiceClassifier(ScriptedClient(
        "It could look ANNOTATOR but it reasons about substance, so GENUINE"))
    assert clf.classify(long) == "GENUINE"


def test_voice_classifier_skips_tiny_and_fails_open():
    clf = ReasoningVoiceClassifier(ScriptedClient("ANNOTATOR"))
    assert clf.classify("short") is None          # below min_chars -> no call
    assert clf.client.calls == 0
    # an error fails open (returns None -> not annotator -> does not gate)
    assert ReasoningVoiceClassifier(ScriptedClient(raise_exc=True)).is_annotator("x" * 60) is False


# --------------------------------------------------------- ReasoningRewriter
def test_rewriter_returns_faithful_rewrite():
    orig = "The user wants five bullets. We need to comply. " + "Owner earnings = 9.8 + 2.5 - 3.0. " * 2
    rw = ReasoningRewriter(ScriptedClient(
        "I work it through: owner earnings = 9.8 + 2.5 - 3.0."))
    out = rw.rewrite(orig)
    assert out is not None and "9.8" in out


def test_rewriter_rejects_invented_figure():
    orig = "The user asks about the survey. We should summarize the headline result."
    # rewrite invents a specific 4-digit figure absent from the original -> reject
    # (guards against fabricated years / citation years / large stats).
    rw = ReasoningRewriter(ScriptedClient(
        "I recall the 2013 survey of 45000 respondents found a clear trend."))
    assert rw.rewrite(orig) is None


def test_rewriter_rejects_balloon():
    orig = "The user wants an overview. Mention 1963 and the TARDIS."
    rw = ReasoningRewriter(ScriptedClient("1963 TARDIS " + "blah " * 200))
    assert rw.rewrite(orig) is None


def test_rewriter_skips_tiny_and_fails_open():
    rw = ReasoningRewriter(ScriptedClient("rewritten"))
    assert rw.rewrite("short") is None and rw.client.calls == 0
    assert ReasoningRewriter(ScriptedClient(raise_exc=True)).rewrite("x" * 60) is None


def test_rewriter_allows_small_derived_numbers():
    # an intermediate sum (12.3) not in the original is fine (it's derived, <4 digits)
    orig = "We need to compute. Net income 9.8 plus D&A 2.5."
    rw = ReasoningRewriter(ScriptedClient("Net income 9.8 plus D&A 2.5 gives 12.3."))
    assert rw.rewrite(orig) is not None


# ----------------------------------------- ReasoningRewriter: synthesis path
_DIRTY = ("The user wants five bullet points and no emojis. We need to comply "
          "with the no-emoji rule. Let's craft the answer covering the trade-offs.")


def test_synthesize_with_answer_anchor_and_genuine_classifier():
    # answer-anchored synthesis verified GENUINE -> returns the clean trace.
    rw = ReasoningRewriter(
        ScriptedClient("I weigh latency against accuracy and land on the batched "
                       "approach since it caps the tail at 1450 ms."),
        voice_classifier=ReasoningVoiceClassifier(ScriptedClient("GENUINE")))
    out = rw.rewrite(_DIRTY, answer="Use batching; tail latency stays under 1450 ms.",
                     user_query="how do I cut latency?")
    assert out is not None and "1450" in out


def test_synthesize_rejected_when_still_annotator():
    # both passes still read ANNOTATOR -> reject (None) so the caller can gate.
    rw = ReasoningRewriter(
        ScriptedClient("Still listing the five bullets and the no-emoji rule."),
        voice_classifier=ReasoningVoiceClassifier(ScriptedClient("ANNOTATOR")))
    assert rw.rewrite(_DIRTY, answer="Use batching.", user_query="q") is None


def test_synthesize_allows_answer_numbers_but_not_invented():
    # a 4-digit figure present in the ANSWER is allowed in the synthesized trace ...
    rw = ReasoningRewriter(ScriptedClient("The 1450 ms ceiling drives the choice."))
    assert rw.rewrite(_DIRTY, answer="Tail latency stays under 1450 ms.") is not None
    # ... but a figure absent from BOTH draft and answer is still rejected.
    rw2 = ReasoningRewriter(ScriptedClient("The 9999 ms ceiling drives the choice."))
    assert rw2.rewrite(_DIRTY, answer="Tail latency stays under 1450 ms.") is None


def test_synthesize_no_classifier_fails_open():
    # without a classifier the synthesis is accepted on the faithfulness guard alone.
    rw = ReasoningRewriter(ScriptedClient("I reason about the actual trade-offs here."))
    assert rw.rewrite(_DIRTY, answer="Use batching for throughput.") is not None


# ------------------------------------------------------------- AnswerRepairer
def _ans_turn(idx, text):
    return Turn(turn_index=idx, segments=[Segment("query", "q"), Segment("answer", text)])


def test_repairer_applies_correction_to_matching_answer():
    turns = [_ans_turn(0, "The breathing pattern is 4 + 7 + 8 = 20 seconds total, very calming.")]
    claims = [{"claim": "4 + 7 + 8 = 20 seconds total",
               "correction": "4 + 7 + 8 = 19 seconds"}]
    rep = AnswerRepairer(ScriptedClient(
        "The breathing pattern is 4 + 7 + 8 = 19 seconds total, very calming."))
    assert rep.repair(turns, claims) is True
    assert "19" in turns[0].segments[1].text


def test_repairer_noop_without_claims():
    turns = [_ans_turn(0, "A clean answer with no errors at all here.")]
    rep = AnswerRepairer(ScriptedClient("should not be called"))
    assert rep.repair(turns, []) is False
    assert rep.client.calls == 0


def test_repairer_skips_unmatched_claim():
    # a claim with no token overlap with any answer is not attributed -> no edit.
    turns = [_ans_turn(0, "Photosynthesis converts light into chemical energy.")]
    claims = [{"claim": "the Roman Empire peaked in 117 CE", "correction": "..."}]
    rep = AnswerRepairer(ScriptedClient("x"))
    assert rep.repair(turns, claims) is False
    assert rep.client.calls == 0


def test_repairer_rejects_runaway_rewrite():
    # a repair that balloons (a sign it rewrote everything) is discarded -> no change.
    turns = [_ans_turn(0, "Short answer 4+7+8=20.")]
    claims = [{"claim": "4+7+8=20", "correction": "4+7+8=19"}]
    rep = AnswerRepairer(ScriptedClient("4+7+8=19 " + "padding " * 100))
    assert rep.repair(turns, claims) is False
    assert "20" in turns[0].segments[1].text  # original kept
