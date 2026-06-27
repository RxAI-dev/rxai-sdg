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
from rxai_sdg.factory.factuality import FactChecker, _last_json_object  # noqa: E402
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
