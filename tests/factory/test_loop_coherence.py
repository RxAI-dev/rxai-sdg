"""Conversation-loop budget + coherence-gate tests (spec §4.4, §7)."""

import random

from rxai_sdg.factory.clients import LLMResponse, MockLLMClient
from rxai_sdg.factory.config import FactoryConfig
from rxai_sdg.factory.ledger import FactLedger, NeedlePlanner
from rxai_sdg.factory.loop import ConversationLoop
from rxai_sdg.factory.prompts import get_prompt_pack
from rxai_sdg.factory.responder import Responder
from rxai_sdg.factory.sampler import IntentPolicySampler
from rxai_sdg.factory.schemas import ConstraintSpec, Segment, Turn
from rxai_sdg.factory.testing import simulator_user_turn_handler
from rxai_sdg.factory.user_simulator import UserSimulator


class _CountingClient:
    """Counts calls; always returns a well-formed but constraint-violating answer."""

    def __init__(self, text):
        self.calls = 0
        self.text = text

    def generate(self, prompt, *, system_prompt="", temperature=0.7,
                 max_tokens=4096, capture_logits=False, **kw):
        self.calls += 1
        return LLMResponse(text=self.text)


def _loop(config, client):
    cfg = config
    tax = cfg.build_taxonomy()
    rng = random.Random(0)
    # restrict to a programmatic intent so every follow-up carries a checkable
    # constraint that the fixed plain-prose answer always fails.
    sampler = IntentPolicySampler(tax, {"reformat": 1.0}, cfg.policy_weights, rng=rng)
    responder = Responder(client)
    return ConversationLoop(responder, sampler, cfg, rng=rng), rng


def test_per_turn_responder_budget_is_bounded():
    cap = 8
    cfg = FactoryConfig(seed=0, max_responder_calls_per_turn=cap, regeneration_limit=4)
    # a refusal fails the quality gate for *every* intent (incl. the lexical
    # fallback), so no attempt can pass and the whole budget is consumed.
    client = _CountingClient("<think>x</think> I can't help with that request.")
    loop, rng = _loop(cfg, client)

    sim = UserSimulator(
        sampler=loop.sampler,
        planner=NeedlePlanner(FactLedger(), rng=rng, min_distance=4),
        client=MockLLMClient(handler=simulator_user_turn_handler), rng=rng)
    prior = [Turn(0, [Segment("query", "Explain entropy."),
                      Segment("answer", "Entropy is disorder.")])]

    client.calls = 0
    turn = loop._followup_turn(sim, get_prompt_pack("general"), prior, 1,
                               active_constraints=[], stats=loop.stats.__class__())
    # all attempts fail the json/markdown constraint -> budget fully consumed,
    # but never exceeded.
    assert client.calls <= cap
    assert client.calls == cap  # worst case: exactly the cap
    assert turn.verification.passed is False


def test_coherence_gate_marks_disclaimer_turn_failed():
    # a turn that satisfies the literal constraint but disclaims memory must NOT
    # be marked passed.
    cfg = FactoryConfig(seed=0)
    tax = cfg.build_taxonomy()
    loop = ConversationLoop(Responder(MockLLMClient()),
                            IntentPolicySampler(tax, rng=random.Random(0)), cfg)

    cs = ConstraintSpec(intent="reformat", type="json_valid", verifier="programmatic")
    good = Turn(1, [Segment("query", "as JSON"), Segment("answer", '{"a": 1}')],
                constraint_spec=cs)
    passed, _ = loop._verify_turn(good, cs, [])
    assert passed is True

    disclaim = Turn(1, [Segment("query", "as JSON"),
                        Segment("answer", '{"note": "I cannot store personal '
                                          'information between conversations"}')],
                    constraint_spec=cs)
    passed2, detail2 = loop._verify_turn(disclaim, cs, [])
    assert passed2 is False
    assert "coherence" in detail2


def test_coherence_gate_rejects_cot_leak_even_if_constraint_holds():
    cfg = FactoryConfig(seed=0)
    tax = cfg.build_taxonomy()
    loop = ConversationLoop(Responder(MockLLMClient()),
                            IntentPolicySampler(tax, rng=random.Random(0)), cfg)
    cs = ConstraintSpec(intent="reformat", type="markdown_format", verifier="programmatic")
    leak = Turn(1, [Segment("query", "markdown"),
                    Segment("answer", "# Heading\n\n- bullet\n\nDraft 1: let's verify this.")],
                constraint_spec=cs)
    passed, detail = loop._verify_turn(leak, cs, [])
    assert passed is False
    assert "coherence" in detail


# --------------------------------------------------------------- generation fixes
def test_has_fabricated_experience_detector():
    from rxai_sdg.factory.loop import has_fabricated_experience as f
    assert f("The tallest one, which a friend of mine owns, is 10cm.")
    assert f("In my experience, a short walk helps.")
    assert f("When I tested it, the buffer flushed immediately.")
    assert f("I keep one in my apartment for testing.")
    # legitimate first-person must NOT trip
    assert not f("I recommend a short walk.")
    assert not f("I think the answer is 4.")
    assert not f("I've outlined the steps below.")


def test_standing_semantic_obligation_rendered_for_responder():
    # a standing style/genre obligation must be re-surfaced to the responder (so it
    # keeps honouring the tone/form instead of drifting), in natural USER voice with
    # NO schema vocabulary.
    style = ConstraintSpec(intent="restyle", type="style",
                           params={"style": "the persona of a pirate"}, lang="en",
                           verifier="llm_judge", scope="standing", applies_from_turn=2)
    genre = ConstraintSpec(intent="genre_convert", type="genre",
                           params={"genre": "limerick"}, lang="en",
                           verifier="llm_judge", scope="standing", applies_from_turn=3)
    note = ConversationLoop._active_constraints_note([style, genre])
    assert "pirate" in note and "limerick" in note
    assert "standing" not in note.lower() and "llm_judge" not in note and "style" not in note


def test_numeric_defect_regenerates_bad_arithmetic_turn():
    from rxai_sdg.factory.loop import _numeric_defect
    from rxai_sdg.factory.schemas import Segment, Turn
    bad = Turn(turn_index=1, segments=[
        Segment("query", "compute it"),
        Segment("reasoning", "clean reasoning"),
        Segment("answer", "So 192/1728 = 0.44 cubic feet, hence ~1 board.")])
    assert _numeric_defect(bad) is not None
    good = Turn(turn_index=1, segments=[
        Segment("query", "compute it"),
        Segment("reasoning", "clean reasoning"),
        Segment("answer", "So 192/1728 = 0.111 cubic feet.")])
    assert _numeric_defect(good) is None
