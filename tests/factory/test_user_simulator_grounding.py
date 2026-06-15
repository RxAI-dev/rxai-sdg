"""User-Simulator grounding / temporal-validity / fact-lifecycle tests (spec §4.2-§4.4, §7)."""

import random

from rxai_sdg.factory.config import FactoryConfig
from rxai_sdg.factory.ledger import FactLedger, NeedlePlanner
from rxai_sdg.factory.prompts import get_prompt_pack
from rxai_sdg.factory.sampler import IntentPolicySampler, SamplerDraw
from rxai_sdg.factory.schemas import ConstraintSpec, Segment, Turn
from rxai_sdg.factory.user_simulator import UserSimulator, GROUNDING_KINDS


def _sim(seed=0, client=None, min_recall_distance=4):
    cfg = FactoryConfig()
    tax = cfg.build_taxonomy()
    rng = random.Random(seed)
    sampler = IntentPolicySampler(tax, cfg.intent_weights, cfg.policy_weights, rng=rng)
    ledger = FactLedger()
    planner = NeedlePlanner(ledger, rng=rng, min_distance=min_recall_distance)
    sim = UserSimulator(sampler=sampler, planner=planner, client=client, rng=rng,
                        min_recall_distance=min_recall_distance)
    return sim, planner, ledger


def _seed_turns(query="Explain how entropy relates to information.", answer="Entropy measures uncertainty."):
    return [Turn(0, [Segment("query", query), Segment("answer", answer)])]


# ---- grounding -------------------------------------------------------------

def test_every_query_is_grounded_metadata():
    sim, _, _ = _sim(1)
    prior = _seed_turns()
    seen = set()
    for i in range(1, 60):
        res = sim.next_query(prior, get_prompt_pack("general"), turn_index=i,
                             active_constraints=[])
        assert res.grounding in GROUNDING_KINDS
        assert res.nl_query.strip()
        seen.add(res.grounding)
    # the deterministic templates exercise more than one grounding kind
    assert len(seen) >= 2


def test_open_chat_continues_running_topic():
    sim, _, _ = _sim(2)
    prior = _seed_turns()
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=3,
                         active_constraints=[],
                         forced_draw=SamplerDraw("open_chat", "immediate"))
    assert res.grounding == "continues_topic"
    head = res.topic.split()[0].lower()
    assert head and head in res.nl_query.lower()  # references the running topic


def test_transformation_targets_prior_answer():
    sim, _, _ = _sim(3)
    prior = _seed_turns()
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                         active_constraints=[],
                         forced_draw=SamplerDraw("reformat", "immediate"))
    assert res.grounding == "transforms_prior"
    assert "answer" in res.nl_query.lower()  # explicitly operates on the prior answer


# ---- temporal-policy validity ---------------------------------------------

def test_cumulative_never_without_prior_active_constraint():
    sim, _, _ = _sim(4)
    prior = _seed_turns()
    for i in range(1, 200):
        res = sim.next_query(prior, get_prompt_pack("general"), turn_index=i,
                             active_constraints=[])
        if res.constraint_spec is not None:
            assert res.constraint_spec.scope != "cumulative"


def test_cumulative_allowed_once_a_constraint_is_active():
    sim, _, _ = _sim(5)
    prior = _seed_turns()
    active = [ConstraintSpec(intent="reformat", type="json_valid", scope="standing",
                             applies_from_turn=1, verifier="programmatic")]
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=6,
                         active_constraints=active,
                         forced_draw=SamplerDraw("compress", "cumulative"))
    assert res.constraint_spec.scope == "cumulative"


def test_delayed_recall_fact_requires_distant_injected_fact():
    sim, planner, ledger = _sim(6)
    prior = _seed_turns()
    # No fact yet -> a forced delayed fact_recall must resample to something else.
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=7,
                         active_constraints=[],
                         forced_draw=SamplerDraw("fact_recall", "delayed_recall"))
    assert not (res.draw.intent == "fact_recall" and res.draw.policy == "delayed_recall")

    # Plant + inject a fact far enough back; now a delayed recall is valid.
    f = planner.plant_fact(turn=1, fact_type="favorite_color")
    ledger.mark_injected(f.fact_id)
    res2 = sim.next_query(prior, get_prompt_pack("general"), turn_index=7,
                          active_constraints=[],
                          forced_draw=SamplerDraw("fact_recall", "delayed_recall"))
    assert res2.draw == SamplerDraw("fact_recall", "delayed_recall")
    assert res2.grounding == "recalls_fact"
    # recall query value matches the ledger value
    assert res2.constraint_spec.params["value"] == ledger.latest(f.fact_id)


# ---- fact lifecycle --------------------------------------------------------

def test_planted_fact_string_appears_in_planting_turn_and_is_injected():
    sim, planner, ledger = _sim(7)
    prior = _seed_turns()
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                         active_constraints=[],
                         forced_draw=SamplerDraw("fact_recall", "immediate"))
    cs = res.constraint_spec
    assert cs.fact_id is not None
    value = str(cs.params["value"])
    assert value.lower() in res.nl_query.lower()         # exact string woven in
    assert ledger.is_injected(cs.fact_id) is True        # marked injected


def test_recall_only_scheduled_against_injected_fact():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=4)
    f = planner.plant_fact(turn=1, fact_type="hometown")
    # un-injected -> not recallable under the injection-aware filter
    assert planner.recallable_fact(turn=7, min_distance=4, require_injected=True) is None
    led.mark_injected(f.fact_id)
    got = planner.recallable_fact(turn=7, min_distance=4, require_injected=True)
    assert got is not None and got.fact_id == f.fact_id


def test_update_value_appears_in_query_and_marks_injected():
    sim, planner, ledger = _sim(8)
    prior = _seed_turns()
    # plant + inject a fact first
    f = planner.plant_fact(turn=1, fact_type="favorite_color")
    ledger.mark_injected(f.fact_id)
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=3,
                         active_constraints=[],
                         forced_draw=SamplerDraw("fact_update", "immediate"))
    cs = res.constraint_spec
    assert cs.intent == "fact_update"
    new_value = str(cs.params["value"])
    assert new_value.lower() in res.nl_query.lower()   # the NEW value is stated
    assert ledger.latest(cs.fact_id) == cs.params["value"]
