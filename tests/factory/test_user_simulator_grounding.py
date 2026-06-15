"""User-Simulator tests for the LLM-driven rewrite (spec §4.2-§4.4, §5.3, §7).

The simulator is now fully LLM-driven: it builds a machine-checkable
``constraint_spec`` programmatically, then drives an (LLM) client - scripted here
with a deterministic Mock that realises the STEER - to produce a natural, grounded
user turn. These tests assert grounding metadata, temporal validity, the
plant→recall→update fact lifecycle (across separate turns), constraint coherence,
the user role, diversity, and that no hardcoded query strings remain.
"""

import random

from rxai_sdg.factory.clients import MockLLMClient
from rxai_sdg.factory.config import FactoryConfig
from rxai_sdg.factory.ledger import FactLedger, NeedlePlanner
from rxai_sdg.factory.prompts import get_prompt_pack
from rxai_sdg.factory.sampler import IntentPolicySampler, SamplerDraw
from rxai_sdg.factory.schemas import ConstraintSpec, Segment, Turn
from rxai_sdg.factory.testing import simulator_user_turn_handler
from rxai_sdg.factory.user_simulator import (
    UserSimulator, GROUNDING_KINDS, PERSONAS, VERBOSITIES,
)


def _client():
    return MockLLMClient(handler=simulator_user_turn_handler)


def _sim(seed=0, client=None, min_recall_distance=4):
    cfg = FactoryConfig()
    tax = cfg.build_taxonomy()
    rng = random.Random(seed)
    sampler = IntentPolicySampler(tax, cfg.intent_weights, cfg.policy_weights, rng=rng)
    ledger = FactLedger()
    planner = NeedlePlanner(ledger, rng=rng, min_distance=min_recall_distance)
    sim = UserSimulator(sampler=sampler, planner=planner,
                        client=client or _client(), rng=rng,
                        min_recall_distance=min_recall_distance)
    return sim, planner, ledger


def _seed_turns(query="Explain how entropy relates to information.",
                answer="Entropy measures uncertainty."):
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


def test_transformation_requests_exact_constraint():
    sim, _, _ = _sim(3)
    prior = _seed_turns()
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                         active_constraints=[],
                         forced_draw=SamplerDraw("reformat", "immediate"))
    assert res.grounding == "transforms_prior"
    fmt_markers = {"json": "json", "yaml": "yaml", "table": "table", "md": "markdown"}
    marker = fmt_markers[res.constraint_spec.params["format"]]
    assert marker in res.nl_query.lower()  # query coherently requests the constraint


def test_lexical_first_letter_query_names_the_letter():
    sim, _, _ = _sim(0, client=_client())
    prior = _seed_turns()
    # draw lexical immediate until a first_letter constraint is produced
    for i in range(1, 80):
        res = sim.next_query(prior, get_prompt_pack("general"), turn_index=i,
                             active_constraints=[],
                             forced_draw=SamplerDraw("lexical_constraint", "immediate"))
        if res.constraint_spec.type == "first_letter":
            letter = res.constraint_spec.params["letter"]
            assert f"'{letter}'" in res.nl_query
            return
    raise AssertionError("never drew a first_letter constraint")


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
    assert res2.constraint_spec.params["value"] == ledger.latest(f.fact_id)
    # the recall must NOT restate the value, and must name the fact
    assert str(ledger.latest(f.fact_id)).lower() not in res2.nl_query.lower()
    assert "favorite" in res2.nl_query.lower()


# ---- fact lifecycle (across separate turns) --------------------------------

def test_planted_fact_string_appears_in_planting_turn_and_is_injected():
    sim, planner, ledger = _sim(7)
    prior = _seed_turns()
    res = sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                         active_constraints=[],
                         forced_draw=SamplerDraw("fact_recall", "immediate"))
    cs = res.constraint_spec
    assert cs.fact_id is not None
    assert res.grounding == "plants_fact"
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


def test_plant_and_recall_are_separate_turns():
    sim, planner, ledger = _sim(11)
    prior = _seed_turns()
    # turn 2: plant
    plant = sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                           active_constraints=[],
                           forced_draw=SamplerDraw("fact_recall", "immediate"))
    fid = plant.constraint_spec.fact_id
    value = str(plant.constraint_spec.params["value"])
    assert value.lower() in plant.nl_query.lower()
    # turn 8 (>= min_recall_distance later): recall the SAME fact, value absent
    recall = sim.next_query(prior, get_prompt_pack("general"), turn_index=8,
                            active_constraints=[],
                            forced_draw=SamplerDraw("fact_recall", "delayed_recall"))
    assert recall.grounding == "recalls_fact"
    assert recall.constraint_spec.fact_id == fid
    assert value.lower() not in recall.nl_query.lower()  # never restates the value


def test_update_value_appears_in_query_and_marks_injected():
    sim, planner, ledger = _sim(8)
    prior = _seed_turns()
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


# ---- full transcript / no token cap / role / diversity (Part 3) ------------

def test_simulator_prompt_contains_full_transcript_not_just_last():
    client = _client()
    sim, _, _ = _sim(2, client=client)
    prior = [
        Turn(0, [Segment("query", "Explain how entropy relates to information."),
                 Segment("answer", "Entropy measures uncertainty in a distribution.")]),
        Turn(1, [Segment("query", "Give a concrete example with coins."),
                 Segment("answer", "A fair coin has one bit of entropy per flip.")]),
    ]
    sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                   active_constraints=[],
                   forced_draw=SamplerDraw("reformat", "immediate"))
    prompt = client.calls[-1]["prompt"]
    # content from an EARLIER turn (not just the last) is present
    assert "Entropy measures uncertainty" in prompt
    assert "Explain how entropy" in prompt


def test_no_max_tokens_cap_forces_short_queries():
    client = _client()
    sim, _, _ = _sim(2, client=client)
    prior = _seed_turns()
    sim.next_query(prior, get_prompt_pack("general"), turn_index=2,
                   active_constraints=[],
                   forced_draw=SamplerDraw("reformat", "immediate"))
    # the simulator must NOT pass a (small) max_tokens that biases toward short turns
    assert "max_tokens" not in client.calls[-1]["kwargs"]


def test_persona_and_length_diversity_across_a_batch():
    sim, _, _ = _sim(3)
    prior = _seed_turns()
    personas, verbosities, lengths = set(), set(), set()
    for i in range(1, 40):
        res = sim.next_query(prior, get_prompt_pack("general"), turn_index=i,
                             active_constraints=[])
        personas.add(res.persona)
        verbosities.add(res.verbosity)
        lengths.add(len(res.nl_query))
    assert personas <= set(PERSONAS) and len(personas) >= 3
    assert verbosities <= set(VERBOSITIES) and len(verbosities) >= 2
    assert len(lengths) >= 3  # query lengths genuinely vary (not all identical/short)


_ROLE_RED_FLAGS = [
    "provide your next question", "pose a question", "as the assistant",
    "i'll answer", "here is my answer", "assistant:",
]


def test_user_role_is_never_inverted():
    sim, _, _ = _sim(4)
    prior = _seed_turns()
    for i in range(1, 60):
        res = sim.next_query(prior, get_prompt_pack("general"), turn_index=i,
                             active_constraints=[])
        low = res.nl_query.lower()
        for flag in _ROLE_RED_FLAGS:
            assert flag not in low, f"role inversion ({flag!r}): {res.nl_query!r}"


# ---- no hardcoded production query strings remain (grep) -------------------

_OLD_TEMPLATES = [
    "Building on that, what's the next logical step",
    "Critique your previous answer: what could be wrong",
    "Go deeper on one key point from your last answer",
    "Honestly, that's a lot to take in",
    "Let's switch gears",
    "role-play explaining this to a curious ten-year-old",
    "Sticking with",
    "On the subject of",
    "By the way, my",
    "Building on your last answer about",
]


def test_no_hardcoded_query_strings_in_simulator_or_constraints():
    import pathlib
    import rxai_sdg.factory.constraints as cmod
    import rxai_sdg.factory.user_simulator as smod

    for mod in (cmod, smod):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        for template in _OLD_TEMPLATES:
            assert template not in src, f"hardcoded query string remained: {template!r}"
    # constraints.py no longer emits user-facing query text at all
    csrc = pathlib.Path(cmod.__file__).read_text(encoding="utf-8")
    assert "nl_query" not in csrc
