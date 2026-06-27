"""Constraint-builder tests (spec §4.4)."""

import random

from rxai_sdg.factory import constraints as C
from rxai_sdg.factory.ledger import FactLedger, NeedlePlanner
from rxai_sdg.factory.taxonomy import POLICY_TO_SCOPE


def _ctx(intent, policy, turn=2, seed=0, planter=None):
    led = FactLedger()
    planner = planter or NeedlePlanner(led, rng=random.Random(seed), min_distance=4)
    return C.BuildContext(
        rng=random.Random(seed), intent=intent, policy=policy, turn=turn,
        lang="en", planner=planner, min_recall_distance=4)


def test_reformat_scope_matches_policy():
    for policy in ("immediate", "cumulative", "standing"):
        res = C.build(_ctx("reformat", policy))
        assert res.constraint_spec.scope == POLICY_TO_SCOPE[policy]
        assert res.constraint_spec.verifier == "programmatic"


def test_standing_sets_applies_from_turn():
    res = C.build(_ctx("reformat", "standing", turn=3))
    assert res.constraint_spec.applies_from_turn == 3


def test_lexical_standing_uses_compose_friendly_only():
    # over many draws, standing lexical never selects positional sub-types or the
    # word-count sub-type (max_words_per_sentence as a standing rule forces per-turn
    # word-counting bookkeeping, so it is current-turn only now).
    seen = set()
    for s in range(200):
        res = C.build(_ctx("lexical_constraint", "standing", seed=s))
        seen.add(res.constraint_spec.type)
    assert "first_letter" not in seen
    assert "alphabetical_sentence_starts" not in seen
    assert "max_words_per_sentence" not in seen
    assert seen <= {"forbidden_token", "no_gendered_pronouns"}


def test_lexical_immediate_can_use_first_letter():
    seen = {C.build(_ctx("lexical_constraint", "immediate", seed=s)).constraint_spec.type
            for s in range(200)}
    assert "first_letter" in seen
    # the word-count sub-type is still represented, just only as a current-turn rewrite
    assert "max_words_per_sentence" in seen


def test_compress_downsamples_word_budget_but_keeps_it():
    # length_tokens (word budget) must remain represented but be a clear minority,
    # even for a standing scope (previously it was used on 100% of standing compress).
    types = [C.build(_ctx("compress", "standing", seed=s)).constraint_spec.type
             for s in range(400)]
    n_len = types.count("length_tokens")
    assert 0 < n_len < len(types) * 0.5          # present, but not dominant
    assert types.count("n_bullets") > n_len      # bullet budget is the majority


def test_fact_recall_delayed_resamples_when_no_fact():
    res = C.build(_ctx("fact_recall", "delayed_recall", turn=5))
    assert res.resample is True
    assert res.constraint_spec is None


def test_fact_recall_delayed_uses_planted_fact():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=4)
    f = planner.plant_fact(turn=1)
    led.mark_injected(f.fact_id)  # a recall only targets an *injected* fact (fix C)
    ctx = C.BuildContext(rng=random.Random(0), intent="fact_recall",
                         policy="delayed_recall", turn=6, lang="en", planner=planner,
                         min_recall_distance=4)
    res = C.build(ctx)
    assert res.resample is False
    assert res.constraint_spec.type == "fact_recall"
    assert res.constraint_spec.fact_id is not None
    assert res.constraint_spec.planted_turn == 1


def test_fact_recall_delayed_resamples_without_injected_fact():
    # a fact that was planted but never injected cannot be recalled (no desync).
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=4)
    planner.plant_fact(turn=1)  # NOT marked injected
    ctx = C.BuildContext(rng=random.Random(0), intent="fact_recall",
                         policy="delayed_recall", turn=6, lang="en", planner=planner,
                         min_recall_distance=4)
    assert C.build(ctx).resample is True


def test_fact_recall_immediate_plants_only():
    # immediate fact_recall is now a PLANT (no same-turn recall): it registers a
    # fact and carries an llm_judge spec (the value is woven into the query by the
    # simulator, not gated on the answer). constraints.py emits no query text.
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=4)
    ctx = C.BuildContext(rng=random.Random(0), intent="fact_recall",
                         policy="immediate", turn=2, lang="en", planner=planner,
                         min_recall_distance=4)
    res = C.build(ctx)
    assert res.resample is False
    assert res.constraint_spec.fact_id is not None
    assert res.constraint_spec.verifier == "llm_judge"  # plant turn isn't answer-gated
    assert "value" in res.constraint_spec.params
    assert len(led) == 1  # exactly one fact planted


def test_fact_update_records_only_injected_stale_values():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(1), min_distance=4)
    f = planner.plant_fact(turn=1, fact_type="favorite_color")
    led.mark_injected(f.fact_id)  # the value actually appeared in the text
    original = f.value
    ctx = C.BuildContext(rng=random.Random(1), intent="fact_update", policy="immediate",
                         turn=3, lang="en", planner=planner, min_recall_distance=4)
    res = C.build(ctx)
    spec = res.constraint_spec
    assert spec.type == "fact_update"
    # the single stale value is the injected pre-update value (no phantom staleness)
    assert spec.params["stale_values"] == [original]
    # the new value is *peeked*, not committed: the ledger is overwritten by the
    # simulator only once the update turn passes (no phantom on a failed update).
    assert spec.params["value"] != original
    assert led.latest(spec.fact_id) == original  # not yet committed


def test_fact_update_resamples_without_injected_fact():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(1), min_distance=4)
    planner.plant_fact(turn=1, fact_type="favorite_color")  # NOT injected
    ctx = C.BuildContext(rng=random.Random(1), intent="fact_update", policy="immediate",
                         turn=3, lang="en", planner=planner, min_recall_distance=4)
    assert C.build(ctx).resample is True


def test_non_verifiable_intents_use_llm_judge():
    for intent in ("expand", "chained_compute", "self_critique", "deepen", "open_chat"):
        res = C.build(_ctx(intent, "immediate"))
        assert res.constraint_spec.verifier == "llm_judge"
