"""Sampler distribution + invalidity-mask + low-yield tests (spec §11.3, §12)."""

import random
import collections

from rxai_sdg.factory.config import FactoryConfig
from rxai_sdg.factory.sampler import IntentPolicySampler
from rxai_sdg.factory.taxonomy import Taxonomy, FACT_INTENTS


def _sampler(seed=0):
    cfg = FactoryConfig()
    tax = cfg.build_taxonomy()
    return IntentPolicySampler(tax, cfg.intent_weights, cfg.policy_weights,
                               rng=random.Random(seed))


def test_mask_never_yields_invalid_pair():
    s = _sampler()
    for _ in range(5000):
        d = s.sample("en")
        assert s.taxonomy.is_valid(d.intent, d.policy)
        # fact intents only immediate/delayed_recall
        if d.intent in FACT_INTENTS:
            assert d.policy in ("immediate", "delayed_recall")
        if d.intent == "chained_compute":
            assert d.policy != "standing"
        if d.intent in ("open_chat", "deepen"):
            assert d.policy == "immediate"
        # delayed_recall is fact-only: a non-fact intent must never draw it
        if d.policy == "delayed_recall":
            assert d.intent in FACT_INTENTS
        # self_critique is a single-turn ask: never standing/cumulative
        if d.intent == "self_critique":
            assert d.policy == "immediate"


def test_policy_distribution_matches_weights_for_transformation_intent():
    # A transformation intent (reformat) is valid with immediate/cumulative/standing
    # but NOT delayed_recall (which is fact-only). The policy marginal is therefore
    # the configured weights RENORMALISED over reformat's valid policies; delayed_recall
    # is never emitted for it.
    cfg = FactoryConfig()
    tax = cfg.build_taxonomy()
    s = IntentPolicySampler(tax, {"reformat": 1.0}, cfg.policy_weights,
                            rng=random.Random(1))
    counts = collections.Counter(s.sample("en").policy for _ in range(40000))
    total = sum(counts.values())
    assert counts["delayed_recall"] == 0  # fact-only policy never drawn for reformat
    valid = tax.valid_policies_for("reformat")
    denom = sum(cfg.policy_weights[p] for p in valid)
    expected = {p: cfg.policy_weights[p] / denom for p in valid}
    for policy, exp in expected.items():
        got = counts[policy] / total
        assert abs(got - exp) < 0.02, f"{policy}: {got:.3f} vs {exp:.3f}"


def test_intent_distribution_roughly_matches_weights():
    s = _sampler(2)
    counts = collections.Counter(s.sample("en").intent for _ in range(60000))
    total = sum(counts.values())
    # transformation intents are valid with all policies, so their relative
    # proportions should track configured weights reasonably closely.
    ra = counts["reformat"] / total
    rb = counts["lexical_constraint"] / total
    # lexical (14) should be sampled more than reformat (12)
    assert rb > ra


def test_lang_valid_intent_set_restricts_sampling():
    s = _sampler()
    s.lang_valid_intents["xx"] = {"reformat", "compress"}
    seen = {s.sample("xx").intent for _ in range(2000)}
    assert seen <= {"reformat", "compress"}


def test_low_yield_downweight():
    s = _sampler()
    # simulate a chronically failing constraint
    for _ in range(10):
        s.record_outcome("reformat", "json_valid", passed=False)
    assert s.pass_rate("reformat", "json_valid") == 0.0
    changed = s.maybe_downweight("reformat", "json_valid", threshold=0.25, min_samples=8)
    assert changed is True
    # penalty reduces effective weight
    ew = s._effective_intent_weights("en")
    assert ew["reformat"] < s.intent_weights["reformat"]


def test_low_yield_not_triggered_below_min_samples():
    s = _sampler()
    for _ in range(3):
        s.record_outcome("reformat", "json_valid", passed=False)
    assert s.maybe_downweight("reformat", "json_valid", 0.25, min_samples=8) is False
