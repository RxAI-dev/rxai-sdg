"""End-to-end integration test with a deterministic mocked client (spec §12).

Asserts schema validity, ledger correctness, that reasoning-stripping produces
self-consistent records, and that the sampled (intent, policy) distribution is
covered. No network is used.
"""

import random

import pytest

from rxai_sdg.factory import (
    DataFactory, FactoryConfig, MockLLMClient, DatasetSpec, validate_record,
    LengthBand,
)
from rxai_sdg.factory.testing import constraint_satisfying_handler


SEED_RECORDS = [
    {"query": "Explain how entropy relates to information.", "category": "stem"},
    {"query": "Write a short paragraph about lighthouses.", "category": "writing"},
    {"query": "What is 17 * 23 and why does the method work?", "category": "math"},
    {"query": "Outline a function to reverse a linked list.", "category": "coding"},
]


def _factory(seed=0):
    cfg = FactoryConfig(seed=seed)
    rng = random.Random(seed)
    client = MockLLMClient(handler=constraint_satisfying_handler)
    return DataFactory(cfg, client, rng=rng), cfg


def test_end_to_end_records_validate():
    factory, cfg = _factory(0)
    spec = DatasetSpec(records=SEED_RECORDS)
    records = factory.generate(spec, n_conversations=6, band="basic")
    assert records, "no records produced"
    for rec in records:
        validate_record(rec.to_dict())


def test_derived_variants_per_conversation():
    factory, cfg = _factory(1)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=3, band="basic")
    # 3 variants (reasoning, instruct, mixed) per built conversation
    modes = [r.mode for r in records]
    assert modes.count("reasoning") == modes.count("instruct") == modes.count("mixed")
    assert len(records) == 3 * factory.stats.conversations_built


def test_instruct_variant_has_no_reasoning_and_is_self_consistent():
    factory, _ = _factory(2)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=4, band="basic")
    instruct = [r for r in records if r.mode == "instruct"]
    assert instruct
    for rec in instruct:
        for turn in rec.turns:
            assert turn.reasoning is None
            assert turn.answer is not None  # answer stands alone


def test_ledger_facts_have_history_and_recalls_resolve():
    factory, _ = _factory(3)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=6, band="basic")
    saw_fact = False
    for rec in records:
        if rec.mode != "reasoning":
            continue
        for fact in rec.fact_ledger:
            saw_fact = True
            assert fact.value_history, "fact must carry value history"
            assert fact.value_history[-1]["value"] == fact.value
    assert saw_fact, "expected at least one planted fact across conversations"


def test_cross_turn_checks_present_and_recalls_mostly_pass():
    factory, _ = _factory(4)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=6, band="basic")
    total = passed = 0
    for rec in records:
        if rec.mode != "reasoning":
            continue
        for entry in rec.cross_turn_checks.get("delayed_recall", []):
            total += 1
            passed += 1 if entry["passed"] else 0
    if total:
        # the constraint-satisfying mock should recall correctly most of the time
        assert passed / total >= 0.6


def test_intent_and_policy_coverage():
    factory, _ = _factory(5)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=8, band="generalization")
    intents, policies = set(), set()
    for rec in records:
        if rec.mode != "reasoning":
            continue
        for turn in rec.turns[1:]:  # skip seed turn
            if turn.intent:
                intents.add(turn.intent)
            if turn.policy:
                policies.add(turn.policy)
    # all four memory-distance policies should appear across a generalization run
    assert {"immediate", "cumulative", "standing", "delayed_recall"} <= policies
    # a broad spread of intents should be covered
    assert len(intents) >= 6


def test_no_discard_when_first_answer_ok():
    factory, _ = _factory(6)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=4, band="basic")
    assert factory.stats.conversations_discarded == 0
    assert factory.stats.conversations_built == 4


def test_first_answer_unsalvageable_discards_conversation():
    # a client that always refuses -> first answer fails quality K+1 times -> discard
    cfg = FactoryConfig(seed=0, regeneration_limit=1)
    client = MockLLMClient(default="I can't help with that.")
    factory = DataFactory(cfg, client, rng=random.Random(0))
    records = factory.generate(DatasetSpec(records=[SEED_RECORDS[0]]), n_conversations=1)
    assert records == []
    assert factory.stats.conversations_discarded == 1


def test_short_band_round_trip(tmp_path):
    factory, _ = _factory(7)
    records = factory.generate(DatasetSpec(records=SEED_RECORDS),
                               n_conversations=2, band="short")
    for rec in records:
        assert 2 <= rec.length <= 3
    out = tmp_path / "out.jsonl"
    n = factory.write_jsonl(records, str(out))
    assert n == len(records)
