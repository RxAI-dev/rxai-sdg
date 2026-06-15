"""End-to-end integration test with a deterministic mocked client (spec §12).

Asserts schema validity, ledger correctness, that exactly one reasoning-mode
record is produced per seed, and that the sampled (intent, policy) distribution
is covered. No network is used.
"""

import random

from rxai_sdg.factory import (
    DataFactory, FactoryConfig, MockLLMClient, validate_record,
)
from rxai_sdg.factory.testing import constraint_satisfying_handler


SEEDS = [
    "Explain how entropy relates to information.",
    "Write a short paragraph about lighthouses.",
    "What is 17 * 23 and why does the method work?",
    "Outline a function to reverse a linked list.",
]


def _factory(seed=0):
    cfg = FactoryConfig(seed=seed)
    client = MockLLMClient(handler=constraint_satisfying_handler)
    return DataFactory(cfg, client, rng=random.Random(seed))


def test_one_record_per_seed_and_records_validate():
    factory = _factory(0)
    records = factory.generate(SEEDS, band="basic")
    assert len(records) == len(SEEDS)            # one conversation per seed
    assert all(r.mode == "reasoning" for r in records)
    assert factory.records == records            # collected for later saving
    for rec in records:
        validate_record(rec.to_dict())


def test_accepts_string_and_dict_seeds():
    factory = _factory(2)
    records = factory.generate(
        ["Explain entropy.", {"query": "Reverse a linked list."}], band="basic")
    assert len(records) == 2
    for rec in records:
        validate_record(rec.to_dict())


def test_records_are_self_contained_reasoning():
    factory = _factory(2)
    records = factory.generate(SEEDS, band="basic")
    for rec in records:
        for turn in rec.turns:
            assert turn.answer is not None
            assert turn.query is not None


def test_ledger_facts_have_history():
    factory = _factory(3)
    records = factory.generate(SEEDS, band="basic")
    saw_fact = False
    for rec in records:
        for fact in rec.fact_ledger:
            saw_fact = True
            assert fact.value_history, "fact must carry value history"
            assert fact.value_history[-1]["value"] == fact.value
    assert saw_fact, "expected at least one planted fact across conversations"


def test_cross_turn_checks_present_and_recalls_mostly_pass():
    factory = _factory(4)
    records = factory.generate(SEEDS, band="basic")
    total = passed = 0
    for rec in records:
        for entry in rec.cross_turn_checks.get("delayed_recall", []):
            total += 1
            passed += 1 if entry["passed"] else 0
    if total:
        assert passed / total >= 0.6


def test_intent_and_policy_coverage():
    factory = _factory(5)
    records = factory.generate(SEEDS, band="generalization")
    intents, policies = set(), set()
    for rec in records:
        for turn in rec.turns[1:]:  # skip seed turn
            if turn.intent:
                intents.add(turn.intent)
            if turn.policy:
                policies.add(turn.policy)
    assert {"immediate", "cumulative", "standing", "delayed_recall"} <= policies
    assert len(intents) >= 6


def test_no_discard_when_first_answer_ok():
    factory = _factory(6)
    factory.generate(SEEDS, band="basic")
    assert factory.stats.conversations_discarded == 0
    assert factory.stats.conversations_built == len(SEEDS)


def test_first_answer_unsalvageable_discards_conversation():
    cfg = FactoryConfig(seed=0, regeneration_limit=1)
    client = MockLLMClient(default="I can't help with that.")
    factory = DataFactory(cfg, client, rng=random.Random(0))
    records = factory.generate(["Explain entropy."])
    assert records == []
    assert factory.stats.conversations_discarded == 1


def test_short_band_and_jsonl(tmp_path):
    factory = _factory(7)
    records = factory.generate(SEEDS, band="short")
    for rec in records:
        assert 2 <= rec.length <= 3
    out = tmp_path / "out.jsonl"
    n = factory.write_jsonl(str(out))
    assert n == len(records)
