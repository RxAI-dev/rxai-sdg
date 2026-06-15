"""Multithreaded-generation reproducibility + stats-consistency tests (spec §6, §7)."""

import random

from rxai_sdg.factory import DataFactory, FactoryConfig, MockLLMClient
from rxai_sdg.factory.testing import constraint_satisfying_handler

SEEDS = [
    "Explain how entropy relates to information.",
    "Write a short paragraph about lighthouses.",
    "What is 17 * 23 and why does the method work?",
    "Outline a function to reverse a linked list.",
    "Describe how a hash map works.",
    "Summarise the water cycle.",
]


def _generate(concurrency, seed=11, band="generalization"):
    cfg = FactoryConfig(seed=seed, concurrency=concurrency)
    # a pure-function handler is thread-safe and deterministic
    client = MockLLMClient(handler=constraint_satisfying_handler)
    factory = DataFactory(cfg, client, rng=random.Random(seed))
    records = factory.generate(SEEDS, band=band)
    return records, factory


def _comparable(records):
    out = []
    for rec in records:
        d = rec.to_dict()
        d.pop("conversation_id")  # the only intentionally random field
        out.append(d)
    return out


def test_parallel_matches_serial_records():
    serial, _ = _generate(concurrency=1)
    parallel, _ = _generate(concurrency=32)
    assert len(serial) == len(parallel) == len(SEEDS)
    assert _comparable(serial) == _comparable(parallel)


def test_parallel_preserves_seed_order():
    parallel, _ = _generate(concurrency=32)
    first_queries = [r.source_seed.first_query for r in parallel]
    assert first_queries == SEEDS  # results returned in seed order


def test_stats_totals_consistent_across_concurrency():
    _, fs = _generate(concurrency=1)
    _, fp = _generate(concurrency=16)
    assert fs.stats.loop == fp.stats.loop
    assert fs.stats.conversations_built == fp.stats.conversations_built
    assert fs.stats.records_emitted == fp.stats.records_emitted


def test_high_concurrency_does_not_corrupt_records():
    records, factory = _generate(concurrency=64)
    from rxai_sdg.factory import validate_record
    assert len(records) == len(SEEDS)
    for rec in records:
        validate_record(rec.to_dict())
