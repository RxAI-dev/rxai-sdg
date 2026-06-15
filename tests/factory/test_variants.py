"""Standalone reasoning -> variant post-processing tests (spec §8)."""

import random

from rxai_sdg.factory.schemas import Seed, Segment, Turn, ConversationRecord
from rxai_sdg.factory.variants import derive_variants


def _record(n=2):
    turns = [
        Turn(i, [Segment("query", f"q{i}"), Segment("reasoning", f"r{i}"),
                 Segment("answer", f"a{i}")])
        for i in range(n)
    ]
    return ConversationRecord(source_seed=Seed("ds", "q0"), turns=turns, mode="reasoning")


def test_derive_all_variants():
    variants = derive_variants(_record(), ["reasoning", "instruct", "mixed"],
                               rng=random.Random(0))
    assert {v.mode for v in variants} == {"reasoning", "instruct", "mixed"}


def test_instruct_strips_all_reasoning():
    instruct = derive_variants(_record(), ["instruct"])[0]
    for turn in instruct.turns:
        assert turn.reasoning is None
        assert turn.reasoning_flag is False
        assert turn.answer is not None  # answer survives


def test_reasoning_variant_keeps_reasoning():
    full = derive_variants(_record(), ["reasoning"])[0]
    assert all(t.reasoning is not None for t in full.turns)


def test_variants_have_distinct_ids():
    variants = derive_variants(_record(), ["reasoning", "instruct"])
    assert variants[0].conversation_id != variants[1].conversation_id


def test_mixed_keeps_subset():
    mixed = derive_variants(_record(20), ["mixed"], rng=random.Random(0),
                            mixed_mode_keep_ratio=0.5)[0]
    kept = sum(1 for t in mixed.turns if t.reasoning is not None)
    assert 3 <= kept <= 17  # roughly half


def test_self_containment_annotation_on_instruct():
    turns = [Turn(0, [Segment("query", "q"), Segment("reasoning", "r"),
                      Segment("answer", "As computed above, it is 42.")])]
    rec = ConversationRecord(source_seed=Seed("ds", "q"), turns=turns)
    instruct = derive_variants(rec, ["instruct"])[0]
    assert "self_containment" in instruct.cross_turn_checks
