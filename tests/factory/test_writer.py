"""SegmentWriter / derived-variant tests (spec §8, §11.9)."""

import json
import random

from rxai_sdg.factory.schemas import (
    Seed, Segment, Turn, ConversationRecord,
)
from rxai_sdg.factory.writer import SegmentWriter, flag_dangling_references


def _record():
    turns = [
        Turn(0, [Segment("query", "q0"), Segment("reasoning", "r0"), Segment("answer", "a0")]),
        Turn(1, [Segment("query", "q1"), Segment("reasoning", "r1"), Segment("answer", "a1")]),
    ]
    return ConversationRecord(source_seed=Seed("ds", "q0"), turns=turns, mode="reasoning")


def test_derive_all_variants():
    w = SegmentWriter(rng=random.Random(0))
    variants = w.derive_variants(_record(), ["reasoning", "instruct", "mixed"])
    modes = {v.mode for v in variants}
    assert modes == {"reasoning", "instruct", "mixed"}


def test_instruct_strips_all_reasoning():
    w = SegmentWriter(rng=random.Random(0))
    instruct = w.derive_variants(_record(), ["instruct"])[0]
    for turn in instruct.turns:
        assert turn.reasoning is None
        assert turn.reasoning_flag is False
        assert turn.answer is not None  # answer survives


def test_reasoning_variant_keeps_reasoning():
    w = SegmentWriter(rng=random.Random(0))
    full = w.derive_variants(_record(), ["reasoning"])[0]
    assert all(t.reasoning is not None for t in full.turns)


def test_variants_have_distinct_ids():
    w = SegmentWriter(rng=random.Random(0))
    variants = w.derive_variants(_record(), ["reasoning", "instruct"])
    assert variants[0].conversation_id != variants[1].conversation_id


def test_mixed_mode_ratio_keeps_subset():
    rec = ConversationRecord(
        source_seed=Seed("ds", "q"),
        turns=[Turn(i, [Segment("query", "q"), Segment("reasoning", "r"),
                        Segment("answer", "a")]) for i in range(20)],
    )
    w = SegmentWriter(rng=random.Random(0), mixed_mode_keep_ratio=0.5)
    mixed = w.derive_variants(rec, ["mixed"])[0]
    kept = sum(1 for t in mixed.turns if t.reasoning is not None)
    assert 3 <= kept <= 17  # roughly half, not all-or-nothing


def test_dangling_reference_flagging():
    assert flag_dangling_references("As computed above, the answer is 42.")
    assert flag_dangling_references("From step 2 we get 7.")
    assert not flag_dangling_references("The answer is 42 because energy disperses.")


def test_self_containment_annotation_on_instruct():
    turns = [Turn(0, [Segment("query", "q"), Segment("reasoning", "r"),
                      Segment("answer", "As computed above, it is 42.")])]
    rec = ConversationRecord(source_seed=Seed("ds", "q"), turns=turns)
    w = SegmentWriter(rng=random.Random(0))
    instruct = w.derive_variants(rec, ["instruct"])[0]
    assert "self_containment" in instruct.cross_turn_checks


def test_write_jsonl(tmp_path):
    w = SegmentWriter(rng=random.Random(0))
    variants = w.derive_variants(_record(), ["reasoning", "instruct"])
    path = tmp_path / "out.jsonl"
    n = w.write_jsonl(variants, str(path))
    assert n == 2
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # valid JSON
