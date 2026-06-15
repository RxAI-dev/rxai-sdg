"""SegmentWriter + dangling-reference tests (spec §5.6)."""

import json
import random

from rxai_sdg.factory.schemas import Seed, Segment, Turn, ConversationRecord
from rxai_sdg.factory.writer import SegmentWriter, flag_dangling_references


def _record():
    turns = [
        Turn(0, [Segment("query", "q0"), Segment("reasoning", "r0"), Segment("answer", "a0")]),
        Turn(1, [Segment("query", "q1"), Segment("reasoning", "r1"), Segment("answer", "a1")]),
    ]
    return ConversationRecord(source_seed=Seed("ds", "q0"), turns=turns, mode="reasoning")


def test_dangling_reference_flagging():
    assert flag_dangling_references("As computed above, the answer is 42.")
    assert flag_dangling_references("From step 2 we get 7.")
    assert not flag_dangling_references("The answer is 42 because energy disperses.")


def test_to_dict_passthrough():
    w = SegmentWriter()
    d = w.to_dict(_record())
    assert d["mode"] == "reasoning"
    assert d["length"] == 2


def test_write_jsonl(tmp_path):
    w = SegmentWriter()
    path = tmp_path / "out.jsonl"
    n = w.write_jsonl([_record(), _record()], str(path))
    assert n == 2
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # valid JSON
