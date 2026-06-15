"""HuggingFace dataset row-building / round-trip tests."""

import json

import pytest

from rxai_sdg.factory.dataset import (
    record_to_row, row_to_record, FactoryDatasetPostprocessor,
    SCALAR_COLUMNS, JSON_COLUMNS,
)
from rxai_sdg.factory.schemas import (
    Seed, Segment, Turn, ConstraintSpec, VerifyResult, Fact, ConversationRecord,
)


def _record():
    turn = Turn(
        turn_index=0,
        segments=[Segment("query", "q"), Segment("reasoning", "r"), Segment("answer", "a")],
        intent="lexical_constraint", policy="cumulative",
        constraint_spec=ConstraintSpec(intent="lexical_constraint", type="first_letter",
                                       params={"letter": "A"}),
        verification=VerifyResult(True, "ok", 1),
    )
    return ConversationRecord(
        source_seed=Seed("ds", "first?", "math", "math", "en", False),
        turns=[turn], mode="reasoning",
        fact_ledger=[Fact("f1", "teal", 0, "favorite_color", [{"turn": 0, "value": "teal"}])],
        cross_turn_checks={"standing": [{"checked_turn": 1, "passed": True}]},
        holistic_score={"coherence": 9},
    )


def test_record_to_row_schema_is_flat_and_stable():
    row = record_to_row(_record())
    assert set(row) == set(SCALAR_COLUMNS) | set(JSON_COLUMNS)
    assert row["category"] == "math"
    assert row["mode"] == "reasoning"
    assert row["length"] == 1
    # variable-keyed fields are JSON strings
    for col in JSON_COLUMNS:
        assert isinstance(row[col], str)
    assert json.loads(row["turns"])[0]["constraint_spec"]["params"] == {"letter": "A"}


def test_row_round_trip():
    rec = _record()
    rebuilt = row_to_record(record_to_row(rec))
    assert rebuilt.to_dict() == rec.to_dict()


def test_rows_have_identical_keys_across_records():
    # heterogeneous records must still yield identical (append-safe) row schemas
    r1 = _record()
    r2 = _record()
    r2.turns[0].constraint_spec = None  # different nested content
    r2.holistic_score = None
    k1 = set(record_to_row(r1))
    k2 = set(record_to_row(r2))
    assert k1 == k2


def test_postprocessor_to_dataset():
    datasets = pytest.importorskip("datasets")
    post = FactoryDatasetPostprocessor([_record(), _record()])
    ds = post.to_dataset()
    assert len(ds) == 2
    assert "turns" in ds.column_names
    # JSON column reconstructs into a full record
    rec = row_to_record(ds[0])
    assert rec.mode == "reasoning"
