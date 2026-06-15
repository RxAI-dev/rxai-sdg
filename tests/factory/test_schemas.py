import pytest

from rxai_sdg.factory.schemas import (
    Seed, Segment, Turn, ConstraintSpec, VerifyResult, Fact,
    ConversationRecord, validate_record, SchemaError,
)


def _record(mode="reasoning"):
    turn = Turn(
        turn_index=0,
        segments=[Segment("query", "q"), Segment("reasoning", "r"), Segment("answer", "a")],
        intent="reformat", policy="immediate",
        constraint_spec=ConstraintSpec(intent="reformat", type="json_valid"),
        verification=VerifyResult(True, "ok", 0),
    )
    return ConversationRecord(
        source_seed=Seed("ds", "first?", "math", "math"),
        turns=[turn], mode=mode,
        fact_ledger=[Fact("f1", "teal", 0, "favorite_color", [{"turn": 0, "value": "teal"}])],
    )


def test_roundtrip_to_from_dict():
    rec = _record()
    d = rec.to_dict()
    rebuilt = ConversationRecord.from_dict(d)
    assert rebuilt.to_dict() == d
    assert rebuilt.length == 1
    assert rebuilt.turns[0].constraint_spec.type == "json_valid"


def test_validate_record_ok():
    validate_record(_record().to_dict())  # should not raise


def test_validate_record_length_mismatch():
    d = _record().to_dict()
    d["length"] = 5
    with pytest.raises(SchemaError):
        validate_record(d)


def test_validate_record_instruct_with_reasoning_fails():
    d = _record(mode="instruct").to_dict()
    with pytest.raises(SchemaError):
        validate_record(d)


def test_validate_record_missing_answer_segment():
    d = _record().to_dict()
    d["turns"][0]["segments"] = [{"segment_type": "query", "text": "q"}]
    with pytest.raises(SchemaError):
        validate_record(d)


def test_constraint_spec_defaults():
    cs = ConstraintSpec(intent="x", type="y")
    assert cs.lang == "en"
    assert cs.verifier == "programmatic"
    assert cs.scope == "current_turn"


def test_turn_segment_accessors():
    t = Turn(0, [Segment("query", "q"), Segment("answer", "a")])
    assert t.query == "q"
    assert t.answer == "a"
    assert t.reasoning is None
