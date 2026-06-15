"""Cross-turn check tests (spec §6.2)."""

from rxai_sdg.factory.cross_turn import run_cross_turn_checks, cross_turn_pass_rate
from rxai_sdg.factory.ledger import FactLedger
from rxai_sdg.factory.schemas import ConstraintSpec, Segment, Turn


def _turn(idx, answer, cs=None):
    return Turn(idx, [Segment("query", "q"), Segment("answer", answer)],
                constraint_spec=cs)


def test_standing_adherence_checked_on_later_turns():
    cs = ConstraintSpec(intent="reformat", type="json_valid", scope="standing",
                        applies_from_turn=1, verifier="programmatic")
    turns = [
        _turn(0, "intro"),
        _turn(1, '{"a": 1}', cs),
        _turn(2, '{"b": 2}'),   # still json -> pass
        _turn(3, "plain text"),  # violates standing json -> fail
    ]
    checks = run_cross_turn_checks(turns, FactLedger())
    standing = checks["standing"]
    by_turn = {e["checked_turn"]: e["passed"] for e in standing}
    assert by_turn[2] is True
    assert by_turn[3] is False


def test_delayed_recall_fidelity():
    led = FactLedger()
    f = led.plant("teal", planted_turn=1, fact_type="favorite_color")
    cs = ConstraintSpec(intent="fact_recall", type="fact_recall",
                        params={"value": "teal", "match": "exact"},
                        scope="delayed_recall", planted_turn=1, fact_id=f.fact_id,
                        verifier="programmatic")
    turns = [_turn(6, "Your favorite color is teal.", cs)]
    checks = run_cross_turn_checks(turns, led)
    assert checks["delayed_recall"][0]["passed"] is True


def test_update_overwrite_latest_value():
    led = FactLedger()
    f = led.plant("teal", 1, "favorite_color")
    led.update(f.fact_id, "crimson", 4)
    cs = ConstraintSpec(intent="fact_update", type="fact_update",
                        params={"value": "crimson", "match": "exact", "stale_values": ["teal"]},
                        fact_id=f.fact_id, verifier="programmatic")
    good = [_turn(5, "It is now crimson.", cs)]
    assert run_cross_turn_checks(good, led)["update_overwrite"][0]["passed"] is True
    bad = [_turn(5, "It is teal.", cs)]
    assert run_cross_turn_checks(bad, led)["update_overwrite"][0]["passed"] is False


def test_pass_rate_is_one_when_no_checks():
    assert cross_turn_pass_rate({"standing": [], "delayed_recall": []}) == 1.0
