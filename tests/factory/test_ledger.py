"""FactLedger / NeedlePlanner tests (spec §11.6, §12)."""

import random

from rxai_sdg.factory.ledger import FactLedger, NeedlePlanner


def test_plant_assigns_unique_ids_and_history():
    led = FactLedger()
    f1 = led.plant("teal", planted_turn=1, fact_type="favorite_color")
    f2 = led.plant("4821", planted_turn=2, fact_type="lucky_number")
    assert f1.fact_id != f2.fact_id
    assert f1.value_history == [{"turn": 1, "value": "teal"}]
    assert len(led) == 2


def test_update_overwrite_tracks_latest_and_stale():
    led = FactLedger()
    f = led.plant("teal", 1, "favorite_color")
    led.update(f.fact_id, "crimson", 5)
    led.update(f.fact_id, "amber", 9)
    assert led.latest(f.fact_id) == "amber"
    assert led.stale_values(f.fact_id) == ["teal", "crimson"]
    assert len(f.value_history) == 3


def test_recall_check_exact():
    led = FactLedger()
    f = led.plant("teal", 1, "favorite_color")
    ok, _ = led.recall_check("Your favorite color is teal.", f.fact_id, "exact")
    assert ok
    bad, _ = led.recall_check("Your favorite color is blue.", f.fact_id, "exact")
    assert not bad


def test_needle_planner_delayed_recall_distance():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=4)
    planner.plant_fact(turn=1)
    # not recallable before distance is reached
    assert planner.recallable_fact(turn=3, min_distance=4) is None
    # recallable at >= min_distance
    assert planner.recallable_fact(turn=5, min_distance=4) is not None


def test_needle_planner_picks_oldest_fact():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0), min_distance=2)
    a = planner.plant_fact(turn=1)
    planner.plant_fact(turn=2)
    picked = planner.recallable_fact(turn=10, min_distance=2)
    assert picked.fact_id == a.fact_id  # oldest


def test_update_value_for_changes_value():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(3))
    f = planner.plant_fact(turn=1, fact_type="favorite_color")
    old = f.value
    planner.update_value_for(f, turn=4)
    assert led.latest(f.fact_id) != old
    assert old in led.stale_values(f.fact_id)


def test_planner_phrasings_reference_value():
    led = FactLedger()
    planner = NeedlePlanner(led, rng=random.Random(0))
    f = planner.plant_fact(turn=1, fact_type="hometown")
    assert str(f.value) in planner.plant_phrasing(f)
    assert "hometown" in planner.recall_question(f)
