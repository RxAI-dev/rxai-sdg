"""Frozen ground-truth regression (deterministic layer, no network).

The 5 human-labeled defective conversations (which the OLD judge+gate ACCEPTED)
must be HARD-FAILED by the deterministic pre-filter; the clean control must pass.
The real-judge half is exercised by ``tools/validate_ground_truth.py``.

This anchor may NOT be weakened.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from rxai_sdg.factory.holistic import deterministic_prefilter  # noqa: E402
from ground_truth.regression import build_regression  # noqa: E402


def test_ground_truth_prefilter_matches_human_labels():
    for name, rec, expect_reject, _note in build_regression():
        res = deterministic_prefilter(rec.turns)
        if expect_reject:
            assert not res.passed, f"{name} should HARD-FAIL but passed the pre-filter"
            assert res.hard_fails, name
        else:
            assert res.passed, f"clean control {name} must pass: {res.reasons}"


def test_ground_truth_covers_each_defect_kind():
    by_name = {n: deterministic_prefilter(r.turns)
               for n, r, rej, _ in build_regression() if rej}
    kinds = {n: {h["kind"] for h in res.hard_fails} for n, res in by_name.items()}
    # A (harness) present in the harness/turnindex/sycophancy fixtures
    assert "harness_in_reasoning" in kinds["gt1_harness"]
    assert "numbered_flow_in_reasoning" in kinds["gt2_turnindex"]
    assert "degenerate_reasoning" in kinds["gt3_degenerate"]
    assert "restart_spiral" in kinds["gt4_inconsistency"]
