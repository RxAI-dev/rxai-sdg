"""Validate the pipeline against the frozen ground-truth anchor (real endpoint).

Runs the deterministic pre-filter + the frozen LLM judge + the gate on the 5
human-labeled defective fixtures (expect REJECT) and the clean control (expect
ACCEPT). Prints per-fixture trigger reasons and judge scores. Exit 0 iff all
verdicts match ground truth.

    RXAI_GEN_API_URL, RXAI_GEN_API_KEY, RXAI_JUDGE_MODEL
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from rxai_sdg.factory import FactoryConfig  # noqa: E402
from rxai_sdg.factory.clients import MockLLMClient, OpenAILLMClient  # noqa: E402
from rxai_sdg.factory.holistic import (  # noqa: E402
    HolisticJudge, RUBRIC_AXES, deterministic_prefilter,
)
from rxai_sdg.factory.loop import ConversationLoop  # noqa: E402
from rxai_sdg.factory.responder import Responder  # noqa: E402
from rxai_sdg.factory.sampler import IntentPolicySampler  # noqa: E402

from ground_truth.regression import build_regression  # noqa: E402


def _env(name, *alts):
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    sys.exit(f"missing required env var: {name}")


def main() -> int:
    base = _env("RXAI_GEN_API_URL", "OVH_BASE_URL")
    key = _env("RXAI_GEN_API_KEY", "OVH_API_KEY")
    model = _env("RXAI_JUDGE_MODEL")
    client = OpenAILLMClient(model_name=model, api_url=base, api_key=key, timeout=240)
    judge = HolisticJudge(client, rng=random.Random(0))

    cfg = FactoryConfig(seed=0)
    sampler = IntentPolicySampler(cfg.build_taxonomy(), cfg.intent_weights,
                                  cfg.policy_weights, rng=random.Random(0))
    loop = ConversationLoop(Responder(MockLLMClient()), sampler, cfg, rng=random.Random(0))

    failures = []
    print(f"\n==== GROUND-TRUTH REGRESSION (judge={model}) ====\n")
    for name, rec, expect_reject, note in build_regression():
        pf = deterministic_prefilter(rec.turns, regen_threshold=cfg.prefilter_regen_threshold)
        score = judge.score(rec.turns) or {}
        gate_ok = loop._holistic_ok(score, pf)   # True == ACCEPT
        rejected = not gate_ok
        verdict = "REJECT" if rejected else "ACCEPT"
        want = "REJECT" if expect_reject else "ACCEPT"
        ok = (rejected == expect_reject)
        if not ok:
            failures.append(f"{name}: got {verdict}, expected {want}")
        kinds = sorted(set(h["kind"] for h in pf.hard_fails))
        lows = [f"{k}={score[k]}" for k in RUBRIC_AXES
                if isinstance(score.get(k), (int, float)) and score[k] < 7]
        mark = "OK " if ok else "XX "
        print(f"{mark}{name:20s} -> {verdict:6s} (want {want})")
        print(f"      note: {note[:75]}")
        print(f"      pre-filter hard-fails: {kinds or '[]'}")
        print(f"      judge lows(<7): {lows or '[]'}  flags={len(score.get('flagged_turns') or [])}")
    print("\n" + "=" * 60)
    if failures:
        print(f"GROUND-TRUTH FAILED ({len(failures)}):")
        for f in failures:
            print("  XX", f)
        return 1
    print("GROUND-TRUTH GREEN: all 5 defective fixtures REJECTED, clean control ACCEPTED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
