"""Phase 2 - validate the overhauled holistic judge on the REAL endpoint.

Runs the deterministic pre-filter + the frozen LLM judge over the regression
fixtures (``tools/judge_fixtures.py``) and asserts the acceptance criteria:

* the pre-filter hard-fails the turn-index-in-answer and harness-leakage examples
  (with the correct reasons);
* the LLM judge gives LOW reasoning_quality / reasoning_answer_consistency /
  sycophancy_resistance to the defective examples and populates ``flagged_turns``;
* all 5 known-bad fixtures are REJECTED by the gate;
* the hand-written CLEAN fixture PASSES;
* an old-vs-new table (old judge: high score + pass; new: flagged + reject).

Endpoint config is read from the environment (never hard-coded)::

    RXAI_GEN_API_URL, RXAI_GEN_API_KEY, RXAI_JUDGE_MODEL

Exit code 0 iff every assertion holds.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from rxai_sdg.factory import FactoryConfig  # noqa: E402
from rxai_sdg.factory.clients import OpenAILLMClient  # noqa: E402
from rxai_sdg.factory.holistic import (  # noqa: E402
    HolisticJudge, RUBRIC_AXES, deterministic_prefilter,
)
from rxai_sdg.factory.loop import ConversationLoop  # noqa: E402
from rxai_sdg.factory.responder import Responder, format_transcript  # noqa: E402
from rxai_sdg.factory.sampler import IntentPolicySampler  # noqa: E402
from rxai_sdg.factory.clients import MockLLMClient  # noqa: E402
import random  # noqa: E402
import json  # noqa: E402

import judge_fixtures  # noqa: E402


def _env(name: str, *alts: str) -> str:
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    sys.exit(f"missing required env var: {name}")


# --- the OLD judge (for the regression comparison) -------------------------
_OLD_AXES = ["instruction_following", "coherence", "naturalness",
             "role_consistency", "recall_fidelity", "appropriateness"]
_OLD_SYSTEM = (
    "You are a strict but fair conversation-quality judge for a training dataset. "
    "You will be given a transcript of a conversation between a USER and an "
    "ASSISTANT, delimited by <conversation> tags. Your ONLY job is to rate the "
    "ASSISTANT's performance across that transcript. Judge the assistant only "
    "against what the USER turns inside the transcript actually request.\n\n"
    "Score each axis from 1 (terrible) to 10 (excellent): " + ", ".join(_OLD_AXES) +
    ". Output ONLY a JSON object with those integer keys and a \"notes\" string.")


def _old_judge(client, turns):
    """Replicate the OLD judge: query+answer only (no reasoning), 6 axes."""
    transcript = format_transcript(turns)  # user+assistant only - no reasoning!
    prompt = ("Rate the ASSISTANT in the conversation below and return the rubric "
              "JSON.\n\n<conversation>\n" + transcript + "\n</conversation>")
    try:
        resp = client.generate(prompt, system_prompt=_OLD_SYSTEM,
                               temperature=0.0, max_tokens=512)
        m = re.search(r"\{.*\}", resp.text or "", re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    out = {a: int(data[a]) for a in _OLD_AXES
           if isinstance(data.get(a), (int, float)) and not isinstance(data.get(a), bool)}
    return out


def _old_gate_pass(score) -> bool:
    coh, appr = score.get("coherence"), score.get("appropriateness")
    if isinstance(coh, (int, float)) and coh < 6:
        return False
    if isinstance(appr, (int, float)) and appr < 7:
        return False
    return True


def main() -> int:
    base = _env("RXAI_GEN_API_URL", "OVH_BASE_URL")
    key = _env("RXAI_GEN_API_KEY", "OVH_API_KEY")
    model = _env("RXAI_JUDGE_MODEL", "JUDGE_MODEL")
    timeout = float(os.environ.get("RXAI_REQUEST_TIMEOUT", "240"))

    client = OpenAILLMClient(model_name=model, api_url=base, api_key=key, timeout=timeout)
    judge = HolisticJudge(client, rng=random.Random(0))

    cfg = FactoryConfig(seed=0)
    sampler = IntentPolicySampler(cfg.build_taxonomy(), cfg.intent_weights,
                                  cfg.policy_weights, rng=random.Random(0))
    loop = ConversationLoop(Responder(MockLLMClient()), sampler, cfg, rng=random.Random(0))

    fixtures = judge_fixtures.build_fixtures()
    failures: list[str] = []
    rows = []

    for fx in fixtures:
        turns = fx.record.turns
        pf = deterministic_prefilter(turns, regen_threshold=cfg.prefilter_regen_threshold)
        new = judge.score(turns) or {}
        old = _old_judge(client, turns)

        new_gate = loop._holistic_ok(new, pf)
        old_gate = _old_gate_pass(old)

        # -- assertions ----------------------------------------------------
        if fx.prefilter_hard_fail and pf.passed:
            failures.append(f"{fx.name}: expected pre-filter HARD-FAIL, got pass")
        if not fx.prefilter_hard_fail and not pf.passed:
            failures.append(f"{fx.name}: unexpected pre-filter hard-fail: {pf.reasons}")
        if fx.prefilter_kinds:
            got = {h['kind'] for h in pf.hard_fails}
            if not fx.prefilter_kinds <= got:
                failures.append(f"{fx.name}: pre-filter kinds {got} missing {fx.prefilter_kinds}")

        for axis in fx.judge_low_axes:
            v = new.get(axis)
            if not (isinstance(v, (int, float)) and v < cfg.holistic_gate.get(axis, 7)):
                failures.append(f"{fx.name}: judge {axis}={v} not LOW (<{cfg.holistic_gate.get(axis,7)})")
        if fx.gate_should_pass and not new_gate:
            failures.append(f"{fx.name}: new gate REJECTED a clean fixture")
        if (not fx.gate_should_pass) and new_gate:
            failures.append(f"{fx.name}: new gate ACCEPTED a known-bad fixture")
        if not fx.gate_should_pass and not new.get("flagged_turns"):
            failures.append(f"{fx.name}: judge returned no flagged_turns for a bad fixture")

        rows.append({
            "name": fx.name, "covers": fx.covers,
            "old": old, "old_gate": old_gate,
            "new": new, "new_gate": new_gate,
            "pf": pf,
        })

    # -- old-vs-new table --------------------------------------------------
    print("\n================ OLD JUDGE vs NEW JUDGE (real endpoint:",
          model, ") ================\n")
    for r in rows:
        o, n, pf = r["old"], r["new"], r["pf"]
        print(f"### {r['name']}  [{r['covers']}]")
        print(f"  OLD judge (query+answer only): coherence={o.get('coherence')} "
              f"appropriateness={o.get('appropriateness')} "
              f"instruction_following={o.get('instruction_following')} "
              f"-> gate {'PASS' if r['old_gate'] else 'REJECT'}")
        print(f"  NEW judge (reasoning+answer):  reasoning_quality={n.get('reasoning_quality')} "
              f"reasoning_answer_consistency={n.get('reasoning_answer_consistency')} "
              f"sycophancy_resistance={n.get('sycophancy_resistance')} "
              f"coherence={n.get('coherence')} appropriateness={n.get('appropriateness')}")
        if not pf.passed:
            print(f"  PRE-FILTER hard-fail: {pf.reasons}")
        ft = n.get("flagged_turns") or []
        if ft:
            shown = "; ".join(f"t{f['turn_index']}/{f['dimension']}/sev{f['severity']}" for f in ft[:5])
            print(f"  flagged_turns: {shown}")
        print(f"  >>> NEW gate {'PASS' if r['new_gate'] else 'REJECT'}"
              f"   (OLD gate {'PASS' if r['old_gate'] else 'REJECT'})\n")

    print("=" * 80)
    if failures:
        print(f"\nPHASE 2 FAILED ({len(failures)} assertion(s)):")
        for f in failures:
            print("  XX", f)
        return 1
    print("\nPHASE 2 PASSED: pre-filter + frozen judge flag and reject all 5 "
          "known-bad fixtures; the clean fixture passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
