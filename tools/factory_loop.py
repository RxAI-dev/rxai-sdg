"""Phase 4 - one iteration of the autonomous generate -> judge -> analyze loop.

Generates a SMALL real batch with the current pipeline, runs the deterministic
pre-filter + the FROZEN LLM judge over it, aggregates the acceptance signals, and
writes a legible audit record so the iteration is reviewable.

Endpoint / model config is read from the environment (never hard-coded):

    RXAI_GEN_API_URL, RXAI_GEN_API_KEY,
    RXAI_RESPONDER_MODEL, RXAI_SIMULATOR_MODEL, RXAI_CURATOR_MODEL, RXAI_JUDGE_MODEL

The judge is FROZEN (prompt/gate/pre-filter); only the generation pipeline may be
changed between iterations. ``--bias-judge-model`` optionally scores the same batch
with a second judge to check the judge<->simulator self-evaluation bias.

Usage::

    python tools/factory_loop.py --iter 1 --n 10 --concurrency 10 \
        --audit audit/loop/iter1.json --out runs/loop/iter1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from rxai_sdg.factory import DataFactory, FactoryConfig, LengthBand  # noqa: E402
from rxai_sdg.factory.clients import OpenAILLMClient  # noqa: E402
from rxai_sdg.factory.holistic import (  # noqa: E402
    HolisticJudge, ASSISTANT_AXES, USER_QUERY_AXES, RUBRIC_AXES,
    deterministic_prefilter,
)
from rxai_sdg.factory.loop import ConversationLoop  # noqa: E402
from rxai_sdg.factory.schemas import validate_record  # noqa: E402


def _env(name: str, *alts: str, default: str | None = None) -> str:
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    if default is not None:
        return default
    sys.exit(f"missing required env var: {name}")


def load_seeds(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_factory(args):
    base = _env("RXAI_GEN_API_URL", "OVH_BASE_URL")
    key = _env("RXAI_GEN_API_KEY", "OVH_API_KEY")
    t = args.request_timeout
    responder = OpenAILLMClient(
        model_name=_env("RXAI_RESPONDER_MODEL", default="Qwen3.5-397B-A17B"),
        api_url=base, api_key=key, reasoning_field_name="reasoning",
        log_first_raw=args.log_raw, timeout=t,
        frequency_penalty=args.frequency_penalty)
    simulator = OpenAILLMClient(
        model_name=_env("RXAI_SIMULATOR_MODEL", default="Qwen3-Coder-30B-A3B-Instruct"),
        api_url=base, api_key=key, timeout=t)
    curator = OpenAILLMClient(
        model_name=_env("RXAI_CURATOR_MODEL", default="Mistral-Small-3.2-24B-Instruct-2506"),
        api_url=base, api_key=key, timeout=t)
    # FROZEN judge model = Qwen3-Coder-30B (the task's specified judge): the most
    # discriminating of the candidates (Mistral scored only 9-10 and never let the
    # gate reject clean-ish data -> pass-rate pinned at 1.0). The judge<->simulator
    # self-eval concern was checked empirically (iter1 bias probe: Qwen 9.89 vs
    # Mistral 10 on user_query_quality - NOT inflated, slightly stricter).
    judge = OpenAILLMClient(
        model_name=_env("RXAI_JUDGE_MODEL", default="Qwen3-Coder-30B-A3B-Instruct"),
        api_url=base, api_key=key, timeout=t)

    cfg = FactoryConfig(
        seed=args.seed,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        regeneration_limit=2,
        max_responder_calls_per_turn=4,
        min_recall_distance=args.min_recall_distance,
        memory_ratio=args.memory_ratio,
        transform_ratio=args.transform_ratio,
        explore_ratio=1.0 - args.transform_ratio - args.memory_ratio,
        holistic_judge_enabled=True,
        # gate OFF during the loop so EVERY conversation is emitted with its score
        # -> we measure the gate pass-rate and inspect failures ourselves.
        holistic_gate_enabled=False,
        seed_curator_enabled=True,
    )
    cfg.length_bands["smoke"] = LengthBand(args.min_turns, args.max_turns)
    factory = DataFactory(
        cfg, responder, simulator_client=simulator,
        holistic_client=judge, curator_client=curator,
        rng=random.Random(args.seed))
    clients = {"responder": responder, "simulator": simulator,
               "curator": curator, "judge": judge}
    return factory, cfg, clients


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 2) if xs else None


def analyze_batch(records, cfg):
    """Aggregate pre-filter + judge signals and the acceptance metrics."""
    loop = ConversationLoop.__new__(ConversationLoop)  # gate-logic only, no deps
    loop.config = cfg

    per_conv = []
    hard_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    flagged_by_dim: dict[str, int] = {}
    axis_values: dict[str, list] = {k: [] for k in RUBRIC_AXES}
    max_regen = 0
    gate_pass = 0

    for rec in records:
        turns = rec.turns
        pf = deterministic_prefilter(turns, regen_threshold=cfg.prefilter_regen_threshold)
        score = rec.holistic_score or {}
        for h in pf.hard_fails:
            hard_counts[h["kind"]] = hard_counts.get(h["kind"], 0) + 1
        for f in pf.flags:
            flag_counts[f["kind"]] = flag_counts.get(f["kind"], 0) + 1
        for ft in (score.get("flagged_turns") or []):
            flagged_by_dim[ft["dimension"]] = flagged_by_dim.get(ft["dimension"], 0) + 1
        for k in RUBRIC_AXES:
            if isinstance(score.get(k), (int, float)):
                axis_values[k].append(score[k])
        for t in turns:
            r = getattr(getattr(t, "verification", None), "regenerations", 0) or 0
            max_regen = max(max_regen, r)
        passed = ConversationLoop._holistic_ok(loop, score, pf)
        gate_pass += int(passed)
        per_conv.append({
            "conversation_id": rec.conversation_id,
            "seed": (rec.source_seed.first_query or "")[:80],
            "n_turns": len(turns),
            "gate_pass": passed,
            "prefilter_hard_fails": pf.hard_fails,
            "prefilter_flags": pf.flags,
            "scores": {k: score.get(k) for k in RUBRIC_AXES},
            "flagged_turns": score.get("flagged_turns") or [],
            "notes": score.get("notes", ""),
        })

    n = len(records)
    means = {k: _mean(v) for k, v in axis_values.items()}
    pass_rate = round(gate_pass / n, 3) if n else None

    acceptance = {
        "turn_index_in_answer": hard_counts.get("turn_index_in_answer", 0) == 0,
        "harness_in_reasoning": hard_counts.get("harness_in_reasoning", 0) == 0,
        "trailing_artifact": hard_counts.get("trailing_artifact", 0) == 0,
        "degenerate_reasoning": flag_counts.get("degenerate_reasoning", 0) == 0,
        "no_regen_gt_2": max_regen <= cfg.prefilter_regen_threshold,
        "mean_reasoning_quality_ge_8": (means.get("reasoning_quality") or 0) >= 8,
        "mean_rac_ge_8": (means.get("reasoning_answer_consistency") or 0) >= 8,
        "gate_pass_rate_in_band": (pass_rate is not None and 0.65 <= pass_rate <= 0.95),
    }
    return {
        "n": n,
        "hard_counts": hard_counts,
        "flag_counts": flag_counts,
        "flagged_by_dimension": flagged_by_dim,
        "judge_means": means,
        "max_regen": max_regen,
        "gate_pass": gate_pass,
        "gate_pass_rate": pass_rate,
        "acceptance": acceptance,
        "acceptance_all": all(acceptance.values()),
        "per_conversation": per_conv,
    }


def bias_probe(records, model, base, key, timeout):
    """Score the SAME batch with a second judge to check self-eval bias."""
    client = OpenAILLMClient(model_name=model, api_url=base, api_key=key, timeout=timeout)
    judge = HolisticJudge(client, rng=random.Random(0))
    uqq, rq = [], []
    for rec in records:
        s = judge.score(rec.turns) or {}
        if isinstance(s.get("user_query_quality"), (int, float)):
            uqq.append(s["user_query_quality"])
        if isinstance(s.get("reasoning_quality"), (int, float)):
            rq.append(s["reasoning_quality"])
    return {"model": model, "user_query_quality_mean": _mean(uqq),
            "reasoning_quality_mean": _mean(rq), "n": len(uqq)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=0)
    ap.add_argument("--seeds", default=str(ROOT / "tools" / "seeds.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--hypothesis", default="", help="hypothesis/diff note for the audit")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=9)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="responder decoding frequency_penalty (breaks degenerate loops)")
    ap.add_argument("--request-timeout", type=float, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-ratio", type=float, default=0.2)
    ap.add_argument("--transform-ratio", type=float, default=0.3)
    ap.add_argument("--min-recall-distance", type=int, default=4)
    ap.add_argument("--bias-judge-model", default=None)
    ap.add_argument("--log-raw", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or str(ROOT / "runs" / "loop" / f"iter{args.iter}.jsonl")
    audit = args.audit or str(ROOT / "audit" / "loop" / f"iter{args.iter}.json")

    seeds = load_seeds(args.seeds)[:args.n]
    factory, cfg, clients = build_factory(args)

    t0 = time.time()
    records = factory.generate(seeds, band="smoke", verbose=True)
    dt = time.time() - t0
    for rec in records:
        validate_record(rec.to_dict())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    factory.writer.write_jsonl(records, out)

    result = analyze_batch(records, cfg)
    usage = {name: c.usage() for name, c in clients.items()}
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())

    st = factory.stats
    bias = None
    if args.bias_judge_model:
        base = _env("RXAI_GEN_API_URL", "OVH_BASE_URL")
        key = _env("RXAI_GEN_API_KEY", "OVH_API_KEY")
        bias = {
            "frozen_judge": bias_probe(
                records, _env("RXAI_JUDGE_MODEL", default="Mistral-Small-3.2-24B-Instruct-2506"),
                base, key, args.request_timeout),
            "alt_judge": bias_probe(records, args.bias_judge_model, base, key, args.request_timeout),
        }

    record = {
        "iteration": args.iter,
        "hypothesis_or_change": args.hypothesis,
        "config": {
            "n_seeds": len(seeds), "concurrency": args.concurrency,
            "band": [args.min_turns, args.max_turns], "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "models": {
                "responder": clients["responder"].model_name,
                "simulator": clients["simulator"].model_name,
                "curator": clients["curator"].model_name,
                "judge": clients["judge"].model_name,
            },
        },
        "generation_stats": {
            "wall_seconds": round(dt, 1),
            "records_emitted": len(records),
            "seeds_used": st.seeds_used, "seeds_skipped": st.seeds_skipped,
            "discarded": st.conversations_discarded,
            "reasoning_missing": st.loop.reasoning_missing,
            "malformed": st.loop.malformed_outputs,
            "regenerations": st.loop.total_regenerations,
            "intent_resamples": st.loop.intent_resamples,
        },
        "token_usage": {"by_client": usage, "total_tokens": total_tokens},
        "analysis": result,
        "bias_probe": bias,
    }

    Path(audit).parent.mkdir(parents=True, exist_ok=True)
    with open(audit, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    # -- console summary ---------------------------------------------------
    a = result["acceptance"]
    print(f"\n===== ITERATION {args.iter}: {len(records)} records in {dt:.0f}s "
          f"| total_tokens={total_tokens} =====")
    print(f"gen: discarded={st.conversations_discarded} regen={st.loop.total_regenerations} "
          f"resamples={st.loop.intent_resamples} malformed={st.loop.malformed_outputs} "
          f"reasoning_missing={st.loop.reasoning_missing}")
    print("pre-filter hard-fails:", result["hard_counts"] or "{}")
    print("pre-filter soft flags:", result["flag_counts"] or "{}")
    print("judge flagged by dim :", result["flagged_by_dimension"] or "{}")
    print("judge means:")
    for k in ASSISTANT_AXES:
        print(f"    {k:28s} {result['judge_means'].get(k)}")
    for k in USER_QUERY_AXES:
        print(f"    {k:28s} {result['judge_means'].get(k)}")
    print(f"max regen across turns: {result['max_regen']}")
    print(f"gate pass: {result['gate_pass']}/{result['n']}  rate={result['gate_pass_rate']}")
    if bias:
        print("bias probe (user_query_quality mean):")
        print(f"    frozen  {bias['frozen_judge']['model']}: "
              f"{bias['frozen_judge']['user_query_quality_mean']}")
        print(f"    alt     {bias['alt_judge']['model']}: "
              f"{bias['alt_judge']['user_query_quality_mean']}")
    print("-" * 60)
    for k, v in a.items():
        print(f"  {'OK ' if v else 'XX '}{k}")
    print("-" * 60)
    print("ACCEPTANCE:", "PASS" if result["acceptance_all"] else "NOT MET")
    print(f"audit written: {audit}")
    return 0 if result["acceptance_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
