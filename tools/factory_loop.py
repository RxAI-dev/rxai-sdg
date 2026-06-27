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
    # The Responder/Teacher MUST be a REASONING model with CLEAN genuine CoT (see
    # prompts.py): gpt-oss-120b (default). The Qwen family bakes in meta-reasoning
    # poison; instruct models only fake <think> (and do so unreliably). The genuine
    # reasoning is captured from the ``reasoning`` field.
    rt = args.max_retries
    responder = OpenAILLMClient(
        model_name=_env("RXAI_RESPONDER_MODEL", default="gpt-oss-120b"),
        api_url=base, api_key=key, reasoning_field_name="reasoning",
        log_first_raw=args.log_raw, timeout=t,
        frequency_penalty=args.frequency_penalty, max_retries=rt)
    simulator = OpenAILLMClient(
        model_name=_env("RXAI_SIMULATOR_MODEL", default="Qwen3-Coder-30B-A3B-Instruct"),
        api_url=base, api_key=key, timeout=t, max_retries=rt)
    curator = OpenAILLMClient(
        model_name=_env("RXAI_CURATOR_MODEL", default="Mistral-Small-3.2-24B-Instruct-2506"),
        api_url=base, api_key=key, timeout=t, max_retries=rt)
    # Judge model. Default Qwen3-Coder-30B (fast, the task's specified judge). NOTE
    # (validated empirically this iteration): the 30B judge is BLIND to confident
    # fabrication - it scores fabricated citations and made-up technical constructs
    # factual_grounding 10. The deterministic detectors (fabricated_citation,
    # restart_spiral, confidence_mismatch) are the robust gate for that class. For a
    # materially STRONGER semantic judge, set RXAI_JUDGE_MODEL=Qwen3.5-397B-A17B: it
    # scores those same fabrications factual_grounding 1-2 with a severity-3 flag,
    # while keeping genuinely-good conversations high. It is a reasoning model, so it
    # needs the generous judge token budget (config.holistic_judge_max_tokens=12000)
    # or it truncates the rubric JSON to None.
    judge = OpenAILLMClient(
        model_name=_env("RXAI_JUDGE_MODEL", default="Qwen3-Coder-30B-A3B-Instruct"),
        api_url=base, api_key=key, timeout=t, max_retries=rt)

    cfg = FactoryConfig(
        seed=args.seed,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_source=_env("RXAI_REASONING_SOURCE", default=args.reasoning_source),
        dataset_name=_env("RXAI_DATASET_NAME", default=args.dataset_name),
        regeneration_limit=3,
        max_responder_calls_per_turn=6,   # room to resample a sporadically-leaky turn
        min_recall_distance=args.min_recall_distance,
        memory_ratio=args.memory_ratio,
        transform_ratio=args.transform_ratio,
        explore_ratio=1.0 - args.transform_ratio - args.memory_ratio,
        holistic_judge_enabled=True,
        # gate OFF during the loop so EVERY conversation is emitted with its score
        # -> we measure the gate pass-rate and inspect failures ourselves.
        holistic_gate_enabled=False,
        seed_curator_enabled=True,
        # new mechanisms (problems 1 & 2). The reasoning-rewrite pass transforms
        # annotator-voice reasoning in place (no yield cost); factuality attaches its
        # result and is applied post-hoc by analyze_batch since the loop runs gate-OFF
        # for measurement.
        factuality_gate_enabled=args.factuality_gate,
        reasoning_rewrite_enabled=args.voice_gate,
        skip_fact_dense_seeds=args.skip_fact_dense,
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
        # focused factuality gate (problem 2): a conversation the decomposed
        # claim-check found a confident-FALSE claim in fails the gate too. Attached
        # by the loop as score["factuality"]; only gate when the check was available.
        fact = score.get("factuality") if isinstance(score, dict) else None
        fact_false = bool(fact and fact.get("available") and not fact.get("passed"))
        if fact_false:
            passed = False
        gate_pass += int(passed)
        per_conv.append({
            "conversation_id": rec.conversation_id,
            "seed": (rec.source_seed.first_query or "")[:80],
            "n_turns": len(turns),
            "gate_pass": passed,
            "factuality_false_claims": (fact or {}).get("false_claims", []) if fact else [],
            "prefilter_hard_fails": pf.hard_fails,
            "prefilter_flags": pf.flags,
            "scores": {k: score.get(k) for k in RUBRIC_AXES},
            "flagged_turns": score.get("flagged_turns") or [],
            "notes": score.get("notes", ""),
        })

    n = len(records)
    means = {k: _mean(v) for k, v in axis_values.items()}
    pass_rate = round(gate_pass / n, 3) if n else None

    # Deterministic hard-fails that survive INTO the emitted (gate-passing) set. By
    # construction a gate-passing conversation has prefilter.passed=True, so this
    # should be empty - it is the "emitted dataset is clean" invariant. (The batch is
    # generated gate-OFF for measurement, so the raw hard_counts above include the
    # rejected conversations and are expected to be non-zero - that is the defect the
    # gate catches, not an acceptance failure.)
    emitted_hard: dict[str, int] = {}
    for c in per_conv:
        if c["gate_pass"]:
            for h in c["prefilter_hard_fails"]:
                emitted_hard[h["kind"]] = emitted_hard.get(h["kind"], 0) + 1

    # Acceptance is the MEASUREMENT being healthy, not the judge scoring high. A judge
    # pinned near 10 is BLIND (the failure this whole effort fixes), so we do NOT
    # require high means - we require the opposite: the judge must DISCRIMINATE, and
    # the gate pass-rate must sit in the 0.65-0.80 band (near 1.0 is suspicious).
    rq_mean = means.get("reasoning_quality")
    fg_mean = means.get("factual_grounding")
    acceptance = {
        # the emitted (gate-passing) dataset carries none of the objective defects
        # (this already covers excess_regenerations, which is itself a hard-fail kind)
        "emitted_no_hard_fails": sum(emitted_hard.values()) == 0,
        # judge is not blind: a discriminating judge does not pin reasoning_quality /
        # factual_grounding at the ceiling across a real, mixed batch.
        "judge_discriminating": (
            (rq_mean is not None and rq_mean < 9.5)
            or (fg_mean is not None and fg_mean < 9.5)
            or (pass_rate is not None and pass_rate < 0.95)),
        # convergence band (task §3 Phase E): scores track isolated review here.
        "gate_pass_rate_in_band": (pass_rate is not None and 0.65 <= pass_rate <= 0.80),
    }
    return {
        "n": n,
        "hard_counts": hard_counts,
        "emitted_hard_counts": emitted_hard,
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
    ap.add_argument("--rejected-out", default=None,
                    help="path for conversations that FAIL the gate (with reasons), so "
                         "the rejected pile can be inspected for false negatives. "
                         "Defaults to '<out>.rejected.jsonl'.")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--hypothesis", default="", help="hypothesis/diff note for the audit")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=9)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--reasoning-source", default="auto",
                    choices=["auto", "field", "inline"],
                    help="where to read the teacher CoT: auto|field|inline "
                         "(auto handles both a reasoning field and inline <think>)")
    ap.add_argument("--max-retries", type=int,
                    default=int(os.environ.get("RXAI_MAX_RETRIES", "4")),
                    help="per-call transient-error retries (5xx/429/timeout) with "
                         "backoff; raise it to ride out a flaky endpoint window")
    ap.add_argument("--dataset-name", default="seeds",
                    help="value stamped onto every emitted example's "
                         "source_seed.dataset (also via RXAI_DATASET_NAME)")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="responder decoding frequency_penalty (breaks degenerate loops)")
    ap.add_argument("--request-timeout", type=float, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-ratio", type=float, default=0.2)
    ap.add_argument("--transform-ratio", type=float, default=0.3)
    ap.add_argument("--min-recall-distance", type=int, default=4)
    ap.add_argument("--bias-judge-model", default=None)
    ap.add_argument("--factuality-gate", action="store_true",
                    help="enable the focused decomposed factuality gate (problem 2)")
    ap.add_argument("--voice-gate", action="store_true",
                    help="enable the reasoning-rewrite pass: re-voice annotator-voice "
                         "reasoning into genuine first-person thinking (problem 1)")
    ap.add_argument("--skip-fact-dense", action="store_true",
                    help="drop fact-dense seeds at curation (problem 2): obscure "
                         "rankings/biographies/stats where the responder fabricates")
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

    # Persist the REJECTED pile separately (the runner generates gate-OFF, so every
    # completed conversation is in --out; here we split out the ones that FAIL the gate,
    # annotated with their rejection reasons, so false negatives can be eyeballed).
    rejected_out = args.rejected_out or (str(Path(out).with_suffix("")) + ".rejected.jsonl")
    gate_by_id = {c["conversation_id"]: c for c in result["per_conversation"]}
    n_rejected = 0
    Path(rejected_out).parent.mkdir(parents=True, exist_ok=True)
    with open(rejected_out, "w", encoding="utf-8") as fh:
        for rec in records:
            info = gate_by_id.get(rec.conversation_id, {})
            if info.get("gate_pass", True):
                continue
            n_rejected += 1
            d = rec.to_dict()
            d["_rejection"] = {
                "prefilter_hard_fails": info.get("prefilter_hard_fails", []),
                "flagged_turns": info.get("flagged_turns", []),
                "scores": info.get("scores", {}),
                "notes": info.get("notes", ""),
            }
            fh.write(json.dumps(d, default=str) + "\n")

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
    print(f"rejected pile: {n_rejected} -> {rejected_out}")
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
