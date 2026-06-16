"""Real-generation iteration harness (task §5).

Generates a small batch of conversations against the configured OpenAI-compatible
endpoint, then runs the §4 defect detectors and reports the holistic-judge
coherence. One invocation = one iteration of the fix loop.

Configuration is read from the environment and **never** hardcoded or written to
any committed file::

    OVH_BASE_URL, OVH_API_KEY
    RESPONDER_MODEL, SIMULATOR_MODEL, CURATOR_MODEL, JUDGE_MODEL
    (JUDGE_BASE_URL / JUDGE_API_KEY default to the OVH endpoint)

Usage::

    python tools/iterate.py --out runs/iter1.jsonl --concurrency 8

Exit code mirrors the detectors (0 = clean batch).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from rxai_sdg.factory import FactoryConfig, DataFactory, LengthBand  # noqa: E402
from rxai_sdg.factory.clients import OpenAILLMClient  # noqa: E402
from rxai_sdg.factory.schemas import validate_record  # noqa: E402
import analyze_records  # noqa: E402


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if not val:
        sys.exit(f"missing required env var: {name}")
    return val


def load_seeds(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_factory(args) -> DataFactory:
    base = _env("OVH_BASE_URL")
    key = _env("OVH_API_KEY")
    judge_base = os.environ.get("JUDGE_BASE_URL") or base
    judge_key = os.environ.get("JUDGE_API_KEY") or key

    responder = OpenAILLMClient(
        model_name=_env("RESPONDER_MODEL", "Qwen3.5-397B-A17B"),
        api_url=base, api_key=key, reasoning_field_name="reasoning",
        log_first_raw=args.log_raw)
    simulator = OpenAILLMClient(
        model_name=_env("SIMULATOR_MODEL", "Qwen3-Coder-30B-A3B-Instruct"),
        api_url=base, api_key=key)
    curator = OpenAILLMClient(
        model_name=_env("CURATOR_MODEL", "Qwen3.6-27B"),
        api_url=base, api_key=key)
    judge = OpenAILLMClient(
        model_name=_env("JUDGE_MODEL", "gpt-oss-120b"),
        api_url=judge_base, api_key=judge_key)

    cfg = FactoryConfig(
        seed=args.seed,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=0.7,
        regeneration_limit=2,
        max_responder_calls_per_turn=4,
        min_recall_distance=args.min_recall_distance,
        memory_ratio=args.memory_ratio,
        explore_ratio=1.0 - args.transform_ratio - args.memory_ratio,
        transform_ratio=args.transform_ratio,
        holistic_judge_enabled=True,
        seed_curator_enabled=True,
    )
    cfg.length_bands["smoke"] = LengthBand(args.min_turns, args.max_turns)
    return DataFactory(
        cfg, responder, simulator_client=simulator,
        holistic_client=judge, curator_client=curator,
        rng=random.Random(args.seed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real-generation iteration harness")
    ap.add_argument("--seeds", default=str(ROOT / "tools" / "seeds.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "iter.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=9)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-ratio", type=float, default=0.2)
    ap.add_argument("--transform-ratio", type=float, default=0.3)
    ap.add_argument("--min-recall-distance", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-raw", action="store_true")
    args = ap.parse_args(argv)

    seeds = load_seeds(args.seeds)
    if args.limit:
        seeds = seeds[:args.limit]

    factory = build_factory(args)
    t0 = time.time()
    records = factory.generate(seeds, band="smoke", verbose=True)
    dt = time.time() - t0

    for rec in records:
        validate_record(rec.to_dict())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = factory.writer.write_jsonl(records, args.out)

    st = factory.stats
    print(f"\n=== generation: {n} records in {dt:.0f}s "
          f"(seeds={st.seeds_used} skipped={st.seeds_skipped} "
          f"discarded={st.conversations_discarded} "
          f"reasoning_missing={st.loop.reasoning_missing} "
          f"malformed={st.loop.malformed_outputs} "
          f"regen={st.loop.total_regenerations} resamples={st.loop.intent_resamples}) ===\n")

    result = analyze_records.analyze([r.to_dict() for r in records])
    s = result["summary"]
    print(f"records={s['records']} turns={s['total_turns']} "
          f"median_coherence={s['median_coherence']} (n={s['coherence_samples']})")
    print("-" * 60)
    for name in analyze_records.DETECTORS:
        offenders = result["defects"][name]
        flag = "OK " if not offenders else "XX "
        print(f"{flag}{name:24s} {len(offenders)}")
        for off in offenders[:6]:
            print(f"      - {off}")
    print("-" * 60)
    clean = s["clean"]
    coh = s["median_coherence"]
    gate = clean and coh is not None and coh >= 8
    print("CLEAN" if clean else "NOT CLEAN",
          f"| median coherence {coh} | GATE {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
