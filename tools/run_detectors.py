#!/usr/bin/env python3
"""Run the defect detectors + the strengthened deterministic pre-filter over a
factory dataset (parquet of emitted records), quantify prevalence, and write a
NEW filtered parquet (the original is never mutated).

Usage:
    python tools/run_detectors.py data/_immutable/original.parquet \
        --out data/repaired/filtered.parquet --report reports/prevalence_pre.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rxai_sdg.factory.dataset import row_to_record  # noqa: E402
from rxai_sdg.factory.detectors import run_all  # noqa: E402
from rxai_sdg.factory.holistic import deterministic_prefilter  # noqa: E402

# per-class repair policy (Step 3): how a conversation flagged by each detector
# should be handled at source. A/B-fabrication on an ungrounded premise cannot be
# patched honestly -> reseed/drop; the rest are regenerate-in-place.
CLASS_POLICY = {
    "A": "reseed_or_drop (fabricated premise - cannot patch honestly)",
    "C": "regenerate code + re-execute to confirm",
    "B": "regenerate answer to drop ungrounded specifics (or reseed if premise ungrounded)",
    "D": "regenerate reasoning as genuine cognition (keep answer)",
    "E": "regenerate reasoning (no filler / no answer duplication)",
    "F": "correct constraint_spec scope; regenerate only if answer obeyed a phantom rule",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--out", default="data/repaired/filtered.parquet")
    ap.add_argument("--report", default="reports/prevalence_pre.json")
    ap.add_argument("--no-code", action="store_true", help="skip code execution (faster)")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    n = len(df)
    det_convs: Counter = Counter()
    det_flags: Counter = Counter()
    prefilter_reject = 0
    reject_kinds: Counter = Counter()
    per_conv = []
    keep_idx = []

    for i, row in df.iterrows():
        rec = row_to_record(row.to_dict())
        rd = {"first_query": row["first_query"], "turns": row["turns"]}
        flags = run_all(rd, run_code=not args.no_code)
        dets = sorted({f.detector for f in flags})
        for d in dets:
            det_convs[d] += 1
        for f in flags:
            det_flags[f.detector] += 1
        pf = deterministic_prefilter(rec.turns)
        if not pf.passed:
            prefilter_reject += 1
            for h in pf.hard_fails:
                reject_kinds[h["kind"]] += 1
        else:
            keep_idx.append(i)
        per_conv.append({
            "conversation_id": row["conversation_id"],
            "first_query": row["first_query"][:80],
            "detectors": dets,
            "prefilter_pass": pf.passed,
            "hard_fails": [h["kind"] for h in pf.hard_fails],
            "flags": [{"det": f.detector, "name": f.name, "sev": f.severity,
                       "turn": f.turn_index, "evidence": f.evidence} for f in flags],
        })

    report = {
        "input": args.parquet,
        "n_conversations": n,
        "prevalence": {d: {"conversations": det_convs[d], "flags": det_flags[d],
                           "conv_pct": round(100 * det_convs[d] / n, 1),
                           "policy": CLASS_POLICY.get(d, "")}
                       for d in ["A", "B", "C", "D", "E", "F"]},
        "prefilter": {
            "rejected": prefilter_reject,
            "kept": n - prefilter_reject,
            "yield": f"{n - prefilter_reject}/{n}",
            "reject_kinds": dict(reject_kinds.most_common()),
        },
        "per_conversation": per_conv,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report, "w"), ensure_ascii=False, indent=1)

    # write NEW filtered parquet (the gate-passing subset); original untouched
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.loc[keep_idx].to_parquet(args.out, index=False)

    print(f"input: {args.parquet}  ({n} conversations)")
    print("prevalence (conversations flagged / total):")
    for d in ["A", "B", "C", "D", "E", "F"]:
        print(f"  {d}: {det_convs[d]:3d} ({100*det_convs[d]/n:4.1f}%)  flags={det_flags[d]}")
    print(f"\nstrengthened pre-filter: KEEP {n-prefilter_reject}/{n}  "
          f"REJECT {prefilter_reject}")
    print(f"  reject kinds: {dict(reject_kinds.most_common())}")
    print(f"\nreport -> {args.report}")
    print(f"filtered parquet -> {args.out}  (original NOT mutated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
