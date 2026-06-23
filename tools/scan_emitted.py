#!/usr/bin/env python3
"""Independent residual scan over a factory run file (``--out`` JSONL).

This is a SECOND, deliberately independent check on top of the in-loop gate: it
re-runs the FROZEN deterministic pre-filter over the emitted reasoning/answers
and, crucially, also sweeps an *exploratory* cue net for meta phrases the frozen
detectors do not yet catch - so a brand-new leak surfaces here instead of in the
shipped dataset. It does not call the endpoint; it reads the judge scores already
recorded in each record's ``holistic_score``.

Usage:
    python tools/scan_emitted.py runs/loop/newloop_25b.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rxai_sdg.factory.holistic import deterministic_prefilter  # noqa: E402
from rxai_sdg.factory.responder import has_harness_leak  # noqa: E402


class _Seg:
    def __init__(self, d):
        self.segment_type = d.get("segment_type")
        self.text = d.get("text", "")


class _Ver:
    def __init__(self, d):
        self.regenerations = (d or {}).get("regenerations", 0) or 0


class _Turn:
    """Minimal Turn shim exposing exactly what the pre-filter reads."""

    def __init__(self, d):
        self.turn_index = d.get("turn_index", 0)
        self._segs = [_Seg(s) for s in d.get("segments", [])]
        self.verification = _Ver(d.get("verification"))

    def _seg(self, kind):
        for s in self._segs:
            if s.segment_type == kind:
                return s.text or ""
        return ""

    @property
    def query(self):
        return self._seg("query")

    @property
    def reasoning(self):
        return self._seg("reasoning")

    @property
    def answer(self):
        return self._seg("answer")


# Exploratory cue net: substrings that are *usually* meta/harness bookkeeping.
# A hit that the frozen `has_harness_leak` does NOT already catch is a candidate
# new detector - it is printed so a human (or the next iteration) can decide.
_EXPLORE_CUES = [
    "thinking process", "final output", "as per", "as an ai", "i should comply",
    "i must follow", "policy", "guideline", "safe completion", "disallowed",
    "openai", "system prompt", "system instruction", "persona:", "audience:",
    "tone:", "format:", "word count", "the user wants me to", "i need to produce",
    "must honor", "matches the provided", "suggested response", "good response",
]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        sys.exit("usage: scan_emitted.py RUN.jsonl")
    path = argv[0]
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    hard_counts: Counter = Counter()
    rq, rac, appr = [], [], []
    explore_new: Counter = Counter()
    explore_examples: dict[str, str] = {}
    n_turns = 0
    leak_records = []

    for ri, rec in enumerate(recs):
        turns = [_Turn(t) for t in rec.get("turns", [])]
        n_turns += len(turns)
        pf = deterministic_prefilter(turns)
        for h in pf.hard_fails:
            hard_counts[h["kind"]] += 1
        if pf.hard_fails:
            leak_records.append((ri, [h["kind"] for h in pf.hard_fails]))
        score = rec.get("holistic_score") or {}
        for key, bucket in (("reasoning_quality", rq),
                            ("reasoning_answer_consistency", rac),
                            ("appropriateness", appr)):
            v = score.get(key)
            if isinstance(v, (int, float)):
                bucket.append(v)
        # exploratory net over reasoning the frozen detector did NOT flag
        for t in turns:
            r = t.reasoning
            if not r:
                continue
            already = has_harness_leak(r)
            low = r.lower()
            for cue in _EXPLORE_CUES:
                if cue in low:
                    # only "new" if the frozen detector missed this reasoning
                    if not already:
                        explore_new[cue] += 1
                        explore_examples.setdefault(
                            cue, _ctx(r, cue))

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    print(f"file: {path}")
    print(f"records: {len(recs)}  turns: {n_turns}")
    print(f"FROZEN pre-filter hard-fails: {dict(hard_counts) or '{}'}")
    if leak_records:
        for ri, kinds in leak_records:
            print(f"   rec{ri}: {kinds}")
    print(f"means  reasoning_quality={mean(rq)}  "
          f"reasoning_answer_consistency={mean(rac)}  appropriateness={mean(appr)}")
    print(f"EXPLORATORY residual cues NOT caught by frozen detector: "
          f"{dict(explore_new) or '{}'}")
    for cue, ex in explore_examples.items():
        print(f"   [{cue}] {ex}")
    # exit non-zero if a frozen hard-fail is present in the emitted file
    return 1 if hard_counts else 0


def _ctx(text, cue, pad=40):
    i = text.lower().find(cue)
    s = max(0, i - pad)
    e = min(len(text), i + len(cue) + pad)
    return "..." + re.sub(r"\s+", " ", text[s:e]).strip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
