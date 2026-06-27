#!/usr/bin/env python3
"""Adversarial reader: read each accepted conversation and flag *self-contained*
defects with code -- the job a careful human (or Claude Chat) does by hand.

This is the metric that replaces judge pass-rate. The judge cannot see these
classes; every check here is decidable from the transcript text alone (or from
a closed-world fact table), so a flag is a defect, not an opinion.

Usage:
    python tools/adversarial_reader.py run100.jsonl
    python tools/adversarial_reader.py run100.jsonl --show D1 --limit 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the existing deterministic machinery rather than re-implement it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rxai_sdg.factory import detectors as _det  # noqa: E402
from rxai_sdg.factory import exec_gate as _exec  # noqa: E402


@dataclass
class Finding:
    conv: str
    turn: int
    cls: str          # defect class, e.g. "D1-meta"
    evidence: str
    severity: int = 1


# --------------------------------------------------------------------------- io
def segs(turn: dict, kind: str) -> list[str]:
    return [s.get("text", "") for s in (turn.get("segments") or [])
            if s.get("segment_type") == kind]


def first(turn: dict, kind: str) -> str:
    xs = segs(turn, kind)
    return xs[0] if xs else ""


# ------------------------------------------------------------------- D1 voice
# Two tiers. HARD = unambiguous annotator/harness narration (the reasoning is
# modelling the *annotation task* or compliance, not the problem). SOFT =
# stylistic third-person task framing that is common in genuine CoT but still
# reads as not-in-character; reported separately so the number is honest.
_D1_HARD = [
    re.compile(r"\b(?:we|i)\s+must\s+comply\b", re.I),
    re.compile(r"\blet'?s\s+comply\b", re.I),
    re.compile(r"\bcompl(?:y|ies|ying)\s+with\s+the\s+(?:request|instruction|user)", re.I),
    re.compile(r"\bno\s+(?:special\s+)?formatting\s+(?:constraints?|requirements?)?\s*"
               r"(?:given|needed|required|requested|specified|provided)\b", re.I),
    re.compile(r"\bno\s+(?:explicit\s+)?constraints?\s+(?:given|specified|provided)\b", re.I),
    re.compile(r"\bwithout\s+(?:anyone|the\s+user)\s+knowing\b", re.I),
    re.compile(r"\bas\s+an?\s+(?:ai|assistant|language\s+model)\b", re.I),
    re.compile(r"\bthe\s+user\s+wants\s+us\b", re.I),
    re.compile(r"\bwe\s+are\s+asked\s+to\b", re.I),
]
_D1_SOFT = [
    re.compile(r"\bwe\s+need\s+to\s+(?:respond|answer|reply|provide|produce|write)\b", re.I),
    re.compile(r"\bwe\s+should\s+(?:respond|answer|reply|provide)\b", re.I),
    re.compile(r"\bthe\s+user\s+(?:wants|is\s+asking|asks|asked|requested)\b", re.I),
    re.compile(r"\bthey\s+want\s+us\s+to\b", re.I),
    re.compile(r"\bthe\s+request\s+is\s+to\b", re.I),
]


def check_reasoning_voice(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    for t in rec.get("turns", []):
        r = first(t, "reasoning")
        if not r.strip():
            continue
        ti = t.get("turn_index", -1)
        hard = next((rx.search(r) for rx in _D1_HARD if rx.search(r)), None)
        if hard:
            out.append(Finding(cid, ti, "D1-meta", hard.group(0)[:60], 2))
        else:
            soft = next((rx.search(r) for rx in _D1_SOFT if rx.search(r)), None)
            if soft:
                out.append(Finding(cid, ti, "D1-soft", soft.group(0)[:60], 1))
    return out


# ----------------------------------------------------------- crisis numbers
# Closed-world table of real helpline numbers. A number printed next to one of
# these org names that is NOT the canonical value is a fabricated/malformed
# safety datum -- the worst kind of self-contained factual error.
_HELPLINES = {
    "samaritans": {"116 123", "116123"},
    "988":        {"988"},          # US Suicide & Crisis Lifeline
    "lifeline australia": {"13 11 14", "131114"},
    "crisis text line": {"741741"},
}
_NUM_NEAR = re.compile(r"([+(]?\d[\d ()‑.\-]{5,}\d)")


def _norm_num(s: str) -> str:
    return re.sub(r"[^\d]", "", s.replace("‑", "-"))


def check_crisis_numbers(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    for t in rec.get("turns", []):
        a = first(t, "answer")
        low = a.lower()
        for org, canon in _HELPLINES.items():
            if org in ("988",):
                continue
            idx = low.find(org)
            if idx < 0:
                continue
            window = a[idx: idx + 80]
            canon_norm = {_norm_num(c) for c in canon}
            for m in _NUM_NEAR.finditer(window):
                num = m.group(1).strip()
                if num.isdigit() and len(num) <= 3:
                    continue  # section numbers etc.
                if _norm_num(num) not in canon_norm and len(_norm_num(num)) >= 4:
                    out.append(Finding(
                        cid, t.get("turn_index", -1), "D4-crisis",
                        f"{org!r} -> {num!r} (canonical {sorted(canon)})", 3))
    return out


# ------------------------------------------------------- JSON newline round-trip
# A limerick/poem/haiku stored in JSON as "line1\\nline2" is *valid JSON* but
# parses to a string containing a literal backslash-n, so it renders as one line
# with a visible \n. json_valid gates pass it; this checks the parsed semantics.
_POETIC_KEY = re.compile(r"limerick|poem|verse|haiku|stanza|lyric|sonnet|line", re.I)


def _walk_json(obj, keypath, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_json(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, keypath, out)
    elif isinstance(obj, str):
        out.append((keypath or "", obj))


def check_json_newlines(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    for t in rec.get("turns", []):
        a = first(t, "answer")
        for blob in _exec._json_candidates(a) if hasattr(_exec, "_json_candidates") else []:
            pairs: list[tuple[str, str]] = []
            _walk_json(blob, "", pairs)
            for key, val in pairs:
                if _POETIC_KEY.search(key) and "\\n" in val:
                    out.append(Finding(
                        cid, t.get("turn_index", -1), "D5-json-newline",
                        f"key {key!r} parses to literal backslash-n: {val[:50]!r}", 2))
    return out


# --------------------------------------------------- structural-claim (enjambment)
_ENJAMB = re.compile(r"\benjamb(?:ed|ment|ments)?\b|\brun[- ]on lines\b|"
                     r"lines?\s+(?:run|flow|carry|spill)\s+(?:on|over|into)", re.I)
_END_STOP = re.compile(r"[.!?;:,—\"')\]]\s*$")


def _poem_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(("#", "|", "```", ">", "-", "*"))]
    # poem-ish: several short lines
    short = [ln for ln in lines if len(ln.split()) <= 12]
    return short if len(short) >= 3 else []


def check_structural_claims(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    turns = rec.get("turns", [])
    for i, t in enumerate(turns):
        a = first(t, "answer")
        if not _ENJAMB.search(a):
            continue
        # find the most recent poem-like artifact at or before this turn
        for j in range(i, -1, -1):
            for kind in ("answer", "query"):
                lines = _poem_lines(first(turns[j], kind))
                if not lines:
                    continue
                end_stopped = sum(1 for ln in lines if _END_STOP.search(ln))
                if end_stopped == len(lines):
                    out.append(Finding(
                        cid, t.get("turn_index", -1), "D5-struct-claim",
                        f"claims enjambment but all {len(lines)} lines are end-stopped", 2))
                break
            else:
                continue
            break
    return out


# ------------------------------------------------------- needle from example
_HEDGE = re.compile(r"\b(?:like|such as|e\.?g\.?|for example|for instance|things? like|"
                    r"places? like|stuff like|imagine|say,?)\b", re.I)


def check_needle_source(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    turns = rec.get("turns", [])
    by_idx = {t.get("turn_index"): t for t in turns}
    for fact in rec.get("fact_ledger", []) or []:
        pt = fact.get("planted_turn")
        val = str(fact.get("value", "")).strip()
        if not val or pt not in by_idx:
            continue
        q = first(by_idx[pt], "query")
        m = re.search(re.escape(val), q, re.I)
        if not m:
            continue
        pre = q[max(0, m.start() - 40): m.start()]
        if _HEDGE.search(pre):
            out.append(Finding(
                cid, pt, "needle-hedged",
                f"fact {fact.get('fact_id')}={val!r} planted via hedge: ...{pre[-30:]!r}", 2))
    return out


# ------------------------------------------------- reuse existing det/exec gates
def check_existing(rec: dict) -> list[Finding]:
    out: list[Finding] = []
    cid = rec.get("conversation_id", "?")[:8]
    turns = rec.get("turns", [])
    try:
        for f in _det.run_all({"turns": turns, "first_query":
                               first(turns[0], "query") if turns else ""}, run_code=True):
            out.append(Finding(cid, f.turn_index if f.turn_index is not None else -1,
                               f"det-{f.detector}-{f.name}", str(f.evidence)[:60],
                               f.severity))
    except Exception as e:  # noqa: BLE001
        out.append(Finding(cid, -1, "det-error", str(e)[:60]))
    try:
        eg = _exec.run_exec_gate(turns, run_code=True)
        for fl in eg.all_flags():
            out.append(Finding(cid, getattr(fl, "turn_index", -1),
                               f"exec-{getattr(fl,'kind',getattr(fl,'name','?'))}",
                               str(getattr(fl, "detail", getattr(fl, "evidence", "")))[:60],
                               getattr(fl, "severity", 1)))
    except Exception as e:  # noqa: BLE001
        out.append(Finding(cid, -1, "exec-error", str(e)[:60]))
    return out


CHECKS = [
    check_reasoning_voice,
    check_crisis_numbers,
    check_json_newlines,
    check_structural_claims,
    check_needle_source,
    check_existing,
]


def audit(path: Path):
    recs = [json.loads(l) for l in path.open()]
    all_find: list[Finding] = []
    convs_with: set[str] = set()
    for rec in recs:
        rec_find: list[Finding] = []
        for chk in CHECKS:
            rec_find += chk(rec)
        all_find += rec_find
        if any(f.cls not in ("D1-soft",) for f in rec_find):
            convs_with.add(rec.get("conversation_id", "?")[:8])
    return recs, all_find


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--show", default=None, help="print findings whose class startswith this")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    recs, finds = audit(args.path)
    by_cls = Counter(f.cls for f in finds)
    convs = len(recs)
    # conversations touched by each class
    cls_convs: dict[str, set] = {}
    for f in finds:
        cls_convs.setdefault(f.cls, set()).add(f.conv)

    print(f"\n=== adversarial reader: {args.path.name} ({convs} accepted convs) ===\n")
    print(f"{'class':<26}{'findings':>9}{'convs':>8}{'% convs':>9}")
    print("-" * 52)
    for cls, n in sorted(by_cls.items(), key=lambda kv: -kv[1]):
        nc = len(cls_convs[cls])
        print(f"{cls:<26}{n:>9}{nc:>8}{100*nc/convs:>8.0f}%")

    hard = {f.conv for f in finds if f.severity >= 2}
    print("-" * 52)
    print(f"convs with >=1 severity>=2 defect: {len(hard)}/{convs} "
          f"({100*len(hard)/convs:.0f}%)")

    if args.show:
        print(f"\n--- findings matching {args.show!r} ---")
        shown = [f for f in finds if f.cls.startswith(args.show)]
        for f in shown[:args.limit]:
            print(f"  [{f.conv} t{f.turn} sev{f.severity}] {f.cls}: {f.evidence}")
        print(f"  ({len(shown)} total)")


if __name__ == "__main__":
    main()
