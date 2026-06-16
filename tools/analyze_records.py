"""Automated defect detectors for generated conversation records (task §4).

Loads a JSONL file of serialised :class:`ConversationRecord` dicts and reports,
per defect class, the count and the offending ``(conversation_id, turn_index)``
pairs. A detector firing means that batch is **not clean**.

Usage::

    python tools/analyze_records.py path/to/records.jsonl
    python tools/analyze_records.py records.jsonl --json   # machine-readable

Exit code is non-zero when any detector fires (so it can gate an iteration).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# Make ``src`` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rxai_sdg.factory.responder import _MEMORY_DISCLAIMER_RE  # noqa: E402
from rxai_sdg.factory.seed_curator import CONTENTLESS_RE  # noqa: E402
from rxai_sdg.factory.taxonomy import (  # noqa: E402
    SENSITIVE_ALLOWED_INTENTS, TRANSFORM_CATEGORY_INTENTS,
)

# Spec-internal tokens that must never appear in any natural-language segment.
SPEC_LEAK_RE = re.compile(
    r"json_valid|top_type|forbidden_token|constraint_spec|standing instruction",
    re.IGNORECASE,
)

# Account / subscription / billing / system framing for a "personal" fact.
BAD_FACT_RE = re.compile(
    r"\b(account|subscription|billing|invoice|payment|password|api[ _-]?key|"
    r"access[ _-]?token|badge|membership|premium|plan\b|tier[- ]?\d|order[ _-]?number|"
    r"credit[ _-]?card|ssn|social security|license[ _-]?key)\b",
    re.IGNORECASE,
)

# User-turn phrasings that betray role confusion (fix E). Tuned to catch a user
# claiming authorship of the assistant's output, offering to do the assistant's
# job, or asking the assistant to pose a question - without flagging benign asks.
ROLE_CONFUSION_RES = [
    # "the table I made", "that summary I wrote" - claiming the assistant's output
    re.compile(r"\b(?:the|that|this)\s+(?:\w+\s+){0,2}i\s+(?:made|wrote|created|generated|produced|built|reformatted)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:rewrite|rephrasing|reformatted version|reformatted answer)\b", re.IGNORECASE),
    # offering to do the assistant's rewriting job
    re.compile(r"\bi(?:'ll| will| can|'d)\s+(?:redo|rewrite|rephrase|reformat|regenerate)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'ll| will| can|'d)\s+try\s+(?:again|rephrasing|rewriting|to rephrase|to rewrite|a rephrasing)\b", re.IGNORECASE),
    # inverting roles: asking the assistant to quiz / question the user
    re.compile(r"\b(?:ask me (?:a|your|the next)|pose (?:a|your|me a) question|give me your (?:next )?question|quiz me)\b", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
]

# Markers at least one of which a verifiable-constraint query must contain.
_CONSTRAINT_MARKERS: dict[str, list[str]] = {
    "json_valid": ["json"],
    "yaml_valid": ["yaml"],
    "markdown_table": ["table"],
    "markdown_format": ["markdown"],
    "alphabetical_sentence_starts": ["alphabet"],
    "no_gendered_pronouns": ["pronoun", "gender"],
    "limerick_structure": ["limerick"],
}


def _markers_for(cs: dict[str, Any]) -> list[str]:
    t = cs.get("type", "")
    p = cs.get("params", {}) or {}
    if t in _CONSTRAINT_MARKERS:
        return list(_CONSTRAINT_MARKERS[t])
    if t == "first_letter":
        L = str(p.get("letter", ""))
        return [f"'{L.lower()}'", f"letter {L.lower()}", f"start with {L.lower()}"]
    if t == "forbidden_token":
        tok = str(p.get("token", "")).lower()
        return [f"'{tok}'", tok, "the word"]
    if t == "max_words_per_sentence":
        return [str(p.get("max_words", "")), "per sentence", "short sentence"]
    if t == "length_tokens":
        return [str(p.get("max_words", "")), "word", "shorter", "concise", "compress", "trim"]
    if t == "n_bullets":
        return [str(p.get("n", "")), "bullet"]
    return []


def _seg(turn: dict, kind: str) -> str:
    for s in turn.get("segments", []):
        if s.get("segment_type") == kind:
            return s.get("text") or ""
    return ""


def _nl_segments(turn: dict) -> str:
    return " \n ".join(_seg(turn, k) for k in ("query", "reasoning", "answer"))


def _has_answer(turn: dict) -> bool:
    return bool(_seg(turn, "answer").strip())


def analyze(records: list[dict]) -> dict[str, Any]:
    """Run every detector over ``records``; return defects + summary stats."""
    defects: dict[str, list[Any]] = {k: [] for k in DETECTORS}
    coherences: list[int] = []

    for rec in records:
        cid = rec.get("conversation_id", "?")
        turns = rec.get("turns", [])
        seed = rec.get("source_seed", {})
        curation = (rec.get("cross_turn_checks") or {}).get("curation") or {}
        sensitive = curation.get("sensitivity") == "sensitive"
        holistic = rec.get("holistic_score") or {}
        if isinstance(holistic.get("coherence"), (int, float)):
            coherences.append(int(holistic["coherence"]))

        # -- contentless_seed_kept (per conversation) ----------------------
        fq = (seed.get("first_query") or "").strip()
        if CONTENTLESS_RE.match(fq) or len(fq) < 6:
            defects["contentless_seed_kept"].append((cid, 0))

        # -- transformation_density (per conversation) ---------------------
        followups = [t for t in turns if t.get("turn_index", 0) >= 1]
        if followups:
            n_tr = sum(1 for t in followups if t.get("intent") in TRANSFORM_CATEGORY_INTENTS)
            if n_tr / len(followups) > 0.60:
                defects["transformation_density"].append(
                    (cid, round(n_tr / len(followups), 2)))

        # -- judge_low (per conversation) ----------------------------------
        coh = holistic.get("coherence")
        appr = holistic.get("appropriateness")
        if (isinstance(coh, (int, float)) and coh < 6) or \
                (isinstance(appr, (int, float)) and appr < 7):
            defects["judge_low"].append((cid, {"coherence": coh, "appropriateness": appr}))

        prev_answer = None
        prior_text_acc = ""  # accumulated text of all *earlier* turns
        for t in turns:
            ti = t.get("turn_index", 0)
            query = _seg(t, "query")
            answer = _seg(t, "answer")
            reasoning = _seg(t, "reasoning")
            cs = t.get("constraint_spec") or {}
            intent = t.get("intent")
            ctype = cs.get("type")
            params = cs.get("params", {}) or {}

            # memory_disclaimer
            if _MEMORY_DISCLAIMER_RE.search(answer):
                defects["memory_disclaimer"].append((cid, ti))

            # reasoning_missing (answer-bearing turns must carry reasoning)
            if _has_answer(t) and not reasoning.strip():
                defects["reasoning_missing"].append((cid, ti))

            # spec_leak (natural-language segments only)
            if SPEC_LEAK_RE.search(query) or SPEC_LEAK_RE.search(reasoning) \
                    or SPEC_LEAK_RE.search(answer):
                defects["spec_leak"].append((cid, ti))

            # role_confusion (user turn only)
            if ti >= 1 and any(r.search(query) for r in ROLE_CONFUSION_RES):
                defects["role_confusion"].append((cid, ti))
            # judge signal: low role_consistency
            if ti == 0 and isinstance(holistic.get("role_consistency"), (int, float)) \
                    and holistic["role_consistency"] < 7:
                defects["role_confusion"].append((cid, "judge"))

            # sensitive_flippant
            if sensitive and intent in (
                    set(TRANSFORM_CATEGORY_INTENTS) | {"fact_recall", "fact_update", "chained_compute"}) \
                    and intent not in SENSITIVE_ALLOWED_INTENTS:
                defects["sensitive_flippant"].append((cid, ti, intent))

            # fact-related detectors
            if cs and cs.get("intent") in ("fact_recall", "fact_update"):
                value = str(params.get("value", "")).strip()
                planted = cs.get("planted_turn")
                scope = cs.get("scope")
                is_recall = cs.get("intent") == "fact_recall" and scope == "delayed_recall"
                # same_turn_fact
                if is_recall and planted == ti:
                    defects["same_turn_fact"].append((cid, ti))
                if is_recall and value and _norm_contains(query, value):
                    defects["same_turn_fact"].append((cid, ti))
                # phantom_stale (fact_update only)
                if cs.get("intent") == "fact_update":
                    for sv in (params.get("stale_values") or []):
                        if str(sv).strip() and not _norm_contains(prior_text_acc, str(sv)):
                            defects["phantom_stale"].append((cid, ti, sv))

            # bad_fact_type (per turn, on the constraint value/type)
            if cs.get("intent") in ("fact_recall", "fact_update"):
                if BAD_FACT_RE.search(str(params.get("value", ""))):
                    defects["bad_fact_type"].append((cid, ti, params.get("value")))

            # constraint_mismatch (verifiable transformation request)
            if cs.get("verifier") in ("programmatic", "hybrid") and ti >= 1 \
                    and cs.get("intent") not in ("fact_recall", "fact_update"):
                markers = _markers_for(cs)
                if markers and not any(m and m.lower() in query.lower() for m in markers):
                    defects["constraint_mismatch"].append((cid, ti, ctype))

            # identical_rewrite
            if intent in TRANSFORM_CATEGORY_INTENTS and prev_answer is not None \
                    and answer.strip() and answer.strip() == prev_answer.strip():
                defects["identical_rewrite"].append((cid, ti))

            prior_text_acc += " \n " + query + " \n " + answer
            prev_answer = answer

        # bad_fact_type also scans the fact ledger values/types
        for fct in rec.get("fact_ledger", []):
            blob = f"{fct.get('fact_type','')} {fct.get('value','')}"
            if BAD_FACT_RE.search(blob):
                defects["bad_fact_type"].append((cid, "ledger", fct.get("value")))

    summary = {
        "records": len(records),
        "total_turns": sum(len(r.get("turns", [])) for r in records),
        "median_coherence": (statistics.median(coherences) if coherences else None),
        "coherence_samples": len(coherences),
        "defect_counts": {k: len(v) for k, v in defects.items()},
        "clean": all(len(v) == 0 for v in defects.values()),
    }
    return {"summary": summary, "defects": defects}


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", (s or "").lower()).strip()


def _norm_contains(haystack: str, needle: str) -> bool:
    n = _norm(needle)
    return bool(n) and n in _norm(haystack)


DETECTORS = [
    "memory_disclaimer", "reasoning_missing", "spec_leak", "same_turn_fact",
    "phantom_stale", "bad_fact_type", "role_confusion", "constraint_mismatch",
    "transformation_density", "contentless_seed_kept", "sensitive_flippant",
    "identical_rewrite", "judge_low",
]


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Defect detectors for generated records")
    ap.add_argument("path", help="JSONL file of conversation records")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--show", type=int, default=4, help="offenders to show per defect")
    args = ap.parse_args(argv)

    records = load_jsonl(args.path)
    result = analyze(records)
    if args.json:
        print(json.dumps(result, default=str, indent=2))
        return 0 if result["summary"]["clean"] else 1

    s = result["summary"]
    print(f"records={s['records']} turns={s['total_turns']} "
          f"median_coherence={s['median_coherence']} (n={s['coherence_samples']})")
    print("-" * 60)
    for name in DETECTORS:
        offenders = result["defects"][name]
        flag = "OK " if not offenders else "XX "
        print(f"{flag}{name:24s} {len(offenders)}")
        for off in offenders[:args.show]:
            print(f"      - {off}")
    print("-" * 60)
    print("CLEAN" if s["clean"] else "NOT CLEAN")
    return 0 if s["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
