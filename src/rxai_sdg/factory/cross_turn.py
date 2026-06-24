"""Cross-turn (end-of-conversation) checks (spec §6.2).

These relational checks only make sense holistically and are implemented
programmatically via the ledger / constraint registry (no LLM):

* ``delayed_recall`` fidelity - the planted fact is present in the recall answer;
* ``standing`` adherence - a standing constraint is still satisfied in later turns;
* ``update_overwrite`` correctness - the latest value (not a stale one) is returned.
"""

from __future__ import annotations

from typing import Any

from .constraints import constraints_conflict
from .ledger import FactLedger
from .schemas import Turn
from .verifiers import ConstraintVerifier

# single-value budgets where a later instance REPLACES the earlier (not additive,
# unlike forbidden_token which accumulates).
_SINGLE_VALUE_TYPES = {"length_tokens", "length_words", "n_bullets",
                       "max_words_per_sentence", "first_letter"}


def run_cross_turn_checks(
    turns: list[Turn],
    ledger: FactLedger,
    verifier: ConstraintVerifier | None = None,
) -> dict[str, list[dict[str, Any]]]:
    verifier = verifier or ConstraintVerifier()
    results: dict[str, list[dict[str, Any]]] = {
        "delayed_recall": [],
        "standing": [],
        "update_overwrite": [],
    }

    for turn in turns:
        cs = turn.constraint_spec
        if cs is None:
            continue
        answer = turn.answer or ""

        # delayed-recall fidelity
        if cs.scope == "delayed_recall" and cs.fact_id is not None:
            try:
                passed, detail = ledger.recall_check(
                    answer, cs.fact_id, cs.params.get("match", "exact"))
            except KeyError:
                passed, detail = False, f"fact {cs.fact_id} not in ledger"
            results["delayed_recall"].append({
                "turn_index": turn.turn_index, "fact_id": cs.fact_id,
                "passed": passed, "detail": detail,
            })

        # update-overwrite correctness
        if cs.intent == "fact_update" and cs.fact_id is not None:
            res = verifier.verify(answer, cs)
            results["update_overwrite"].append({
                "turn_index": turn.turn_index, "fact_id": cs.fact_id,
                "passed": res.passed, "detail": res.detail,
            })

    # standing-instruction adherence: re-verify each standing constraint against
    # every later turn's answer - but stop once it is SUPERSEDED by a later
    # standing/cumulative constraint of a conflicting form (e.g. "always JSON"
    # replaces "always 3 bullets"); otherwise the replaced rule false-fails on every
    # subsequent turn.
    for turn in turns:
        cs = turn.constraint_spec
        if cs is None or cs.scope != "standing" or cs.verifier not in ("programmatic", "hybrid"):
            continue
        superseded_at = None
        for mid in turns:
            if mid.turn_index <= turn.turn_index:
                continue
            mcs = mid.constraint_spec
            # superseded by a later standing/cumulative constraint that either
            # CONFLICTS in form, or is a newer single-value budget of the SAME type
            # (a later "<=150 words" / "5 bullets" replaces the earlier limit).
            if mcs is not None and mcs.scope in ("standing", "cumulative") and (
                    constraints_conflict(mcs.type, cs.type)
                    or (mcs.type == cs.type and cs.type in _SINGLE_VALUE_TYPES)):
                superseded_at = mid.turn_index
                break
        for later in turns:
            if later.turn_index <= turn.turn_index:
                continue
            if superseded_at is not None and later.turn_index >= superseded_at:
                continue
            res = verifier.verify(later.answer or "", cs)
            results["standing"].append({
                "constraint_turn": turn.turn_index,
                "checked_turn": later.turn_index,
                "type": cs.type,
                "passed": res.passed,
                "detail": res.detail,
            })

    return results


def cross_turn_pass_rate(checks: dict[str, list[dict[str, Any]]]) -> float:
    """Overall fraction of cross-turn checks that passed (1.0 if none ran)."""
    total = passed = 0
    for entries in checks.values():
        for e in entries:
            total += 1
            passed += 1 if e.get("passed") else 0
    return 1.0 if total == 0 else passed / total
