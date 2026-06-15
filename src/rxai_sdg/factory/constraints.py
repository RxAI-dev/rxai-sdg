"""Constraint builders: turn a ``(intent, policy)`` draw into a concrete
machine-checkable ``constraint_spec``.

Builders are deliberately separate from *checkers* (the
:mod:`rxai_sdg.factory.verifiers` package) **and** from the user-facing query
text. A builder *produces* a ``constraint_spec`` with concrete, machine-chosen
parameters (e.g. letter ``"A"``, ``max_words=50``); a checker *verifies* an
answer against that spec; the **User-Simulator** (and, in production, the
instruct LLM it drives) is solely responsible for turning a spec into a natural
user message.

This module therefore emits **no user-facing query text** - only the
``constraint_spec``. Choosing parameters programmatically (rather than parsing
them back out of an LLM response) is what makes the whole pipeline
deterministically testable; keeping the phrasing out of here is what lets the
simulator produce diverse, natural, grounded turns instead of templates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .schemas import ConstraintSpec
from .taxonomy import POLICY_TO_SCOPE
from .ledger import NeedlePlanner


@dataclass
class BuildContext:
    rng: random.Random
    intent: str
    policy: str
    turn: int
    lang: str
    planner: NeedlePlanner
    min_recall_distance: int = 4


@dataclass
class BuildResult:
    #: The machine-checkable spec for the turn (``None`` only when the draw was
    #: unsatisfiable and ``resample`` is set). The simulator owns the query text.
    constraint_spec: Optional[ConstraintSpec]
    #: When True the draw could not be satisfied (e.g. no fact far enough in the
    #: past for a delayed recall); the simulator should resample the intent.
    resample: bool = False


# ---------------------------------------------------------------------------
# scope helpers
# ---------------------------------------------------------------------------

def _scope_for(policy: str) -> str:
    return POLICY_TO_SCOPE.get(policy, "current_turn")


def _make_spec(ctx: BuildContext, ctype: str, params: dict, verifier: str,
               fact_id: Optional[str] = None, planted_turn: Optional[int] = None) -> ConstraintSpec:
    scope = _scope_for(ctx.policy)
    return ConstraintSpec(
        intent=ctx.intent,
        type=ctype,
        params=params,
        lang=ctx.lang,
        verifier=verifier,
        scope=scope,
        applies_from_turn=ctx.turn if scope in ("standing", "cumulative") else None,
        planted_turn=planted_turn,
        fact_id=fact_id,
    )


# ---------------------------------------------------------------------------
# transformation builders
# ---------------------------------------------------------------------------

#: format id -> the verifier constraint type it maps onto.
_REFORMAT_FORMATS = {
    "json": "json_valid",
    "yaml": "yaml_valid",
    "table": "markdown_table",
    "md": "markdown_format",
}


def build_reformat(ctx: BuildContext) -> BuildResult:
    # Rigid forms (table/yaml) make sense as a one-off rewrite but are unrealistic
    # to enforce on *every* later turn; for standing/cumulative scopes we draw
    # only composable forms (json/markdown), matching the spec's "always JSON"
    # standing example.
    if ctx.policy in ("standing", "cumulative"):
        fmt = ctx.rng.choice(["json", "md"])
    else:
        fmt = ctx.rng.choice(list(_REFORMAT_FORMATS))
    ctype = _REFORMAT_FORMATS[fmt]
    params: dict[str, Any] = {"format": fmt}
    if fmt == "json":
        params["top_type"] = "object"
    return BuildResult(_make_spec(ctx, ctype, params, "programmatic"))


def build_lexical(ctx: BuildContext) -> BuildResult:
    # Positional / sentence-ordering constraints (first_letter,
    # alphabetical_sentence_starts) describe a single rewrite; persisting them
    # verbatim across an entire conversation is unrealistic. For standing /
    # cumulative scopes we therefore draw only compose-friendly sub-types that a
    # teacher can keep satisfying turn after turn (matches the spec's own
    # examples: reformat x standing, lexical x cumulative).
    compose_friendly: list[Callable[[BuildContext], tuple[str, dict]]] = [
        _lex_forbidden, _lex_no_pronouns, _lex_max_words,
    ]
    single_rewrite: list[Callable[[BuildContext], tuple[str, dict]]] = [
        _lex_first_letter, _lex_alphabetical,
    ]
    if ctx.policy in ("standing", "cumulative"):
        choices = compose_friendly
    else:
        choices = compose_friendly + single_rewrite
    ctype, params = ctx.rng.choice(choices)(ctx)
    return BuildResult(_make_spec(ctx, ctype, params, "programmatic"))


def _lex_first_letter(ctx: BuildContext) -> tuple[str, dict]:
    letter = ctx.rng.choice("ABCDFHILMOPRST")
    return "first_letter", {"letter": letter}


def _lex_max_words(ctx: BuildContext) -> tuple[str, dict]:
    n = ctx.rng.choice([8, 10, 12, 15])
    return "max_words_per_sentence", {"max_words": n}


def _lex_forbidden(ctx: BuildContext) -> tuple[str, dict]:
    token = ctx.rng.choice(["very", "thing", "really", "good", "important", "actually"])
    return "forbidden_token", {"token": token}


def _lex_no_pronouns(ctx: BuildContext) -> tuple[str, dict]:
    return "no_gendered_pronouns", {}


def _lex_alphabetical(ctx: BuildContext) -> tuple[str, dict]:
    return "alphabetical_sentence_starts", {}


def build_restyle(ctx: BuildContext) -> BuildResult:
    style = ctx.rng.choice(
        ["a friendly ELI5 tone", "a formal academic tone", "the persona of a pirate",
         "an enthusiastic marketing voice", "a terse expert tone"])
    return BuildResult(_make_spec(ctx, "style", {"style": style}, "llm_judge"))


def build_compress(ctx: BuildContext) -> BuildResult:
    # A fixed bullet count as a standing/cumulative rule is rigid; persist only
    # the loose word-budget form across turns.
    use_length = ctx.policy in ("standing", "cumulative") or ctx.rng.random() < 0.5
    if use_length:
        n = ctx.rng.choice([30, 50, 75])
        spec = _make_spec(ctx, "length_tokens", {"max_words": n}, "hybrid")
    else:
        n = ctx.rng.choice([3, 4, 5])
        spec = _make_spec(ctx, "n_bullets", {"n": n}, "hybrid")
    return BuildResult(spec)


def build_expand(ctx: BuildContext) -> BuildResult:
    # No machine-checkable constraint; quality handled by the LLM judge.
    return BuildResult(_make_spec(ctx, "expand", {}, "llm_judge"))


def build_genre_convert(ctx: BuildContext) -> BuildResult:
    genre = ctx.rng.choice(["limerick", "formal email", "code comment", "haiku"])
    # A rigid genre structure (e.g. a limerick) only makes sense as a one-off
    # rewrite; as a standing/cumulative rule it is judged holistically, not gated
    # programmatically on every turn. The chosen genre is kept in ``params`` so
    # the simulator can ask for it by name.
    if genre == "limerick" and ctx.policy not in ("standing", "cumulative"):
        spec = _make_spec(ctx, "limerick_structure", {"genre": genre}, "hybrid")
    else:
        spec = _make_spec(ctx, "genre", {"genre": genre}, "llm_judge")
    return BuildResult(spec)


# ---------------------------------------------------------------------------
# fact builders
# ---------------------------------------------------------------------------

def build_fact_recall(ctx: BuildContext) -> BuildResult:
    if ctx.policy == "delayed_recall":
        fact = ctx.planner.recallable_fact(ctx.turn, ctx.min_recall_distance)
        if fact is None:
            return BuildResult(None, resample=True)
        match = ctx.rng.choice(["exact", "fuzzy"])
        spec = _make_spec(
            ctx, "fact_recall", {"value": fact.value, "match": match}, "programmatic",
            fact_id=fact.fact_id, planted_turn=fact.planted_turn)
        return BuildResult(spec)
    # immediate: PLANT a new fact in this turn (no same-turn recall - the recall
    # is a separate, later delayed_recall turn). The plant turn is not gated on
    # the answer (the assistant just acknowledges), so it carries an llm_judge
    # spec that documents the fact without forcing the value into the answer.
    fact = ctx.planner.plant_fact(ctx.turn)
    spec = _make_spec(
        ctx, "fact_recall", {"value": fact.value, "match": "exact"}, "llm_judge",
        fact_id=fact.fact_id, planted_turn=fact.planted_turn)
    return BuildResult(spec)


def build_fact_update(ctx: BuildContext) -> BuildResult:
    fact = ctx.planner.recallable_fact(ctx.turn, 1) or ctx.planner.any_fact()
    if fact is None:
        # nothing to update yet -> plant first, then overwrite to a new value.
        fact = ctx.planner.plant_fact(ctx.turn)
    stale = list(ctx.planner.ledger.stale_values(fact.fact_id)) + [fact.value]
    ctx.planner.update_value_for(fact, ctx.turn)
    spec = _make_spec(
        ctx, "fact_update",
        {"value": fact.value, "match": "exact", "stale_values": stale},
        "programmatic", fact_id=fact.fact_id, planted_turn=fact.planted_turn)
    return BuildResult(spec)


# ---------------------------------------------------------------------------
# non-verifiable (LLM-judge / open) builders
# ---------------------------------------------------------------------------

def build_chained_compute(ctx: BuildContext) -> BuildResult:
    return BuildResult(_make_spec(ctx, "chained_compute", {}, "llm_judge"))


def build_self_critique(ctx: BuildContext) -> BuildResult:
    return BuildResult(_make_spec(ctx, "self_critique", {}, "llm_judge"))


def build_deepen(ctx: BuildContext) -> BuildResult:
    return BuildResult(_make_spec(ctx, "deepen", {}, "llm_judge"))


def build_open_chat(ctx: BuildContext) -> BuildResult:
    return BuildResult(_make_spec(ctx, "open_chat", {}, "llm_judge"))


#: intent id -> builder
BUILDERS: dict[str, Callable[[BuildContext], BuildResult]] = {
    "reformat": build_reformat,
    "lexical_constraint": build_lexical,
    "restyle": build_restyle,
    "compress": build_compress,
    "expand": build_expand,
    "genre_convert": build_genre_convert,
    "fact_recall": build_fact_recall,
    "fact_update": build_fact_update,
    "chained_compute": build_chained_compute,
    "self_critique": build_self_critique,
    "deepen": build_deepen,
    "open_chat": build_open_chat,
}


def build(ctx: BuildContext) -> BuildResult:
    builder = BUILDERS.get(ctx.intent)
    if builder is None:
        raise KeyError(f"no constraint builder for intent {ctx.intent!r}")
    return builder(ctx)
