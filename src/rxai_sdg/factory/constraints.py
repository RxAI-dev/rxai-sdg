"""Constraint builders: turn a ``(intent, policy)`` draw into a concrete
``constraint_spec`` plus a natural-language follow-up phrasing.

Builders are deliberately separate from *checkers* (the
:mod:`rxai_sdg.factory.verifiers` package). A builder *produces* a
``constraint_spec`` with concrete, machine-chosen parameters (e.g. letter
``"A"``, ``max_words=50``); a checker *verifies* an answer against that spec.
Choosing parameters programmatically (rather than parsing them back out of an
LLM response) is what makes the whole pipeline deterministically testable.
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
    nl_query: str
    constraint_spec: Optional[ConstraintSpec]
    #: When True the draw could not be satisfied (e.g. no fact far enough in the
    #: past for a delayed recall); the simulator should resample the intent.
    resample: bool = False


# ---------------------------------------------------------------------------
# scope / phrasing helpers
# ---------------------------------------------------------------------------

def _scope_for(policy: str) -> str:
    return POLICY_TO_SCOPE.get(policy, "current_turn")


def _policy_prefix(policy: str) -> str:
    """A natural-language prefix that conveys the memory-distance policy."""
    if policy == "cumulative":
        return ("Keep all the constraints you've been following so far, and "
                "additionally ")
    if policy == "standing":
        return "For the rest of this conversation, always "
    return ""


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

_REFORMAT_FORMATS = {
    "json": ("json_valid", "as a single valid JSON object"),
    "yaml": ("yaml_valid", "as valid YAML"),
    "table": ("markdown_table", "as a markdown table"),
    "md": ("markdown_format", "using markdown formatting (headings and bullets)"),
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
    ctype, phrase = _REFORMAT_FORMATS[fmt]
    params: dict[str, Any] = {"format": fmt}
    if fmt == "json":
        params["top_type"] = "object"
    verb = "answer " if ctx.policy == "standing" else "reformat your previous answer "
    nl = f"{_policy_prefix(ctx.policy)}{verb}{phrase}."
    return BuildResult(nl.strip(), _make_spec(ctx, ctype, params, "programmatic"))


def build_lexical(ctx: BuildContext) -> BuildResult:
    # Positional / sentence-ordering constraints (first_letter,
    # alphabetical_sentence_starts) describe a single rewrite; persisting them
    # verbatim across an entire conversation is unrealistic. For standing /
    # cumulative scopes we therefore draw only compose-friendly sub-types that a
    # teacher can keep satisfying turn after turn (matches the spec's own
    # examples: reformat x standing, lexical x cumulative).
    compose_friendly: list[Callable[[BuildContext], tuple[str, dict, str]]] = [
        _lex_forbidden, _lex_no_pronouns, _lex_max_words,
    ]
    single_rewrite: list[Callable[[BuildContext], tuple[str, dict, str]]] = [
        _lex_first_letter, _lex_alphabetical,
    ]
    if ctx.policy in ("standing", "cumulative"):
        choices = compose_friendly
    else:
        choices = compose_friendly + single_rewrite
    ctype, params, phrase = ctx.rng.choice(choices)(ctx)
    nl = f"{_policy_prefix(ctx.policy)}{phrase}".strip()
    if not nl.endswith("."):
        nl += "."
    return BuildResult(nl, _make_spec(ctx, ctype, params, "programmatic"))


def _lex_first_letter(ctx: BuildContext) -> tuple[str, dict, str]:
    letter = ctx.rng.choice("ABCDFHILMOPRST")
    return ("first_letter", {"letter": letter},
            f"rewrite the answer so every sentence starts with the letter '{letter}'")


def _lex_max_words(ctx: BuildContext) -> tuple[str, dict, str]:
    n = ctx.rng.choice([8, 10, 12, 15])
    return ("max_words_per_sentence", {"max_words": n},
            f"keep every sentence to at most {n} words")


def _lex_forbidden(ctx: BuildContext) -> tuple[str, dict, str]:
    token = ctx.rng.choice(["very", "thing", "really", "good", "important", "actually"])
    return ("forbidden_token", {"token": token},
            f"rewrite the answer without ever using the word '{token}'")


def _lex_no_pronouns(ctx: BuildContext) -> tuple[str, dict, str]:
    return ("no_gendered_pronouns", {},
            "rewrite the answer using no gendered pronouns (he/she/him/her/his/hers)")


def _lex_alphabetical(ctx: BuildContext) -> tuple[str, dict, str]:
    return ("alphabetical_sentence_starts", {},
            "rewrite the answer so the first letters of consecutive sentences go in "
            "alphabetical order")


def build_restyle(ctx: BuildContext) -> BuildResult:
    style = ctx.rng.choice(
        ["a friendly ELI5 tone", "a formal academic tone", "the persona of a pirate",
         "an enthusiastic marketing voice", "a terse expert tone"])
    nl = f"{_policy_prefix(ctx.policy)}restyle the answer in {style}.".strip()
    spec = _make_spec(ctx, "style", {"style": style}, "llm_judge")
    return BuildResult(nl, spec)


def build_compress(ctx: BuildContext) -> BuildResult:
    # A fixed bullet count as a standing/cumulative rule is rigid; persist only
    # the loose word-budget form across turns.
    use_length = ctx.policy in ("standing", "cumulative") or ctx.rng.random() < 0.5
    if use_length:
        n = ctx.rng.choice([30, 50, 75])
        nl = f"{_policy_prefix(ctx.policy)}compress the answer to at most {n} words.".strip()
        spec = _make_spec(ctx, "length_tokens", {"max_words": n}, "hybrid")
    else:
        n = ctx.rng.choice([3, 4, 5])
        nl = f"{_policy_prefix(ctx.policy)}compress the answer into exactly {n} bullet points.".strip()
        spec = _make_spec(ctx, "n_bullets", {"n": n}, "hybrid")
    return BuildResult(nl, spec)


def build_expand(ctx: BuildContext) -> BuildResult:
    nl = f"{_policy_prefix(ctx.policy)}expand the answer with more detail and a concrete example.".strip()
    # No machine-checkable constraint; quality handled by the LLM judge.
    spec = _make_spec(ctx, "expand", {}, "llm_judge")
    return BuildResult(nl, spec)


def build_genre_convert(ctx: BuildContext) -> BuildResult:
    genre = ctx.rng.choice(["limerick", "formal email", "code comment", "haiku"])
    # A rigid genre structure (e.g. a limerick) only makes sense as a one-off
    # rewrite; as a standing/cumulative rule it is judged holistically, not gated
    # programmatically on every turn.
    if genre == "limerick" and ctx.policy not in ("standing", "cumulative"):
        spec = _make_spec(ctx, "limerick_structure", {}, "hybrid")
    else:
        spec = _make_spec(ctx, "genre", {"genre": genre}, "llm_judge")
    nl = f"{_policy_prefix(ctx.policy)}rewrite the answer as a {genre}.".strip()
    return BuildResult(nl, spec)


# ---------------------------------------------------------------------------
# fact builders
# ---------------------------------------------------------------------------

def build_fact_recall(ctx: BuildContext) -> BuildResult:
    if ctx.policy == "delayed_recall":
        fact = ctx.planner.recallable_fact(ctx.turn, ctx.min_recall_distance)
        if fact is None:
            return BuildResult("", None, resample=True)
        match = ctx.rng.choice(["exact", "fuzzy"])
        spec = _make_spec(
            ctx, "fact_recall", {"value": fact.value, "match": match}, "programmatic",
            fact_id=fact.fact_id, planted_turn=fact.planted_turn)
        return BuildResult(ctx.planner.recall_question(fact), spec)
    # immediate: plant a fact in this very query and ask the teacher to echo it.
    fact = ctx.planner.plant_fact(ctx.turn)
    plant = ctx.planner.plant_phrasing(fact)
    question = ctx.planner.recall_question(fact)
    spec = _make_spec(
        ctx, "fact_recall", {"value": fact.value, "match": "exact"}, "programmatic",
        fact_id=fact.fact_id, planted_turn=fact.planted_turn)
    return BuildResult(f"{plant} {question}", spec)


def build_fact_update(ctx: BuildContext) -> BuildResult:
    fact = ctx.planner.recallable_fact(ctx.turn, 1) or ctx.planner.any_fact()
    if fact is None:
        # plant first, then immediately overwrite and query the current value.
        fact = ctx.planner.plant_fact(ctx.turn)
        plant = ctx.planner.plant_phrasing(fact)
    else:
        plant = ""
    stale = list(ctx.planner.ledger.stale_values(fact.fact_id)) + [fact.value]
    ctx.planner.update_value_for(fact, ctx.turn)
    update = ctx.planner.update_phrasing(fact)
    question = ctx.planner.recall_question(fact)
    spec = _make_spec(
        ctx, "fact_update",
        {"value": fact.value, "match": "exact", "stale_values": stale},
        "programmatic", fact_id=fact.fact_id, planted_turn=fact.planted_turn)
    nl = " ".join(p for p in [plant, update, question] if p)
    return BuildResult(nl, spec)


# ---------------------------------------------------------------------------
# non-verifiable (LLM-judge / open) builders
# ---------------------------------------------------------------------------

def build_chained_compute(ctx: BuildContext) -> BuildResult:
    nl = "Building on that, what's the next logical step or computation? Show your work."
    return BuildResult(nl, _make_spec(ctx, "chained_compute", {}, "llm_judge"))


def build_self_critique(ctx: BuildContext) -> BuildResult:
    nl = "Critique your previous answer: what could be wrong or improved, and fix it."
    return BuildResult(nl, _make_spec(ctx, "self_critique", {}, "llm_judge"))


def build_deepen(ctx: BuildContext) -> BuildResult:
    nl = "Go deeper on one key point from your last answer and give a concrete example."
    return BuildResult(nl, _make_spec(ctx, "deepen", {}, "llm_judge"))


def build_open_chat(ctx: BuildContext) -> BuildResult:
    nl = ctx.rng.choice([
        "Honestly, that's a lot to take in. How would you advise someone feeling overwhelmed by it?",
        "Let's switch gears - what's your take on why this matters to everyday people?",
        "Can you role-play explaining this to a curious ten-year-old?",
    ])
    return BuildResult(nl, _make_spec(ctx, "open_chat", {}, "llm_judge"))


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
