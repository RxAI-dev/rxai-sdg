"""The follow-up taxonomy (spec §4).

The design principle is that *what* a follow-up asks for (the **base intent**)
is orthogonal to *the memory distance* at which it operates (the **distance
policy**). Memory stress comes from the distance, not the intent. The sampler
therefore draws a cross-product ``(base_intent x distance_policy)`` subject to an
**invalidity mask** encoded here as data, not as hardcoded ``if`` statements.

Everything in this module is plain data so it can be overridden from config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Axis 1 - base intents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseIntent:
    id: str
    description: str
    weight: float
    verification: str  # "programmatic" | "hybrid" | "llm_judge"
    capability: str


# Default weights sum to 100 (spec §4.1).
BASE_INTENTS: dict[str, BaseIntent] = {
    "reformat": BaseIntent(
        "reformat", "Reformat prior answer (JSON/YAML/table/markdown)",
        12, "programmatic", "transformation"),
    "lexical_constraint": BaseIntent(
        "lexical_constraint", "Lexical/structural constraint",
        14, "programmatic", "transformation"),
    "restyle": BaseIntent(
        "restyle", "Restyle (tone, ELI5, persona, formality)",
        8, "llm_judge", "transformation"),
    "compress": BaseIntent(
        "compress", "Compress (N bullets / N words / TL;DR)",
        8, "hybrid", "transformation"),
    "expand": BaseIntent(
        "expand", "Expand (more detail/examples)",
        5, "llm_judge", "transformation"),
    "genre_convert": BaseIntent(
        "genre_convert", "Genre/format conversion (limerick, email, code comment)",
        8, "hybrid", "transformation"),
    "fact_recall": BaseIntent(
        "fact_recall", "Recall a fact stated earlier",
        12, "programmatic", "recall"),
    "fact_update": BaseIntent(
        "fact_update", "Ask for the current value of an updated/overwritten fact",
        9, "programmatic", "memory_update"),
    "chained_compute": BaseIntent(
        "chained_compute", "Next computation/logic step",
        8, "llm_judge", "reasoning"),
    "self_critique": BaseIntent(
        "self_critique", "Evaluate / correct own answer",
        6, "llm_judge", "stem"),
    "deepen": BaseIntent(
        "deepen", "Deepen / example for a prior point",
        5, "llm_judge", "coherence"),
    "open_chat": BaseIntent(
        "open_chat", "Open conversational (advice, empathy, roleplay)",
        5, "llm_judge", "breadth"),
}

#: Intents that perform a content transformation of the prior answer; these are
#: valid with all four distance policies. (Used for *grounding* classification -
#: an intent here operates on the prior answer.)
TRANSFORMATION_INTENTS = frozenset(
    {"reformat", "lexical_constraint", "restyle", "compress", "expand", "genre_convert"}
)

#: Intents that operate against the fact ledger.
FACT_INTENTS = frozenset({"fact_recall", "fact_update"})


# ---------------------------------------------------------------------------
# Conversation-composition categories (fix B)
# ---------------------------------------------------------------------------
#
# A conversation is a topical thread, not a stack of format conversions. The
# per-conversation planner allocates a balanced mix across three categories; the
# sampler then draws an intent *within* the category chosen for each turn.
#
#  * ``explore``   - topical follow-ups / deepening (the dominant ~50%).
#  * ``transform`` - reformat / lexical / restyle / compress / genre (~30%).
#  * ``memory``    - recall-of-prior-content and (occasional) fact plant/recall (~20%).
#
# NB: ``expand`` is an *exploration* intent (it adds detail to the topic) even
# though it also operates on the prior answer for grounding purposes.

COMPOSITION_CATEGORIES: dict[str, frozenset[str]] = {
    "explore": frozenset(
        {"deepen", "expand", "chained_compute", "self_critique", "open_chat"}),
    "transform": frozenset(
        {"reformat", "lexical_constraint", "restyle", "compress", "genre_convert"}),
    "memory": frozenset({"fact_recall", "fact_update"}),
}

#: intent id -> composition category (the inverse of COMPOSITION_CATEGORIES).
INTENT_TO_CATEGORY: dict[str, str] = {
    intent: cat
    for cat, intents in COMPOSITION_CATEGORIES.items()
    for intent in intents
}

#: Intents counted as "transformations" by the transformation-density detector
#: and capped by the planner (matches the spec's reformat/lexical/restyle/
#: compress/genre wording - ``expand`` is exploration, not a transformation).
TRANSFORM_CATEGORY_INTENTS = COMPOSITION_CATEGORIES["transform"]

#: For a *sensitive* seed the sampler is restricted to this safe, supportive
#: subset (fix A): deepen / expand / compress / open_chat / self_critique in a
#: supportive register. Trivializing transformations (restyle to pirate/marketing,
#: genre conversion to limerick/haiku, format gymnastics, fact-planting) are
#: forbidden. ``recall_content`` (a supportive recall of real prior content) is
#: always permitted and is not a taxonomy intent.
SENSITIVE_ALLOWED_INTENTS: frozenset[str] = frozenset(
    {"deepen", "expand", "compress", "open_chat", "self_critique"})


# ---------------------------------------------------------------------------
# Axis 2 - memory-distance policies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistancePolicy:
    id: str
    description: str
    weight: float
    stresses: str


# Default weights sum to 100 (spec §4.2). The three memory-stressing policies
# (cumulative + standing + delayed_recall = 70) are the differentiator;
# ``immediate`` at 30 preserves the depth-1 capability we already have.
DISTANCE_POLICIES: dict[str, DistancePolicy] = {
    "immediate": DistancePolicy(
        "immediate", "Applies to current / just-prior turn",
        30, "base transformation, short recall (preserve, don't regress)"),
    "cumulative": DistancePolicy(
        "cumulative", "Adds a constraint while all prior constraints remain enforced",
        25, "constraint accumulation"),
    "standing": DistancePolicy(
        "standing", "Issued once, enforced across all remaining turns",
        20, "instruction retention"),
    "delayed_recall": DistancePolicy(
        "delayed_recall", "Plant at turn k, query at turn k+D (D>=4)",
        25, "long-range recall"),
}

#: Distance policies that map onto a ``constraint_spec.scope`` value of the same
#: name. ``immediate`` maps to the ``current_turn`` scope.
POLICY_TO_SCOPE: dict[str, str] = {
    "immediate": "current_turn",
    "cumulative": "cumulative",
    "standing": "standing",
    "delayed_recall": "delayed_recall",
}


# ---------------------------------------------------------------------------
# Invalidity mask (encoded as data)
# ---------------------------------------------------------------------------

#: Set of ``(intent_id, policy_id)`` pairs that are *invalid* and must be
#: resampled. Encoded as data so it can be edited / extended from config without
#: touching sampler logic. Guidance from spec §4.3:
#:
#: * ``fact_recall``/``fact_update`` are only valid with ``immediate`` or
#:   ``delayed_recall`` (recall has no cumulative/standing meaning).
#: * ``chained_compute`` is invalid with ``standing``.
#: * ``open_chat``/``deepen`` default to ``immediate`` only.
#: * all transformation intents are valid with all four policies.
def default_invalid_pairs() -> set[tuple[str, str]]:
    invalid: set[tuple[str, str]] = set()

    # Fact intents: only immediate / delayed_recall.
    for intent in FACT_INTENTS:
        for policy in ("cumulative", "standing"):
            invalid.add((intent, policy))

    # chained_compute cannot be a standing instruction.
    invalid.add(("chained_compute", "standing"))

    # open_chat / deepen default to immediate only.
    for intent in ("open_chat", "deepen"):
        for policy in ("cumulative", "standing", "delayed_recall"):
            invalid.add((intent, policy))

    return invalid


@dataclass
class Taxonomy:
    """Bundles the two axes plus the invalidity mask.

    A fresh :class:`Taxonomy` carries the spec defaults; the
    :class:`~rxai_sdg.factory.config.FactoryConfig` overlays user weight/mask
    overrides on top of it.
    """

    base_intents: dict[str, BaseIntent] = field(default_factory=lambda: dict(BASE_INTENTS))
    distance_policies: dict[str, DistancePolicy] = field(
        default_factory=lambda: dict(DISTANCE_POLICIES))
    invalid_pairs: set[tuple[str, str]] = field(default_factory=default_invalid_pairs)

    def is_valid(self, intent: str, policy: str) -> bool:
        return (intent, policy) not in self.invalid_pairs

    def valid_policies_for(self, intent: str) -> list[str]:
        return [p for p in self.distance_policies if self.is_valid(intent, p)]

    def has_any_valid_pair(self, intent: str) -> bool:
        return bool(self.valid_policies_for(intent))
