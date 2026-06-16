"""Per-conversation composition planner (fix B).

A conversation is a **topical thread**, not a stack of format conversions. Before
the loop runs, :func:`plan_conversation` allocates a balanced category mix across
the follow-up turns and enforces topical continuity:

* ~50% **exploration / deepening** - natural topical follow-ups (the dominant share);
* ~30% **transformations** - reformat / lexical / restyle / compress / genre;
* ~20% **memory tests** - recall of real prior content by default, with an
  occasional, topically-woven fact plant recalled at a later turn.

Transformation density is hard-capped (default 60%, the detector threshold; the
planner targets 30%) so no conversation degenerates into format gymnastics. The
ratios are config-driven.

The planner also schedules the memory sub-kind per memory turn (``content`` /
``plant`` / ``recall`` / ``update``) so that a *plant* always precedes its
*recall* by at least ``min_recall_distance`` turns - the simulator never has to
invent an artificial same-turn plant+recall, and recalls only fire when a real
fact was planted earlier. For *sensitive* seeds the memory category is restricted
to ``content`` (supportive recall of prior content) - no personal-fact planting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class TurnPlan:
    """The planned shape of one follow-up turn."""

    #: composition category: ``explore`` | ``transform`` | ``memory``
    category: str
    #: for memory turns only: ``content`` | ``plant`` | ``recall`` | ``update``
    memory_kind: Optional[str] = None


@dataclass
class CompositionRatios:
    """Target category mix (fractions need not sum to exactly 1)."""

    explore: float = 0.5
    transform: float = 0.3
    memory: float = 0.2
    #: hard cap on the fraction of follow-up turns that may be transformations.
    max_transform: float = 0.6


def plan_conversation(
    target_length: int,
    rng: random.Random,
    ratios: Optional[CompositionRatios] = None,
    min_recall_distance: int = 4,
    sensitive: bool = False,
) -> list[TurnPlan]:
    """Plan the ``target_length - 1`` follow-up turns (turn 0 is the seed).

    Returns one :class:`TurnPlan` per follow-up turn (index ``i`` -> turn
    ``i + 1``).
    """
    ratios = ratios or CompositionRatios()
    n = max(0, target_length - 1)
    if n == 0:
        return []

    # -- category counts ---------------------------------------------------
    # Transformations are capped well below the detector threshold; exploration
    # absorbs the remainder so it always dominates.
    cap = int(n * min(ratios.max_transform, 0.5))
    n_transform = min(round(ratios.transform * n), cap)
    n_memory = round(ratios.memory * n)
    # Keep at least one exploration turn whenever there is room.
    if n_transform + n_memory > n - 1 and n >= 2:
        n_memory = max(0, n - 1 - n_transform)
    n_explore = max(0, n - n_transform - n_memory)

    categories = (["explore"] * n_explore
                  + ["transform"] * n_transform
                  + ["memory"] * n_memory)
    # The very first follow-up is a natural topical follow-up, not a transform of
    # a possibly one-line seed answer.
    rng.shuffle(categories)
    categories = _front_load_explore(categories)

    plans = [TurnPlan(category=c) for c in categories]

    # -- memory sub-kinds --------------------------------------------------
    memory_idx = [i for i, c in enumerate(categories) if c == "memory"]
    for i in memory_idx:
        plans[i].memory_kind = "content"  # default: recall of real prior content

    if not sensitive and len(memory_idx) >= 2:
        plant_i = memory_idx[0]
        plant_turn = plant_i + 1  # turn_index of the plant
        # the recall must sit at least D turns after the plant
        recall_candidates = [
            i for i in memory_idx[1:] if (i + 1) - plant_turn >= min_recall_distance
        ]
        if recall_candidates:
            plans[plant_i].memory_kind = "plant"
            plans[recall_candidates[0]].memory_kind = "recall"
            # a second, later memory slot may exercise an update->confirm
            later = [
                i for i in recall_candidates[1:]
                if (i + 1) - plant_turn >= min_recall_distance
            ]
            if later and rng.random() < 0.5:
                plans[later[0]].memory_kind = "update"
    return plans


def _front_load_explore(categories: list[str]) -> list[str]:
    """Ensure the first follow-up turn is an exploration turn when one exists."""
    if not categories or categories[0] == "explore":
        return categories
    for j, c in enumerate(categories):
        if c == "explore":
            categories[0], categories[j] = categories[j], categories[0]
            break
    return categories
