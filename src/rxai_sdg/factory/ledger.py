"""Fact ledger and needle planner (spec §5.5).

The :class:`FactLedger` is the **single source of truth** for facts planted in a
conversation. It registers facts with stable ``fact_id``s, tracks value history
for overwrites, exposes recall checks, and - critically - records which facts
have actually been **injected** into the conversation text. The
:class:`NeedlePlanner` schedules ``delayed_recall`` queries at distance
``D >= min_distance`` and ``update`` sequences (plant -> update -> query).

The enforced lifecycle is::

    plant(fact_id, value, turn)          # registered, value_history started
    -> mark_injected(fact_id)            # only after the exact value appears in text
    -> recallable_fact / updatable_fact  # only ever return *injected* facts
    -> recall_question / update_phrasing  # templated from the ledger value

A recall / update is therefore never scheduled against a fact whose plant string
did not appear in a prior turn (the "Meridian" desync bug). This module is
stateless with respect to the model: it only plans and bookkeeps text-level
facts. No STM / memory machinery lives here.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .schemas import Fact
from .verifiers.universal import _value_present


# A varied pool of plantable facts spanning names, places, numbers, preferences,
# dates and project codenames. Each entry yields a (fact_type, value) builder;
# values are chosen to be distinctive and exact-matchable. A larger, varied pool
# (sampled without immediate repetition, see ``plant_fact``) keeps the planted
# memories diverse across a batch rather than reusing deadline / favourite colour.
_FACT_KINDS = [
    ("favorite_color", lambda r: r.choice(
        ["teal", "crimson", "amber", "indigo", "magenta", "olive"])),
    ("lucky_number", lambda r: str(r.randint(100, 9999))),
    ("project_codename", lambda r: r.choice(
        ["Falcon", "Meridian", "Lighthouse", "Granite", "Juniper", "Halcyon"])),
    ("hometown", lambda r: r.choice(
        ["Brindale", "Coverton", "Ashmoor", "Veldon", "Pellbrook", "Marrow"])),
    ("deadline", lambda r: r.choice(
        ["Tuesday", "the 14th", "next Friday", "March 3rd", "end of quarter"])),
    ("pet_name", lambda r: r.choice(
        ["Biscuit", "Nimbus", "Pebble", "Mango", "Sergeant", "Waffles"])),
    ("mentor_name", lambda r: r.choice(
        ["Okafor", "Priya", "Halloran", "Agnès", "Devlin", "Whitlock"])),
    ("office_city", lambda r: r.choice(
        ["Trondheim", "Cordoba", "Nagasaki", "Wellington", "Reykjavik", "Salta"])),
    ("badge_number", lambda r: str(r.randint(10000, 99999))),
    ("subscription_tier", lambda r: r.choice(
        ["Bronze", "Platinum", "Founder", "Sapphire", "Tier-3"])),
    ("anniversary_date", lambda r: r.choice(
        ["June 9th", "the 22nd", "October 1st", "next spring", "December 30th"])),
    ("preferred_language", lambda r: r.choice(
        ["Rust", "Basque", "Esperanto", "Haskell", "Swahili", "Kotlin"])),
    ("seat_number", lambda r: f"{r.choice('ABCDEF')}{r.randint(1, 42)}"),
]


class FactLedger:
    """Registry of salient facts planted during a conversation."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._counter = 0
        #: fact_ids whose exact plant/update value has appeared in conversation text
        self._injected: set[str] = set()

    def __len__(self) -> int:
        return len(self._facts)

    def plant(self, value: Any, planted_turn: int, fact_type: str = "generic") -> Fact:
        self._counter += 1
        fact_id = f"f{self._counter}"
        fact = Fact(
            fact_id=fact_id,
            value=value,
            planted_turn=planted_turn,
            fact_type=fact_type,
            value_history=[{"turn": planted_turn, "value": value}],
        )
        self._facts[fact_id] = fact
        return fact

    def update(self, fact_id: str, new_value: Any, turn: int) -> Fact:
        """Overwrite a fact's value, recording the change in ``value_history``."""
        fact = self._facts[fact_id]
        fact.value = new_value
        fact.value_history.append({"turn": turn, "value": new_value})
        return fact

    def get(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def latest(self, fact_id: str) -> Any:
        return self._facts[fact_id].value

    def stale_values(self, fact_id: str) -> list[Any]:
        fact = self._facts[fact_id]
        return [h["value"] for h in fact.value_history[:-1]]

    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    # -- injection lifecycle ---------------------------------------------------
    def mark_injected(self, fact_id: str) -> None:
        """Record that ``fact_id``'s current value has appeared in the text."""
        if fact_id in self._facts:
            self._injected.add(fact_id)

    def is_injected(self, fact_id: str) -> bool:
        return fact_id in self._injected

    def injected_facts(self) -> list[Fact]:
        return [self._facts[fid] for fid in self._injected if fid in self._facts]

    def recall_check(self, answer: str, fact_id: str, match: str = "exact") -> tuple[bool, str]:
        """Programmatic check that ``answer`` recalls the latest value."""
        value = self.latest(fact_id)
        if _value_present(answer, value, match):
            return True, f"{fact_id} latest value present ({match})"
        return False, f"{fact_id} expected {value!r} absent ({match})"


class NeedlePlanner:
    """Plans plant/recall/update events over a conversation (spec §5.5)."""

    def __init__(
        self,
        ledger: FactLedger,
        rng: Optional[random.Random] = None,
        min_distance: int = 4,
    ):
        self.ledger = ledger
        self.rng = rng or random.Random()
        self.min_distance = min_distance
        #: last randomly-chosen fact type, to avoid immediate repetition in a batch
        self._last_kind: Optional[str] = None

    # ---------------------------------------------------------------- planting
    def plant_fact(self, turn: int, fact_type: Optional[str] = None) -> Fact:
        """Create and register a new fact, returning it.

        The caller must surface the fact's plant phrasing in the conversation text
        and then call :meth:`FactLedger.mark_injected` (see
        :meth:`UserSimulator` for the enforced confirm-then-mark step).
        """
        if fact_type is None:
            # sample without immediate repetition so a batch sees a varied pool
            choices = [(ft, fn) for ft, fn in _FACT_KINDS if ft != self._last_kind] \
                or _FACT_KINDS
            fact_type, value_fn = self.rng.choice(choices)
            self._last_kind = fact_type
        else:
            value_fn = next(
                (fn for ft, fn in _FACT_KINDS if ft == fact_type),
                lambda r: r.randint(1, 999),
            )
        value = value_fn(self.rng)
        return self.ledger.plant(value, planted_turn=turn, fact_type=fact_type)

    @staticmethod
    def plant_phrasing(fact: Fact) -> str:
        readable = fact.fact_type.replace("_", " ")
        return f"By the way, my {readable} is {fact.value}."

    @staticmethod
    def update_phrasing(fact: Fact) -> str:
        readable = fact.fact_type.replace("_", " ")
        return f"Actually, update my {readable} to {fact.value}."

    @staticmethod
    def recall_question(fact: Fact) -> str:
        readable = fact.fact_type.replace("_", " ")
        return f"What is my {readable}?"

    # --------------------------------------------------------------- selecting
    def recallable_fact(
        self,
        turn: int,
        min_distance: Optional[int] = None,
        require_injected: bool = False,
    ) -> Optional[Fact]:
        """Return a fact planted far enough in the past to be a long-range recall.

        Picks the oldest eligible fact (planted at ``turn - min_distance`` or
        earlier). When ``require_injected`` is set, only facts whose plant string
        actually appeared in the text are eligible. Returns ``None`` when no fact
        qualifies yet.
        """
        dist = self.min_distance if min_distance is None else min_distance
        eligible = [
            f for f in self.ledger.facts()
            if turn - f.planted_turn >= dist
            and (not require_injected or self.ledger.is_injected(f.fact_id))
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda f: f.planted_turn)
        return eligible[0]

    def updatable_fact(self, turn: int, require_injected: bool = True) -> Optional[Fact]:
        """Return the most recent fact eligible to be updated/overwritten.

        Defaults to requiring the fact to have been injected, so an update never
        references a value that never appeared in the conversation.
        """
        candidates = [
            f for f in self.ledger.facts()
            if f.planted_turn <= turn
            and (not require_injected or self.ledger.is_injected(f.fact_id))
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda f: f.planted_turn)
        return candidates[-1]  # most recently planted injected fact

    def any_fact(self) -> Optional[Fact]:
        facts = self.ledger.facts()
        return facts[-1] if facts else None

    def update_value_for(self, fact: Fact, turn: int) -> Fact:
        """Pick a fresh value distinct from **every** prior value and overwrite.

        Distinct-from-history (not merely from the current value) matters because
        the ``fact_update`` checker requires all stale values to be absent: a "new"
        value that coincides with an earlier one would read as a stale leak.
        """
        value_fn = next(
            (fn for ft, fn in _FACT_KINDS if ft == fact.fact_type),
            lambda r: r.randint(1, 999),
        )
        seen = {str(h["value"]) for h in fact.value_history}
        new_value = value_fn(self.rng)
        for _ in range(20):
            if str(new_value) not in seen:
                break
            new_value = value_fn(self.rng)
        return self.ledger.update(fact.fact_id, new_value, turn)
