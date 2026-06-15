"""Fact ledger and needle planner (spec §5.5).

The :class:`FactLedger` registers salient facts with stable ``fact_id``s, tracks
value history for overwrites, and exposes recall checks. The
:class:`NeedlePlanner` schedules ``delayed_recall`` queries at distance
``D >= min_distance`` and ``update_overwrite`` sequences (plant -> update ->
query), so long conversations actually train slow-timescale memory rather than a
sequence of independent depth-1 transitions.

This module is stateless with respect to the model: it only plans and bookkeeps
text-level facts. No STM / memory machinery lives here.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .schemas import Fact
from .verifiers.universal import _value_present


# A small pool of plantable facts. Each entry yields a (fact_type, value,
# plant phrasing, recall question) tuple builder. Values are chosen to be
# distinctive and exact-matchable.
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
]


class FactLedger:
    """Registry of salient facts planted during a conversation."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._counter = 0

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

    # ---------------------------------------------------------------- planting
    def plant_fact(self, turn: int, fact_type: Optional[str] = None) -> Fact:
        """Create and register a new fact, returning it.

        The caller is responsible for surfacing the fact's plant phrasing in the
        conversation text (see :meth:`plant_phrasing`).
        """
        if fact_type is None:
            fact_type, value_fn = self.rng.choice(_FACT_KINDS)
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
    def recallable_fact(self, turn: int, min_distance: Optional[int] = None) -> Optional[Fact]:
        """Return a fact planted far enough in the past to be a long-range recall.

        Picks the oldest eligible fact (planted at ``turn - min_distance`` or
        earlier). Returns ``None`` when no fact qualifies yet.
        """
        dist = self.min_distance if min_distance is None else min_distance
        eligible = [
            f for f in self.ledger.facts()
            if turn - f.planted_turn >= dist
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda f: f.planted_turn)
        return eligible[0]

    def any_fact(self) -> Optional[Fact]:
        facts = self.ledger.facts()
        return facts[-1] if facts else None

    def update_value_for(self, fact: Fact, turn: int) -> Fact:
        """Pick a fresh distinct value and apply it as an overwrite."""
        value_fn = next(
            (fn for ft, fn in _FACT_KINDS if ft == fact.fact_type),
            lambda r: r.randint(1, 999),
        )
        new_value = value_fn(self.rng)
        # ensure it differs from the current value
        for _ in range(5):
            if str(new_value) != str(fact.value):
                break
            new_value = value_fn(self.rng)
        return self.ledger.update(fact.fact_id, new_value, turn)
