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


# Personal-fact generation.
#
# The PRODUCTION source of planted facts is the LLM curator: it samples a few
# topic-grounded, distinctive personal details per conversation (see
# ``SeedDirective.facts``), so a large dataset never recycles a fixed handful of
# values. The combinatorial generator below is only an OFFLINE FALLBACK (no curator
# client) and is built from composable pieces so even it yields hundreds-to-
# thousands of distinct values rather than a fixed list.
#
# IMPORTANT: only **personal details a user legitimately shares in passing** -
# names, places, preferences, numbers, pet names. NEVER account / subscription /
# billing / system data (those correctly trigger "I can't access your account"
# refusals from a well-aligned assistant).

_NAME_A = ["Tov", "Mar", "Pri", "Okaf", "Hall", "Dev", "Nim", "Wend", "Cael", "Dris",
           "Fenn", "Garr", "Hux", "Iver", "Joss", "Kael", "Lor", "Mira", "Sor", "Yar",
           "Bex", "Call", "Dov", "Esm", "Frey", "Git", "Holl", "Ines", "Jag", "Kit",
           "Lub", "Nad", "Orin", "Petr", "Quill", "Ros", "Sael", "Tamsin", "Ud", "Vesh"]
_NAME_B = ["or", "isse", "ya", "an", "ic", "lin", "ette", "is", "en", "ara",
           "ix", "wen", "old", "ric", "a", "iel", "us", "ow", "ka", "ima"]
_PLACE_A = ["Brin", "Cover", "Ash", "Vel", "Pell", "Mar", "Trond", "Cord", "Wex",
            "Tin", "Harrow", "Calder", "Wend", "Brack", "Gild", "Fenn", "Holm", "Ravens"]
_PLACE_B = ["dale", "ton", "moor", "don", "brook", "row", "heim", "gate", "mere",
            "stead", "thorpe", "vale", "wick", "ford", "haven", "field", "by", "combe"]
_STREET_A = ["Maple", "Birch", "Halcyon", "Juniper", "Granite", "Willow", "Cedar",
             "Sparrow", "Linden", "Ember", "Foxglove", "Marigold", "Thistle", "Quartz"]
_STREET_B = ["Lane", "Row", "Drive", "Court", "Way", "Close", "Terrace", "Walk", "Rise"]
_COLOR = ["teal", "crimson", "amber", "indigo", "magenta", "olive", "saffron",
          "cerulean", "vermilion", "chartreuse", "ochre", "periwinkle", "sienna"]
_FOOD = ["ramen", "paella", "dumplings", "tiramisu", "shakshuka", "pho", "laksa",
         "gnocchi", "bibimbap", "khachapuri", "tagine", "okonomiyaki", "borscht"]
_LANG = ["Rust", "Basque", "Esperanto", "Haskell", "Swahili", "Kotlin", "Welsh",
         "Tagalog", "Elixir", "Quechua", "Faroese", "Zig", "Twi", "Romansh"]
_PROJ = ["Falcon", "Meridian", "Lighthouse", "Granite", "Juniper", "Halcyon",
         "Driftwood", "Saffron", "Cobalt", "Tundra", "Lantern", "Mosaic", "Verdant"]


def _name(r: random.Random) -> str:
    return r.choice(_NAME_A) + r.choice(_NAME_B)


#: label -> value generator (fallback only). Labels read naturally after "my ___".
_FALLBACK_KINDS: dict[str, Any] = {
    "favorite color": lambda r: r.choice(_COLOR),
    "lucky number": lambda r: str(r.randint(2, 999)),
    "home town": lambda r: r.choice(_PLACE_A) + r.choice(_PLACE_B),
    "pet's name": lambda r: _name(r),
    "mentor's name": lambda r: _name(r),
    "favorite dish": lambda r: r.choice(_FOOD),
    "favorite author": lambda r: _name(r) + " " + r.choice(_NAME_A) + r.choice(_NAME_B),
    "preferred language": lambda r: r.choice(_LANG),
    "childhood street": lambda r: r.choice(_STREET_A) + " " + r.choice(_STREET_B),
    "current project's codename": lambda r: r.choice(_PROJ),
}
_FALLBACK_LABELS = list(_FALLBACK_KINDS)


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
        fact_pool: Optional[list[dict[str, str]]] = None,
    ):
        self.ledger = ledger
        self.rng = rng or random.Random()
        self.min_distance = min_distance
        #: per-conversation, topic-grounded facts from the curator (preferred source).
        self._fact_pool = list(fact_pool or [])
        self._pool_idx = 0
        #: fact_id -> a distinct alternate value, used if the fact is later updated.
        self._alt_value: dict[str, str] = {}
        #: last fallback label, to avoid immediate repetition when generating.
        self._last_label: Optional[str] = None

    # ---------------------------------------------------------------- planting
    def plant_fact(self, turn: int, fact_type: Optional[str] = None) -> Fact:
        """Create and register a new fact, returning it.

        Draws from the curator's per-conversation pool first (topic-grounded,
        diverse), then from the combinatorial fallback. The caller must surface the
        plant phrasing in the text and call :meth:`FactLedger.mark_injected` (the
        loop does this only once the turn is accepted).
        """
        label, value, alt = self._next_fact(fact_type)
        fact = self.ledger.plant(value, planted_turn=turn, fact_type=label)
        if alt:
            self._alt_value[fact.fact_id] = alt
        return fact

    def _next_fact(self, fact_type: Optional[str] = None) -> tuple[str, str, str]:
        """Return ``(label, value, alt_value)`` for the next plant."""
        # An explicit fact_type (used by unit tests) bypasses the curator pool.
        if fact_type is None:
            while self._pool_idx < len(self._fact_pool):
                item = self._fact_pool[self._pool_idx]
                self._pool_idx += 1
                label = str(item.get("label") or "").strip()
                value = str(item.get("value") or "").strip()
                if not label or len(value) < 2:
                    continue
                alt = str(item.get("new_value") or "").strip()
                return label, value, alt
        return self._generate_fact(fact_type)

    def _generate_fact(self, fact_type: Optional[str] = None) -> tuple[str, str, str]:
        """Combinatorial fallback: ``(label, value, alt_value)`` (no curator)."""
        if fact_type and fact_type in _FALLBACK_KINDS:
            label = fact_type
        elif fact_type:
            # an unknown explicit type: keep it, generate a name-like value.
            label = fact_type.replace("_", " ")
            return label, _name(self.rng), _name(self.rng)
        else:
            choices = [l for l in _FALLBACK_LABELS if l != self._last_label] or _FALLBACK_LABELS
            label = self.rng.choice(choices)
            self._last_label = label
        gen = _FALLBACK_KINDS[label]
        value = gen(self.rng)
        alt = gen(self.rng)
        for _ in range(8):
            if str(alt) != str(value):
                break
            alt = gen(self.rng)
        return label, value, alt

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

    def next_update_value(self, fact: Fact) -> Any:
        """Pick a fresh value distinct from **every** prior value, WITHOUT committing.

        The caller commits the overwrite (``FactLedger.update``) only once the
        update turn actually passes - otherwise a failed/resampled update would
        leave a phantom value in the ledger (which a later update would then list as
        a stale value that never appeared in the text). Distinct-from-history (not
        just from the current value) keeps the ``fact_update`` checker happy.
        """
        seen = {str(h["value"]) for h in fact.value_history}
        # Prefer the curator's alternate value for this exact fact (topic-grounded).
        alt = self._alt_value.get(fact.fact_id)
        if alt and str(alt) not in seen and len(str(alt)) >= 2:
            return alt
        # Otherwise generate a fresh value of the same kind, distinct from history.
        for _ in range(20):
            _, new_value, _ = self._generate_fact(fact.fact_type)
            if str(new_value) not in seen:
                return new_value
        return alt or _name(self.rng)

    def update_value_for(self, fact: Fact, turn: int) -> Fact:
        """Pick a fresh value and commit it immediately (back-compat helper)."""
        return self.ledger.update(fact.fact_id, self.next_update_value(fact), turn)
