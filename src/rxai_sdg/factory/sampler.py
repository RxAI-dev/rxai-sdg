"""The ``(intent x policy)`` sampler (spec §4.3, §5.3).

The sampler draws an ``intent`` and a ``policy`` **independently** by their
configured weights, then applies the taxonomy's **invalidity mask** and
resamples on an invalid pair. It also supports runtime **down-weighting** of
chronically low-yield ``(intent, constraint_type)`` pairs (spec §3) and a
per-language valid-constraint set (spec §9).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class SamplerDraw:
    intent: str
    policy: str


class IntentPolicySampler:
    def __init__(
        self,
        taxonomy: Taxonomy,
        intent_weights: Optional[dict[str, float]] = None,
        policy_weights: Optional[dict[str, float]] = None,
        rng: Optional[random.Random] = None,
        max_resamples: int = 64,
    ):
        self.taxonomy = taxonomy
        self.rng = rng or random.Random()
        self.max_resamples = max_resamples
        self.intent_weights = dict(
            intent_weights or {k: v.weight for k, v in taxonomy.base_intents.items()})
        self.policy_weights = dict(
            policy_weights or {k: v.weight for k, v in taxonomy.distance_policies.items()})
        #: multiplicative penalties applied to specific intents at runtime
        self._intent_penalty: dict[str, float] = {}
        #: per-(intent, constraint_type) pass-rate bookkeeping for low-yield down-weighting
        self._attempts: dict[tuple[str, str], int] = {}
        self._passes: dict[tuple[str, str], int] = {}
        #: optional per-language allow-set of intents
        self.lang_valid_intents: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ weights
    def _effective_intent_weights(
        self, lang: str, allowed_intents: Optional[set[str]] = None,
    ) -> dict[str, float]:
        lang_allowed = self.lang_valid_intents.get(lang)
        out: dict[str, float] = {}
        for intent, w in self.intent_weights.items():
            if lang_allowed is not None and intent not in lang_allowed:
                continue
            if allowed_intents is not None and intent not in allowed_intents:
                continue
            if not self.taxonomy.has_any_valid_pair(intent):
                continue
            penalty = self._intent_penalty.get(intent, 1.0)
            ew = w * penalty
            if ew > 0:
                out[intent] = ew
        return out

    @staticmethod
    def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
        keys = list(weights)
        vals = [weights[k] for k in keys]
        return rng.choices(keys, weights=vals, k=1)[0]

    # ------------------------------------------------------------------- sample
    def sample(
        self,
        lang: str = "en",
        rng: Optional[random.Random] = None,
        allowed_intents: Optional[set[str]] = None,
    ) -> SamplerDraw:
        """Draw a valid ``(intent, policy)`` pair respecting weights + mask.

        ``rng`` overrides the sampler's own RNG so each conversation can draw from
        its own ``Random(seed + index)`` stream (thread-safe, reproducible).
        ``allowed_intents``, when given, restricts the draw to that intent set
        (the per-turn composition category and/or a sensitive seed's safe subset).
        """
        rng = rng or self.rng
        intent_weights = self._effective_intent_weights(lang, allowed_intents)
        if not intent_weights:
            raise ValueError(
                f"no valid intents available for lang={lang!r} "
                f"allowed={sorted(allowed_intents) if allowed_intents else None}")
        for _ in range(self.max_resamples):
            intent = self._weighted_choice(rng, intent_weights)
            policy = self._weighted_choice(rng, self.policy_weights)
            if self.taxonomy.is_valid(intent, policy):
                return SamplerDraw(intent, policy)
        # Fallback: intent fixed, choose uniformly among its valid policies.
        intent = self._weighted_choice(rng, intent_weights)
        valid = self.taxonomy.valid_policies_for(intent)
        return SamplerDraw(intent, rng.choice(valid))

    # -------------------------------------------------------------- low-yield
    def record_outcome(self, intent: str, constraint_type: str, passed: bool) -> None:
        key = (intent, constraint_type)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        if passed:
            self._passes[key] = self._passes.get(key, 0) + 1

    def pass_rate(self, intent: str, constraint_type: str) -> Optional[float]:
        key = (intent, constraint_type)
        n = self._attempts.get(key, 0)
        if n == 0:
            return None
        return self._passes.get(key, 0) / n

    def maybe_downweight(
        self,
        intent: str,
        constraint_type: str,
        threshold: float,
        min_samples: int,
        penalty: float = 0.25,
    ) -> bool:
        """Down-weight ``intent`` if a constraint type chronically fails.

        Returns ``True`` (and applies a multiplicative penalty) when the pass
        rate over at least ``min_samples`` attempts is below ``threshold``.
        """
        key = (intent, constraint_type)
        n = self._attempts.get(key, 0)
        if n < min_samples:
            return False
        rate = self._passes.get(key, 0) / n
        if rate < threshold:
            self._intent_penalty[intent] = min(
                self._intent_penalty.get(intent, 1.0), penalty)
            return True
        return False
