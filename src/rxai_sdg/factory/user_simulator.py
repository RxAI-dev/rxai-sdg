"""User-Simulator component (spec §5.3).

Owns the ``(intent x policy)`` sampler, the invalidity mask (via the taxonomy),
persona/difficulty conditioning for diversity, opportunistic fact planting (to
feed the needle planner), and structured ``constraint_spec`` emission.

``next_query`` returns ``(nl_query, constraint_spec | None)``. The NL query is
produced from deterministic templates (in :mod:`rxai_sdg.factory.constraints`)
and can optionally be *naturalised* by an injected LLM client - keeping a client
optional is what makes the simulator unit-testable without any network.

The simulator should use a **different** model/client from the Responder where
possible, to avoid self-collusion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from . import constraints as C
from .clients import LLMClient
from .ledger import NeedlePlanner
from .prompts import PromptPack
from .sampler import IntentPolicySampler, SamplerDraw
from .schemas import ConstraintSpec, Turn
from .taxonomy import FACT_INTENTS

_PERSONAS = [
    "a busy professional", "a curious student", "a skeptical reviewer",
    "an enthusiastic hobbyist", "a non-native English speaker", "a terse manager",
]
_DIFFICULTIES = ["easy", "medium", "hard"]


@dataclass
class SimulatorResult:
    nl_query: str
    constraint_spec: Optional[ConstraintSpec]
    draw: SamplerDraw


class UserSimulator:
    def __init__(
        self,
        sampler: IntentPolicySampler,
        planner: NeedlePlanner,
        client: Optional[LLMClient] = None,
        rng: Optional[random.Random] = None,
        lang: str = "en",
        min_recall_distance: int = 4,
        plant_probability: float = 0.35,
        naturalize: bool = False,
    ):
        self.sampler = sampler
        self.planner = planner
        self.client = client
        self.rng = rng or random.Random()
        self.lang = lang
        self.min_recall_distance = min_recall_distance
        self.plant_probability = plant_probability
        self.naturalize = naturalize

    def next_query(
        self,
        prior_turns: list[Turn],
        prompt_pack: PromptPack,
        turn_index: int,
        forced_draw: Optional[SamplerDraw] = None,
        avoid_intents: Optional[set[str]] = None,
        max_intent_resamples: int = 16,
    ) -> SimulatorResult:
        """Sample an intent/policy and emit a follow-up query + constraint_spec.

        On a non-satisfiable draw (e.g. a delayed recall with no fact far enough
        in the past) the simulator transparently resamples the intent, up to
        ``max_intent_resamples`` times, before falling back to an immediate
        transformation it can always satisfy.
        """
        avoid = set(avoid_intents or set())
        for _ in range(max_intent_resamples):
            draw = forced_draw or self.sampler.sample(self.lang)
            forced_draw = None  # only honour the forced draw on the first attempt
            if draw.intent in avoid:
                continue
            ctx = C.BuildContext(
                rng=self.rng, intent=draw.intent, policy=draw.policy,
                turn=turn_index, lang=self.lang, planner=self.planner,
                min_recall_distance=self.min_recall_distance,
            )
            result = C.build(ctx)
            if result.resample:
                continue
            nl = result.nl_query
            # Opportunistically plant a needle on non-fact turns so future
            # delayed-recall draws have material to recall.
            if draw.intent not in FACT_INTENTS and self.rng.random() < self.plant_probability:
                fact = self.planner.plant_fact(turn_index)
                nl = f"{self.planner.plant_phrasing(fact)} {nl}"
            if self.naturalize and self.client is not None:
                nl = self._naturalize(nl, prompt_pack)
            return SimulatorResult(nl_query=nl, constraint_spec=result.constraint_spec, draw=draw)

        # Fallback: an immediate lexical constraint is always expressible.
        ctx = C.BuildContext(
            rng=self.rng, intent="lexical_constraint", policy="immediate",
            turn=turn_index, lang=self.lang, planner=self.planner,
            min_recall_distance=self.min_recall_distance,
        )
        result = C.build(ctx)
        return SimulatorResult(
            nl_query=result.nl_query, constraint_spec=result.constraint_spec,
            draw=SamplerDraw("lexical_constraint", "immediate"))

    def _naturalize(self, templated: str, prompt_pack: PromptPack) -> str:
        persona = self.rng.choice(_PERSONAS)
        difficulty = self.rng.choice(_DIFFICULTIES)
        prompt = (
            f"Rephrase the following follow-up request as {persona} would say it "
            f"({difficulty} difficulty), preserving its exact requirement. Reply "
            f"with only the rephrased message.\n\nRequest: {templated}")
        try:
            resp = self.client.generate(  # type: ignore[union-attr]
                prompt, system_prompt=prompt_pack.simulator_system,
                temperature=0.9, max_tokens=256)
            text = (resp.text or "").strip()
            return text or templated
        except Exception:
            return templated
