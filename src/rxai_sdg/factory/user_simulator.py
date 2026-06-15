"""User-Simulator component (spec §5.3).

Owns the ``(intent x policy)`` sampler draw (via the injected sampler), the
temporal-policy validity gate, persona/difficulty-free **grounding** of the
follow-up query, the fact-plant/recall/update lifecycle, and structured
``constraint_spec`` emission.

Coherence contract (see the fix report):

* **One generation per query. No second-pass rewrite** - ``naturalize`` is gone.
  The grounded query is produced directly. An optional injected client may phrase
  a transformation/topic follow-up in a single pass, but the machine-checkable
  ``constraint_spec`` (and its exact parameters) is always built programmatically
  via :mod:`rxai_sdg.factory.constraints`.
* **Every follow-up operates on real content.** Transformation intents target the
  prior answer ("rewrite your previous answer as ..."); ``deepen`` /
  ``self_critique`` / ``chained_compute`` operate on the prior answer's claims;
  ``open_chat`` continues the running topic. The simulator never introduces an
  unrelated topic except as a deliberate, ledger-tracked fact plant whose exact
  string is woven into the turn.
* **The ledger is the single source of truth for facts.** A planted/updated value
  is asserted present in the emitted query, then the fact is marked *injected*; a
  delayed recall only fires for an already-injected, sufficiently-distant fact.
* **Temporal-policy validity** is enforced here: ``cumulative`` needs >=1 prior
  active constraint; ``standing`` may bootstrap on any follow-up turn;
  ``delayed_recall`` of a fact needs an injected fact planted >= D turns earlier.

The simulator is fully usable with **no** client (deterministic grounded
templates), which is what the unit/integration tests use.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from . import constraints as C
from .clients import LLMClient
from .ledger import NeedlePlanner
from .prompts import PromptPack
from .sampler import IntentPolicySampler, SamplerDraw
from .schemas import ConstraintSpec, Turn
from .taxonomy import FACT_INTENTS, TRANSFORMATION_INTENTS

# Words stripped when deriving a short "running topic" phrase from the seed query.
_TOPIC_STOPWORDS = {
    "explain", "describe", "write", "what", "why", "how", "outline", "give",
    "tell", "summarize", "summarise", "compute", "calculate", "reverse", "is",
    "are", "the", "a", "an", "of", "to", "me", "please", "could", "would", "you",
    "your", "about", "and", "or", "in", "on", "for", "with", "do", "does", "can",
}

#: Grounding categories surfaced as structured metadata (asserted in tests).
GROUNDING_KINDS = frozenset({
    "transforms_prior", "operates_on_prior", "continues_topic",
    "plants_fact", "recalls_fact", "updates_fact",
})


@dataclass
class SimulatorResult:
    nl_query: str
    constraint_spec: Optional[ConstraintSpec]
    draw: SamplerDraw
    #: how this query is grounded in real content (structured metadata)
    grounding: str = "transforms_prior"
    #: the running-topic phrase the query was grounded against
    topic: str = ""


class UserSimulator:
    def __init__(
        self,
        sampler: IntentPolicySampler,
        planner: NeedlePlanner,
        client: Optional[LLMClient] = None,
        rng: Optional[random.Random] = None,
        lang: str = "en",
        min_recall_distance: int = 4,
        max_intent_resamples: int = 16,
    ):
        self.sampler = sampler
        self.planner = planner
        self.client = client
        self.rng = rng or random.Random()
        self.lang = lang
        self.min_recall_distance = min_recall_distance
        self.max_intent_resamples = max_intent_resamples

    # ------------------------------------------------------------------- public
    def next_query(
        self,
        prior_turns: list[Turn],
        prompt_pack: PromptPack,
        turn_index: int,
        active_constraints: Optional[list[ConstraintSpec]] = None,
        avoid_intents: Optional[set[str]] = None,
        forced_draw: Optional[SamplerDraw] = None,
    ) -> SimulatorResult:
        """Sample a temporally-valid ``(intent, policy)`` and emit a grounded query.

        Resamples on a temporally-invalid policy (e.g. ``cumulative`` with no prior
        active constraint) or a non-satisfiable fact draw, then falls back to an
        immediate lexical transformation of the prior answer (always satisfiable).
        """
        active = list(active_constraints or [])
        avoid = set(avoid_intents or set())
        topic = self._topic_phrase(prior_turns)

        for _ in range(self.max_intent_resamples):
            draw = forced_draw or self.sampler.sample(self.lang, rng=self.rng)
            forced_draw = None  # honour the forced draw only on the first attempt
            if draw.intent in avoid:
                continue
            if not self._temporally_valid(draw, turn_index, active):
                continue
            ctx = C.BuildContext(
                rng=self.rng, intent=draw.intent, policy=draw.policy,
                turn=turn_index, lang=self.lang, planner=self.planner,
                min_recall_distance=self.min_recall_distance,
            )
            result = C.build(ctx)
            if result.resample:
                continue
            sim = self._finalize(result, draw, prior_turns, prompt_pack, topic)
            if sim is not None:
                return sim

        return self._fallback(turn_index, topic)

    # -------------------------------------------------------------- validity
    def _temporally_valid(
        self, draw: SamplerDraw, turn_index: int, active: list[ConstraintSpec],
    ) -> bool:
        """Reject policies that have nothing to operate on at this turn (§4.4)."""
        policy = draw.policy
        if policy == "cumulative":
            # must have a prior active constraint to accumulate onto
            return len(active) >= 1
        if policy == "standing":
            # a standing rule may bootstrap the stack, but only on a follow-up turn
            return turn_index >= 1
        if policy == "delayed_recall":
            if draw.intent in FACT_INTENTS:
                fact = self.planner.recallable_fact(
                    turn_index, self.min_recall_distance, require_injected=True)
                return fact is not None
            # a transformation labelled delayed_recall just operates on the prior
            # answer; valid as long as there is a prior turn.
            return turn_index >= 1
        return True  # immediate is always valid on a follow-up

    # -------------------------------------------------------------- finalize
    def _finalize(
        self,
        result: C.BuildResult,
        draw: SamplerDraw,
        prior_turns: list[Turn],
        prompt_pack: PromptPack,
        topic: str,
    ) -> Optional[SimulatorResult]:
        intent = draw.intent
        spec = result.constraint_spec
        nl = result.nl_query
        grounding = self._grounding_for(intent, draw.policy)

        # Ground the content-light intents in the simulator (constraints.py, which
        # only builds machine-checkable specs, is left untouched).
        if intent == "open_chat":
            nl = self._open_chat_query(topic)
        elif intent == "chained_compute":
            nl = self._chained_compute_query(topic)

        # Enforce the fact lifecycle (assert exact string, mark injected).
        if intent in FACT_INTENTS:
            if not self._handle_fact_lifecycle(intent, draw.policy, spec, nl):
                return None  # resample

        # Optional single LLM grounding pass for transformation / topic intents.
        # Fact turns keep their deterministic templates so the exact value is never
        # dropped (the value is verbatim in the template by construction).
        if self.client is not None and grounding in (
                "transforms_prior", "operates_on_prior", "continues_topic"):
            phrased = self._phrase_with_llm(nl, prior_turns, prompt_pack)
            if self._llm_phrasing_ok(phrased, intent, topic):
                nl = phrased

        return SimulatorResult(
            nl_query=nl, constraint_spec=spec, draw=draw,
            grounding=grounding, topic=topic)

    def _fallback(self, turn_index: int, topic: str) -> SimulatorResult:
        """An immediate lexical transformation of the prior answer is always valid."""
        ctx = C.BuildContext(
            rng=self.rng, intent="lexical_constraint", policy="immediate",
            turn=turn_index, lang=self.lang, planner=self.planner,
            min_recall_distance=self.min_recall_distance,
        )
        result = C.build(ctx)
        return SimulatorResult(
            nl_query=result.nl_query, constraint_spec=result.constraint_spec,
            draw=SamplerDraw("lexical_constraint", "immediate"),
            grounding="transforms_prior", topic=topic)

    # ----------------------------------------------------------- fact lifecycle
    def _handle_fact_lifecycle(
        self, intent: str, policy: str, spec: Optional[ConstraintSpec], nl: str,
    ) -> bool:
        """Assert the planted/recalled fact is grounded, then mark it injected.

        Returns ``False`` (-> resample) if the contract can't be met.
        """
        if spec is None or spec.fact_id is None:
            return False
        if not self._fact_exists(spec.fact_id):
            return False
        fact = self.planner.ledger.get(spec.fact_id)

        recall_only = intent == "fact_recall" and policy == "delayed_recall"
        if recall_only:
            # The value must NOT be supplied (the model must recall it); the fact
            # must already be injected from its plant turn, and the recall question
            # must name the fact.
            if not self.planner.ledger.is_injected(spec.fact_id):
                return False
            descriptor = fact.fact_type.replace("_", " ").split()[0].lower()
            return descriptor in nl.lower()

        # plant / immediate-recall / update: the (new) value must appear verbatim.
        value = str(spec.params.get("value", fact.value))
        if value.lower() not in nl.lower():
            return False
        self.planner.ledger.mark_injected(spec.fact_id)
        return True

    def _fact_exists(self, fact_id: str) -> bool:
        try:
            self.planner.ledger.get(fact_id)
            return True
        except KeyError:
            return False

    # --------------------------------------------------------------- grounding
    @staticmethod
    def _grounding_for(intent: str, policy: str) -> str:
        if intent == "fact_update":
            return "updates_fact"
        if intent == "fact_recall":
            return "recalls_fact" if policy == "delayed_recall" else "plants_fact"
        if intent in TRANSFORMATION_INTENTS:
            return "transforms_prior"
        if intent == "open_chat":
            return "continues_topic"
        return "operates_on_prior"  # chained_compute / self_critique / deepen

    def _topic_phrase(self, prior_turns: list[Turn]) -> str:
        seed_query = prior_turns[0].query if prior_turns else ""
        words = re.findall(r"[A-Za-z0-9']+", seed_query or "")
        kept = [w for w in words if w.lower() not in _TOPIC_STOPWORDS]
        phrase = " ".join(kept[:6]).strip()
        if phrase:
            return phrase
        return " ".join(words[:6]).strip() or "this topic"

    def _open_chat_query(self, topic: str) -> str:
        return self.rng.choice([
            f"Sticking with {topic} for a moment - why does this matter for "
            f"everyday people?",
            f"On the subject of {topic}, what's a common misconception worth "
            f"clearing up?",
            f"Could you explain {topic} to a curious ten-year-old?",
        ])

    @staticmethod
    def _chained_compute_query(topic: str) -> str:
        return (f"Building on your last answer about {topic}, what's the next "
                f"logical step or computation? Show your work.")

    # ----------------------------------------------------------- LLM phrasing
    def _phrase_with_llm(
        self, directive_query: str, prior_turns: list[Turn], prompt_pack: PromptPack,
    ) -> str:
        prior_answer = prior_turns[-1].answer if prior_turns else ""
        prompt = (
            "The assistant just said:\n\"\"\"\n" + (prior_answer or "") +
            "\n\"\"\"\n\nWrite a single natural follow-up message from the user that "
            "makes exactly this request, keeping the intent identical:\n"
            f"{directive_query}\n\nOutput only the user's message.")
        try:
            resp = self.client.generate(  # type: ignore[union-attr]
                prompt, system_prompt=prompt_pack.simulator_system,
                temperature=0.8, max_tokens=192)
            return (resp.text or "").strip()
        except Exception:
            return ""

    def _llm_phrasing_ok(self, text: str, intent: str, topic: str) -> bool:
        if not text:
            return False
        if intent == "open_chat" and topic:
            head = topic.split()[0].lower()
            return head in text.lower()
        return True
