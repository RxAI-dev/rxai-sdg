"""User-Simulator component (spec §5.3).

A **genuine, LLM-driven user** that sees the full conversation history and drives
the conversation. The sampled ``(intent, policy)`` and the constraint params are a
**steer**, never a template: the simulator picks a temporally-valid draw, builds
the machine-checkable ``constraint_spec`` programmatically (via
:mod:`rxai_sdg.factory.constraints`), then asks the instruct LLM to write **one**
natural user message that realises that steer, grounded in the real transcript.

Contracts (see the fix report):

* **Fully LLM-driven, full transcript.** Every query is produced by the simulator
  LLM from the *entire* conversation so far - not the last answer alone, and never
  a hardcoded template. There is no second pass and no fallback to canned
  production strings. (Unit tests script a Mock client.)
* **Spec-first for verifiable constraints.** For intents whose
  ``constraint_spec.verifier`` is ``programmatic``/``hybrid`` the exact params are
  chosen programmatically (letter ``A``, ``format=json``, ...); the LLM is then
  instructed to write a turn that *explicitly and naturally requests that exact
  constraint*, grounded in the prior answer. The spec stays machine-checkable.
* **Diversity.** A persona (curious / skeptical / frustrated / enthusiastic /
  terse-expert / casual) and a verbosity target (short ↔ long) are sampled per
  call so user turns range from terse to long and rambling. No fixed phrasing.
* **Fact lifecycle across turns (never same-turn).** A *plant* weaves the exact
  value into a natural, topical turn (value asserted present, fact marked
  injected); a *recall* fires at a later turn ``>= min_recall_distance`` away and
  must **not** restate the value; an *update* states a new value. The
  plant-and-recall-in-one-turn path is gone - it tests no memory.
* **Post-generation coherence check.** For verifiable constraints the emitted
  query must mention the constraint (format / letter / forbidden token / length);
  for facts the value must be present (plant/update) or absent (recall). On
  failure the simulator regenerates once, then resamples the intent (bounded by
  the existing resample budget).
* **Temporal-policy validity** (``_temporally_valid``) is unchanged: ``cumulative``
  needs >=1 prior active constraint; ``standing`` bootstraps on any follow-up;
  ``delayed_recall`` of a fact needs an injected, sufficiently-distant fact.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
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

#: Personas sampled to diversify the user's voice.
PERSONAS = (
    "curious", "skeptical", "frustrated", "enthusiastic", "terse-expert", "casual",
)
#: Verbosity targets - from a one-line ask to a long, multi-sentence message.
VERBOSITIES = ("short", "medium", "long")

_REASONING_EMPTY = {"", ".", "..", "...", "…"}


@dataclass
class SimulatorResult:
    nl_query: str
    constraint_spec: Optional[ConstraintSpec]
    draw: SamplerDraw
    #: how this query is grounded in real content (structured metadata)
    grounding: str = "transforms_prior"
    #: the running-topic phrase the query was grounded against
    topic: str = ""
    #: the sampled persona / verbosity target (diversity metadata)
    persona: str = "curious"
    verbosity: str = "medium"


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
        self._last_persona: Optional[str] = None

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
        active constraint), a non-satisfiable fact draw, or a query that fails the
        coherence check twice, then falls back to an immediate lexical
        transformation of the prior answer (always satisfiable).
        """
        if self.client is None:
            raise RuntimeError(
                "UserSimulator requires an LLM client - it is fully LLM-driven. "
                "Pass a (mock) client; there is no hardcoded template fallback.")
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
            result = C.build(self._ctx(draw, turn_index))
            if result.resample or result.constraint_spec is None:
                continue
            sim = self._generate(result.constraint_spec, draw, prior_turns,
                                 prompt_pack, topic)
            if sim is not None:
                return sim

        return self._fallback(prior_turns, prompt_pack, turn_index, topic)

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

    def _ctx(self, draw: SamplerDraw, turn_index: int) -> C.BuildContext:
        return C.BuildContext(
            rng=self.rng, intent=draw.intent, policy=draw.policy,
            turn=turn_index, lang=self.lang, planner=self.planner,
            min_recall_distance=self.min_recall_distance,
        )

    # -------------------------------------------------------------- generate
    def _generate(
        self,
        spec: ConstraintSpec,
        draw: SamplerDraw,
        prior_turns: list[Turn],
        prompt_pack: PromptPack,
        topic: str,
    ) -> Optional[SimulatorResult]:
        """Drive the simulator LLM to write one grounded, coherent user message.

        Returns ``None`` (-> resample the intent) when the generated query fails the
        coherence check twice in a row, or when a delayed recall would target a
        not-yet-injected fact.
        """
        intent, policy = draw.intent, draw.policy
        grounding = self._grounding_for(intent, policy)

        # A delayed recall must target an already-injected fact (no desync).
        if grounding == "recalls_fact" and not self.planner.ledger.is_injected(
                spec.fact_id or ""):
            return None

        persona = self._sample_persona()
        verbosity = self.rng.choice(VERBOSITIES)
        steer = self._steer(intent, policy, spec, topic)
        prompt = self._build_prompt(prior_turns, persona, verbosity, steer)

        for _ in range(2):  # initial generation + one regeneration
            text = self._call_llm(prompt, prompt_pack)
            if self._coherence_ok(text, intent, policy, spec, topic):
                if grounding in ("plants_fact", "updates_fact") and spec.fact_id:
                    self.planner.ledger.mark_injected(spec.fact_id)
                return SimulatorResult(
                    nl_query=text, constraint_spec=spec, draw=draw,
                    grounding=grounding, topic=topic,
                    persona=persona, verbosity=verbosity)
        return None

    def _fallback(
        self, prior_turns: list[Turn], prompt_pack: PromptPack,
        turn_index: int, topic: str,
    ) -> SimulatorResult:
        """An immediate lexical transformation of the prior answer is always valid.

        Used only when every sampled intent failed the coherence check within the
        resample budget; the LLM still writes the message (no canned template).
        """
        draw = SamplerDraw("lexical_constraint", "immediate")
        result = C.build(self._ctx(draw, turn_index))
        spec = result.constraint_spec
        persona = self._sample_persona()
        verbosity = self.rng.choice(VERBOSITIES)
        steer = self._steer(draw.intent, draw.policy, spec, topic)
        prompt = self._build_prompt(prior_turns, persona, verbosity, steer)
        text = self._call_llm(prompt, prompt_pack)
        return SimulatorResult(
            nl_query=text, constraint_spec=spec, draw=draw,
            grounding="transforms_prior", topic=topic,
            persona=persona, verbosity=verbosity)

    def _sample_persona(self) -> str:
        """Sample a persona, avoiding immediate repetition (diversity)."""
        choices = [p for p in PERSONAS if p != self._last_persona] or list(PERSONAS)
        persona = self.rng.choice(choices)
        self._last_persona = persona
        return persona

    # --------------------------------------------------------------- steer
    def _steer(
        self, intent: str, policy: str, spec: ConstraintSpec, topic: str,
    ) -> dict[str, str]:
        """Build the machine-readable steer the LLM (and the test Mock) realise.

        The ``op`` field tells the simulator LLM what kind of turn to write; the
        ``say`` directive describes the exact constraint to request (for verifiable
        turns); ``fact_*`` carry the ledger fact for plant/recall/update.
        """
        grounding = self._grounding_for(intent, policy)
        if grounding == "plants_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            return {"op": "plant_fact", "fact_label": _readable(fact.fact_type),
                    "fact_value": str(spec.params.get("value", fact.value))}
        if grounding == "recalls_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            # NB: no fact_value - the user must NOT restate the value when recalling.
            return {"op": "recall_fact", "fact_label": _readable(fact.fact_type)}
        if grounding == "updates_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            return {"op": "update_fact", "fact_label": _readable(fact.fact_type),
                    "fact_value": str(spec.params.get("value", fact.value))}
        if intent == "open_chat":
            return {"op": "continue_topic", "topic": topic,
                    "say": f"keep the conversation going about {topic} with an open, "
                           f"curious follow-up"}
        # transformation / operates-on-prior: describe the exact constraint to ask for.
        return {"op": "request_constraint", "say": _describe_constraint(intent, spec)}

    def _build_prompt(
        self, prior_turns: list[Turn], persona: str, verbosity: str,
        steer: dict[str, str],
    ) -> str:
        """Assemble the simulator prompt: full transcript + persona/length + steer."""
        transcript = self._format_transcript(prior_turns)
        length_hint = {
            "short": "a single short sentence",
            "medium": "two or three sentences",
            "long": "a longer, multi-sentence message that rambles a little",
        }[verbosity]
        lines = [
            "Full conversation so far (you are the user):",
            transcript or "(no messages yet)",
            "",
            "Write the user's NEXT message. Make it a coherent continuation that "
            "engages the assistant's real prior content above.",
            f"- Persona: {persona}.",
            f"- Length: {length_hint}.",
        ]
        op = steer.get("op")
        if op == "plant_fact":
            lines.append(
                f"- This turn: while doing something topical, mention in passing "
                f"that your {steer['fact_label']} is {steer['fact_value']}. State "
                f"that exact value.")
        elif op == "recall_fact":
            lines.append(
                f"- This turn: ask the assistant to tell you your "
                f"{steer['fact_label']} again. Do NOT restate the value yourself - "
                f"you are testing whether it remembers.")
        elif op == "update_fact":
            lines.append(
                f"- This turn: tell the assistant your {steer['fact_label']} is now "
                f"{steer['fact_value']} and ask it to confirm the current value. "
                f"State that exact new value.")
        else:  # request_constraint / continue_topic
            lines.append(f"- Your request to the assistant this turn: {steer.get('say', '')}.")
        lines.append(
            "\nOutput only the user's message - no labels, no quotes, do not answer "
            "your own question.")
        # The machine-readable steer lets the deterministic test client realise the
        # same turn without an LLM; a real LLM simply follows the prose above.
        lines.append("\n=== STEER ===")
        for key in ("op", "say", "fact_label", "fact_value", "topic"):
            if key in steer:
                lines.append(f"{key}: {steer[key]}")
        lines.append(f"persona: {persona}")
        lines.append(f"length: {verbosity}")
        lines.append("=== END STEER ===")
        return "\n".join(lines)

    @staticmethod
    def _format_transcript(prior_turns: list[Turn]) -> str:
        out: list[str] = []
        for t in prior_turns:
            if t.query:
                out.append(f"User: {t.query}")
            if t.answer:
                out.append(f"Assistant: {t.answer}")
        return "\n".join(out)

    def _call_llm(self, prompt: str, prompt_pack: PromptPack) -> str:
        # No max_tokens cap: the verbosity target (not a token budget) controls
        # length, so long user turns are possible.
        try:
            resp = self.client.generate(  # type: ignore[union-attr]
                prompt, system_prompt=prompt_pack.simulator_system, temperature=0.9)
            return (resp.text or "").strip()
        except Exception:
            return ""

    # ----------------------------------------------------------- coherence
    def _coherence_ok(
        self, text: str, intent: str, policy: str, spec: ConstraintSpec, topic: str,
    ) -> bool:
        if not text or not text.strip():
            return False
        low = text.lower()
        grounding = self._grounding_for(intent, policy)

        if grounding in ("plants_fact", "updates_fact", "recalls_fact"):
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            value = str(spec.params.get("value", fact.value)).lower()
            label_head = _readable(fact.fact_type).split()[0].lower()
            if grounding == "recalls_fact":
                # value must be ABSENT and the fact named
                return value not in low and label_head in low
            return value in low  # plant / update: value woven in

        if spec.verifier in ("programmatic", "hybrid"):
            markers = _constraint_markers(spec)
            return any(m and m.lower() in low for m in markers)

        if intent == "open_chat" and topic:
            return topic.split()[0].lower() in low

        # restyle / expand / deepen / self_critique / chained_compute: grounded by
        # construction (they operate on the prior answer); accept any non-empty turn.
        return True

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


# ---------------------------------------------------------------------------
# spec -> directive / markers (steer the LLM; not the final user message)
# ---------------------------------------------------------------------------

def _readable(fact_type: str) -> str:
    return fact_type.replace("_", " ")


def _describe_constraint(intent: str, spec: ConstraintSpec) -> str:
    """The exact request the user should make this turn, phrased to the assistant.

    This is a *steer* (the content of the user's request), not a ready-to-send
    message - the LLM rewrites it into a natural, persona-flavoured user turn. It
    is written in the second person ("your previous answer") because the user is
    addressing the assistant directly.
    """
    t, p = spec.type, spec.params
    if intent == "chained_compute":
        return ("build on your previous answer with the next logical step or "
                "computation, and show your work")
    if intent == "self_critique":
        return "critique your previous answer and fix any weaknesses you find"
    if intent == "deepen":
        return ("go deeper on one key point from your previous answer with a "
                "concrete example")
    if intent == "expand":
        return "expand your previous answer with more detail and a concrete example"
    if t == "style":
        return f"restyle your previous answer in {p.get('style', 'a different tone')}"
    if t in ("genre", "limerick_structure"):
        return f"rewrite your previous answer as a {p.get('genre', 'limerick')}"
    if t == "json_valid":
        return "reformat your previous answer as a single valid JSON object"
    if t == "yaml_valid":
        return "reformat your previous answer as valid YAML"
    if t == "markdown_table":
        return "reformat your previous answer as a markdown table"
    if t == "markdown_format":
        return "reformat your previous answer using markdown formatting (headings and bullets)"
    if t == "first_letter":
        return f"rewrite your previous answer so that every sentence starts with the letter '{p.get('letter', 'A')}'"
    if t == "alphabetical_sentence_starts":
        return "rewrite your previous answer so consecutive sentences start in alphabetical order"
    if t == "max_words_per_sentence":
        return f"rewrite your previous answer keeping every sentence to at most {p.get('max_words')} words"
    if t == "forbidden_token":
        return f"rewrite your previous answer without ever using the word '{p.get('token')}'"
    if t == "no_gendered_pronouns":
        return "rewrite your previous answer using no gendered pronouns"
    if t == "length_tokens":
        return f"compress your previous answer to at most {p.get('max_words')} words"
    if t == "n_bullets":
        return f"compress your previous answer into exactly {p.get('n')} bullet points"
    return "revise your previous answer"


def _constraint_markers(spec: ConstraintSpec) -> list[str]:
    """Tokens at least one of which must appear in a verifiable-constraint query."""
    t, p = spec.type, spec.params
    table: dict[str, list[str]] = {
        "json_valid": ["json"],
        "yaml_valid": ["yaml"],
        "markdown_table": ["table"],
        "markdown_format": ["markdown"],
        "first_letter": [f"'{p.get('letter', '')}'", f"letter {p.get('letter', '')}"],
        "alphabetical_sentence_starts": ["alphabetical"],
        "max_words_per_sentence": [str(p.get("max_words", ""))],
        "forbidden_token": [f"'{p.get('token', '')}'", str(p.get("token", ""))],
        "no_gendered_pronouns": ["pronoun"],
        "length_tokens": [str(p.get("max_words", ""))],
        "n_bullets": [str(p.get("n", ""))],
        "limerick_structure": ["limerick"],
        "genre": [str(p.get("genre", ""))],
        "style": [str(p.get("style", "")).split()[0] if p.get("style") else ""],
    }
    return table.get(t, [])
