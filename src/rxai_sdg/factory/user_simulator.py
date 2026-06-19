"""User-Simulator component (spec §5.3, fixes B/C/D/E).

A **genuine, LLM-driven user** that sees the full conversation history and drives
the conversation. Generation is steered by the per-conversation **planner**: each
follow-up turn carries a composition *category* (``explore`` / ``transform`` /
``memory``) and, for memory turns, a sub-kind (``content`` / ``plant`` /
``recall`` / ``update``). The simulator turns that steer into one natural user
message grounded in the real transcript.

Contracts:

* **Topical thread, not format gymnastics (B).** The category is chosen by the
  planner (exploration dominant); the simulator samples an intent *within* the
  category and respects a sensitive seed's safe ``allowed_intents`` subset.
* **Memory realism (C).** The default memory test is **recall of real prior
  content** ("earlier you said X about <topic> - can you expand?"), needing no
  artificial injection. Explicit personal-fact plants are topically woven and
  occasional; a plant and its recall are always **different turns**; a recall never
  restates the value; stale values are real, injected values only.
* **Builders steer, never templated queries (D).** The constraint builder returns
  ``(directive, constraint_spec)``; the directive is one input to the LLM. No
  hardcoded query ever reaches the output.
* **Strict user role (E).** Enforced by the simulator system prompt.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from . import constraints as C
from .clients import LLMClient
from .ledger import NeedlePlanner
from .planner import TurnPlan
from .prompts import PromptPack
from .responder import sanitize_generated_text
from .sampler import IntentPolicySampler, SamplerDraw
from .schemas import ConstraintSpec, Turn
from .taxonomy import (
    COMPOSITION_CATEGORIES, FACT_INTENTS, TRANSFORMATION_INTENTS,
)

# Words stripped when deriving a short "running topic" phrase from the seed query
# (fallback only - the LLM curator's ``topic`` is preferred, fix A).
_TOPIC_STOPWORDS = {
    "explain", "describe", "write", "what", "why", "how", "outline", "give",
    "tell", "summarize", "summarise", "compute", "calculate", "reverse", "is",
    "are", "the", "a", "an", "of", "to", "me", "please", "could", "would", "you",
    "your", "about", "and", "or", "in", "on", "for", "with", "do", "does", "can",
}

#: Grounding categories surfaced as structured metadata (asserted in tests).
GROUNDING_KINDS = frozenset({
    "transforms_prior", "operates_on_prior", "continues_topic",
    "plants_fact", "recalls_fact", "updates_fact", "recalls_content",
})

#: Personas sampled to diversify the user's voice.
PERSONAS = (
    "curious", "skeptical", "frustrated", "enthusiastic", "terse-expert", "casual",
)
#: Verbosity targets - from a one-line ask to a long, multi-sentence message.
VERBOSITIES = ("short", "medium", "long")

_STOP = _TOPIC_STOPWORDS | {"this", "that", "these", "those", "it", "its", "as", "be"}

# User-turn phrasings that betray role confusion (fix E): claiming authorship of
# the assistant's output, offering to do the assistant's job, inverting roles. A
# generated turn matching any of these is regenerated.
_ROLE_CONFUSION_RE = re.compile(
    r"(?:\b(?:the|that|this)\s+(?:\w+\s+){0,2}i\s+(?:made|wrote|created|generated|produced|built|reformatted)\b)"
    r"|(?:\bmy\s+(?:rewrite|rephrasing|reformatted version|reformatted answer)\b)"
    r"|(?:\bi(?:'ll| will| can|'d)\s+(?:redo|rewrite|rephrase|reformat|regenerate)\b)"
    r"|(?:\bi(?:'ll| will| can|'d)\s+try\s+(?:again|rephrasing|rewriting|to rephrase|to rewrite|a rephrasing)\b)"
    r"|(?:\b(?:ask me (?:a|your|the next)|pose (?:a|your|me a) question|give me your (?:next )?question|quiz me)\b)"
    r"|(?:\bas an ai\b)",
    re.IGNORECASE,
)


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
        topic: str = "",
        seed_allowed_intents: Optional[set[str]] = None,
    ):
        self.sampler = sampler
        self.planner = planner
        self.client = client
        self.rng = rng or random.Random()
        self.lang = lang
        self.min_recall_distance = min_recall_distance
        self.max_intent_resamples = max_intent_resamples
        #: curator topic (preferred over the stopword-stripped fallback).
        self.topic = topic
        #: sensitive seeds restrict the intent pool to a safe subset (``None`` = all).
        self.seed_allowed_intents = set(seed_allowed_intents) if seed_allowed_intents else None
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
        turn_plan: Optional[TurnPlan] = None,
        target_length: Optional[int] = None,
    ) -> SimulatorResult:
        """Emit one grounded user message for ``turn_index``.

        ``turn_plan`` (from the conversation planner) selects the composition
        category and, for memory turns, the sub-kind. When omitted the simulator
        samples freely (back-compat). Memory turns always return a result (they
        fall back to a recall-of-content turn); other categories resample on
        invalid draws and fall back to a recall-of-content turn.
        """
        if self.client is None:
            raise RuntimeError(
                "UserSimulator requires an LLM client - it is fully LLM-driven. "
                "Pass a (mock) client; there is no hardcoded template fallback.")
        active = list(active_constraints or [])
        avoid = set(avoid_intents or set())
        topic = self.topic or self._topic_phrase(prior_turns)

        if turn_plan is not None and turn_plan.category == "memory":
            return self._memory_turn(
                prior_turns, prompt_pack, turn_index, turn_plan, topic)

        allowed = self._allowed_for(turn_plan)
        for _ in range(self.max_intent_resamples):
            draw = forced_draw or self.sampler.sample(
                self.lang, rng=self.rng, allowed_intents=allowed)
            forced_draw = None  # honour the forced draw only on the first attempt
            if draw.intent in avoid:
                continue
            if not self._temporally_valid(draw, turn_index, active):
                continue
            result = C.build(self._ctx(draw, turn_index))
            if result.resample or result.constraint_spec is None:
                continue
            sim = self._generate(
                result.constraint_spec, draw, prior_turns, prompt_pack, topic,
                turn_index, directive=result.directive)
            if sim is not None:
                return sim

        return self._recall_content_turn(prior_turns, prompt_pack, turn_index, topic)

    # -------------------------------------------------------------- allowed set
    def _allowed_for(self, turn_plan: Optional[TurnPlan]) -> Optional[set[str]]:
        """Intents permitted this turn: category ∩ seed allow-set (``None`` = all)."""
        cat = COMPOSITION_CATEGORIES.get(turn_plan.category) if turn_plan else None
        sets = [s for s in (cat, self.seed_allowed_intents) if s is not None]
        if not sets:
            return None
        allowed = set(sets[0])
        for s in sets[1:]:
            allowed &= set(s)
        return allowed or None

    # ----------------------------------------------------------- memory turns
    def _memory_turn(
        self, prior_turns: list[Turn], prompt_pack: PromptPack, turn_index: int,
        turn_plan: TurnPlan, topic: str,
    ) -> SimulatorResult:
        """Realise a memory turn; recall-of-content is the default and the fallback."""
        kind = turn_plan.memory_kind or "content"
        facts_allowed = (self.seed_allowed_intents is None
                         or bool(FACT_INTENTS & self.seed_allowed_intents))
        if not facts_allowed:
            kind = "content"  # sensitive seeds: no personal-fact planting/recall

        if kind == "plant":
            res = self._fact_turn(SamplerDraw("fact_recall", "immediate"),
                                  prior_turns, prompt_pack, turn_index, topic)
            if res is not None:
                return res
        elif kind == "recall":
            res = self._fact_turn(SamplerDraw("fact_recall", "delayed_recall"),
                                  prior_turns, prompt_pack, turn_index, topic)
            if res is not None:
                return res
        elif kind == "update":
            res = self._fact_turn(SamplerDraw("fact_update", "immediate"),
                                  prior_turns, prompt_pack, turn_index, topic)
            if res is not None:
                return res
        return self._recall_content_turn(prior_turns, prompt_pack, turn_index, topic)

    def _fact_turn(
        self, draw: SamplerDraw, prior_turns: list[Turn], prompt_pack: PromptPack,
        turn_index: int, topic: str,
    ) -> Optional[SimulatorResult]:
        result = C.build(self._ctx(draw, turn_index))
        if result.resample or result.constraint_spec is None:
            return None
        return self._generate(result.constraint_spec, draw, prior_turns,
                              prompt_pack, topic, turn_index,
                              directive=result.directive)

    def _recall_content_turn(
        self, prior_turns: list[Turn], prompt_pack: PromptPack, turn_index: int,
        topic: str,
    ) -> SimulatorResult:
        """A natural recall of real prior content (the default memory test)."""
        spec = ConstraintSpec(
            intent="recall_content", type="recall_content", params={},
            lang=self.lang, verifier="llm_judge", scope="current_turn")
        draw = SamplerDraw("recall_content", "immediate")
        persona = self._sample_persona()
        verbosity = self.rng.choice(VERBOSITIES)
        steer = {"op": "recall_content", "topic": topic}
        prompt = self._build_prompt(prior_turns, persona, verbosity, steer)
        text = self._call_llm(prompt, prompt_pack) or (
            f"Earlier you touched on {topic} - can you say more about that point?")
        return SimulatorResult(
            nl_query=text, constraint_spec=spec, draw=draw,
            grounding="recalls_content", topic=topic,
            persona=persona, verbosity=verbosity)

    # -------------------------------------------------------------- validity
    def _temporally_valid(
        self, draw: SamplerDraw, turn_index: int, active: list[ConstraintSpec],
    ) -> bool:
        """Reject policies that have nothing to operate on at this turn (§4.4)."""
        policy = draw.policy
        if policy == "cumulative":
            return len(active) >= 1
        if policy == "standing":
            return turn_index >= 1
        if policy == "delayed_recall":
            if draw.intent in FACT_INTENTS:
                fact = self.planner.recallable_fact(
                    turn_index, self.min_recall_distance, require_injected=True)
                return fact is not None
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
        turn_index: int = 0,
        directive: str = "",
    ) -> Optional[SimulatorResult]:
        """Drive the simulator LLM to write one grounded, coherent user message."""
        intent, policy = draw.intent, draw.policy
        grounding = self._grounding_for(intent, policy)

        # A delayed recall must target an already-injected fact (no desync).
        if grounding == "recalls_fact" and not self.planner.ledger.is_injected(
                spec.fact_id or ""):
            return None

        persona = self._sample_persona()
        verbosity = self.rng.choice(VERBOSITIES)
        steer = self._steer(intent, policy, spec, topic, directive)
        prompt = self._build_prompt(prior_turns, persona, verbosity, steer)

        for _ in range(2):  # initial generation + one regeneration
            text = self._call_llm(prompt, prompt_pack)
            if self._coherence_ok(text, intent, policy, spec, topic):
                # NB: the ledger plant/update commit is intentionally NOT done here.
                # A generated query can still be discarded by the loop if the
                # *responder* fails verification and the intent is resampled; the
                # loop therefore commits the ledger mutation (mark-injected / value
                # overwrite) only once the turn is accepted (see
                # ConversationLoop._commit_fact_turn) - so a discarded fact turn
                # never leaves a phantom value behind.
                return SimulatorResult(
                    nl_query=text, constraint_spec=spec, draw=draw,
                    grounding=grounding, topic=topic,
                    persona=persona, verbosity=verbosity)
        return None

    def _sample_persona(self) -> str:
        """Sample a persona, avoiding immediate repetition (diversity)."""
        choices = [p for p in PERSONAS if p != self._last_persona] or list(PERSONAS)
        persona = self.rng.choice(choices)
        self._last_persona = persona
        return persona

    # --------------------------------------------------------------- steer
    def _steer(
        self, intent: str, policy: str, spec: ConstraintSpec, topic: str,
        directive: str,
    ) -> dict[str, str]:
        grounding = self._grounding_for(intent, policy)
        if grounding == "plants_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            return {"op": "plant_fact", "fact_label": _readable(fact.fact_type),
                    "fact_value": str(spec.params.get("value", fact.value)),
                    "topic": topic}
        if grounding == "recalls_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            # NB: no fact_value - the user must NOT restate the value when recalling.
            return {"op": "recall_fact", "fact_label": _readable(fact.fact_type)}
        if grounding == "updates_fact":
            fact = self.planner.ledger.get(spec.fact_id)  # type: ignore[arg-type]
            return {"op": "update_fact", "fact_label": _readable(fact.fact_type),
                    "fact_value": str(spec.params.get("value", fact.value))}
        if grounding == "recalls_content":
            return {"op": "recall_content", "topic": topic}
        if intent == "open_chat":
            return {"op": "continue_topic", "topic": topic,
                    "say": f"keep the conversation going about {topic} with an open, "
                           f"curious follow-up"}
        # transformation / operates-on-prior: the builder's directive is the steer.
        return {"op": "request_constraint",
                "say": directive or "revise your previous answer"}

    def _build_prompt(
        self, prior_turns: list[Turn], persona: str, verbosity: str,
        steer: dict[str, str],
    ) -> str:
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
                f"- This turn: while staying on {steer.get('topic', 'the topic')}, "
                f"mention in passing - as a real personal detail - that your "
                f"{steer['fact_label']} is {steer['fact_value']}. State that exact "
                f"value, woven naturally into a topical message (not a bare 'my X is Y').")
        elif op == "recall_fact":
            lines.append(
                f"- This turn: ask the assistant to remind you of YOUR own "
                f"{steer['fact_label']} (a personal detail you mentioned earlier). "
                f"Phrase it as 'my {steer['fact_label']}', never 'our'. Do NOT restate "
                f"the value yourself - you are testing whether it remembers.")
        elif op == "update_fact":
            lines.append(
                f"- This turn: tell the assistant your {steer['fact_label']} is now "
                f"{steer['fact_value']} and ask it to confirm the current value. "
                f"State that exact new value.")
        elif op == "recall_content":
            lines.append(
                f"- This turn: refer back to one specific thing the assistant told "
                f"you earlier in this conversation (name it), and ask it to expand on "
                f"that point or relate it to {steer.get('topic', 'the topic')}. Do not "
                f"start a brand-new unrelated topic.")
        else:  # request_constraint / continue_topic
            lines.append(f"- Your request to the assistant this turn: {steer.get('say', '')}.")
        lines.append(
            "\nOutput only the user's message - no labels, no quotes, do not answer "
            "your own question.")
        # Machine-readable steer for the deterministic test client (a real LLM
        # simply follows the prose above).
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
        try:
            resp = self.client.generate(  # type: ignore[union-attr]
                prompt, system_prompt=prompt_pack.simulator_system, temperature=0.9)
            # strip any glued trailing generation artifact (failure mode C) from the
            # user query before it is verified / stored.
            return sanitize_generated_text(resp.text or "") or ""
        except Exception:
            return ""

    # ----------------------------------------------------------- coherence
    def _coherence_ok(
        self, text: str, intent: str, policy: str, spec: ConstraintSpec, topic: str,
    ) -> bool:
        if not text or not text.strip():
            return False
        # The simulated user must never claim authorship of the assistant's output,
        # offer to do its job, or invert roles (fix E). Reject and regenerate.
        if _ROLE_CONFUSION_RE.search(text):
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

        if grounding == "recalls_content":
            return self._references_prior(text, topic)

        if intent == "open_chat" and topic:
            return topic.split()[0].lower() in low

        # restyle / expand / deepen / self_critique / chained_compute: grounded by
        # construction (they operate on the prior answer); accept any non-empty turn.
        return True

    @staticmethod
    def _references_prior(text: str, topic: str) -> bool:
        """A recall-of-content turn should reference earlier content or the topic."""
        low = text.lower()
        topic_words = [w for w in re.findall(r"[a-z0-9']+", topic.lower()) if w not in _STOP]
        if any(w in low for w in topic_words):
            return True
        # otherwise require an explicit back-reference cue
        return bool(re.search(
            r"\b(earlier|before|you (?:said|mentioned|noted|wrote)|previously|"
            r"that point|you brought up|going back)\b", low))

    # --------------------------------------------------------------- grounding
    @staticmethod
    def _grounding_for(intent: str, policy: str) -> str:
        if intent == "recall_content":
            return "recalls_content"
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
# spec -> markers (used to verify a verifiable-constraint query asks for it)
# ---------------------------------------------------------------------------

def _readable(fact_type: str) -> str:
    return fact_type.replace("_", " ")


def _constraint_markers(spec: ConstraintSpec) -> list[str]:
    """Tokens at least one of which must appear in a verifiable-constraint query."""
    t, p = spec.type, spec.params
    table: dict[str, list[str]] = {
        "json_valid": ["json"],
        "yaml_valid": ["yaml"],
        "markdown_table": ["table"],
        "markdown_format": ["markdown"],
        "first_letter": [f"'{p.get('letter', '')}'", f"letter {p.get('letter', '')}",
                         f"start with {p.get('letter', '')}"],
        "alphabetical_sentence_starts": ["alphabetical"],
        "max_words_per_sentence": [str(p.get("max_words", "")), "words per sentence",
                                   "short sentence"],
        # require the forbidden word to be named (or an explicit "the word ..."),
        # not a coincidental "without" elsewhere in the message.
        "forbidden_token": [f"'{p.get('token', '')}'", str(p.get("token", "")),
                            "the word"],
        "no_gendered_pronouns": ["pronoun", "gender"],
        "length_tokens": [str(p.get("max_words", "")), "words", "shorter", "concise"],
        "n_bullets": [str(p.get("n", "")), "bullet"],
        "limerick_structure": ["limerick"],
        "genre": [str(p.get("genre", ""))],
        "style": [str(p.get("style", "")).split()[0] if p.get("style") else ""],
    }
    return table.get(t, [])
