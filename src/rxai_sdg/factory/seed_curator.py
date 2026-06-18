"""Seed curation (spec §5.1, fix A).

The :class:`SeedCurator` turns the raw seeds you provide - a list of prompt
strings, or dicts with a ``"query"`` field - into typed :class:`Seed` objects
paired with a :class:`SeedDirective`. The directive is produced by an **LLM
curator** (the ``CURATOR_MODEL``) from the *first query only* plus a steering
prompt (cheap), replacing the old rule-based keyword classifier:

* ``domain`` / ``topic`` - a short grounding phrase used by the simulator
  (replaces the crude stopword-stripping topic heuristic).
* ``action`` - ``keep`` | ``skip``. Contentless seeds (bare greetings, "hi",
  "what's up doc?"), low-information or malformed seeds are **skipped**.
* ``sensitivity`` - ``none`` | ``sensitive``. Mental-health, self-harm, crisis,
  medical, grief and similar seeds are flagged ``sensitive`` and **kept** (we want
  representation) but restricted to a safe, supportive intent subset.

When no curator client is supplied (or a call fails) a transparent heuristic
fallback runs so the package stays usable offline and unit-testable.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union

from .schemas import Seed
from .prompts import PromptPack, get_prompt_pack, CURATOR_SYSTEM
from .taxonomy import SENSITIVE_ALLOWED_INTENTS

EVAL_CATEGORIES = [
    "writing", "math", "coding", "extraction", "stem", "humanities",
    "reasoning", "roleplay",
]

# A seed whose first query is at least this many characters is treated as a
# "haystack" seed (interaction 1 is a long document) to preserve needle-retrieval
# coverage. A dict seed may also set ``"is_haystack": True`` explicitly.
HAYSTACK_MIN_CHARS = 1500

# Keyword-based domain tagging (heuristic fallback only).
_DOMAIN_KEYWORDS = {
    "math": ["calculate", "equation", "integral", "probability", "sum of", "derivative",
             "solve for", "prime", "factor", "%", "how many"],
    "coding": ["python", "javascript", "function", "code", "bug", "compile", "regex",
               "algorithm", "class ", "api"],
    "extraction": ["extract", "list all", "parse", "from the following", "table of",
                   "json", "summarize the"],
    "stem": ["physics", "chemistry", "biology", "molecule", "force", "cell", "energy",
             "quantum"],
    "humanities": ["history", "philosophy", "ethics", "literature", "society", "culture"],
    "reasoning": ["why", "logic", "puzzle", "deduce", "if all", "infer", "step by step"],
    "roleplay": ["pretend", "role-play", "roleplay", "you are a", "act as", "imagine you"],
    "writing": ["write", "poem", "story", "essay", "draft", "compose"],
}

# Contentless / low-information openers the curator should skip. Used by the
# heuristic fallback and re-used by the ``contentless_seed_kept`` detector.
CONTENTLESS_RE = re.compile(
    r"^\s*(?:hi+|hey+|hello+|yo+|sup|what'?s up(?:\s+doc)?|good (?:morning|afternoon|evening)"
    r"|howdy|greetings|test|ping|knock knock|anyone (?:there|home)\??)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# Sensitive-topic markers (heuristic fallback). Mental-health / self-harm /
# crisis / grief / medical-distress / minors. The LLM curator is the primary
# signal; this only catches the obvious cases when offline.
SENSITIVE_RE = re.compile(
    r"\b(suicid\w*|kill myself|end my life|self[- ]?harm|hurt(?:ing)? myself|hopeless|"
    r"want to die|don'?t want to (?:be here|live)|worthless|cutting myself|overdose|"
    r"depress\w*|anxiety|panic attack|grie(?:f|ving)|abuse[d]?|assault\w*|trauma\w*|"
    r"crisis|mental health|eating disorder|anorexi\w*|bulimi\w*|relapse)\b",
    re.IGNORECASE,
)

SeedInput = Union[str, dict]


@dataclass
class SeedDirective:
    """Curation directives that steer generation for one seed."""

    domain: str = "general"
    topic: str = "this topic"
    action: str = "keep"            # keep | skip
    sensitivity: str = "none"       # none | sensitive
    #: when set, the sampler may only draw intents from this allow-set.
    allowed_intents: Optional[list[str]] = None
    #: per-conversation, topic-grounded personal facts the user might share, each
    #: ``{"label", "value", "new_value"}`` (fix: no hardcoded fact pool). The LLM
    #: curator samples these freshly per seed, so memory plants are diverse across a
    #: large dataset rather than recycling a fixed handful of values.
    facts: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain, "topic": self.topic, "action": self.action,
            "sensitivity": self.sensitivity, "allowed_intents": self.allowed_intents,
            "facts": self.facts,
        }


@dataclass
class CuratedSeed:
    """A curated :class:`Seed` bundled with its :class:`SeedDirective`."""

    seed: Seed
    directive: SeedDirective


class SeedCurator:
    def __init__(
        self,
        rng: Optional[random.Random] = None,
        dedup_threshold: float = 0.9,
        client: Any = None,
        classifier_client: Any = None,
        max_tokens: int = 2048,
    ):
        self.rng = rng or random.Random()
        self.dedup_threshold = dedup_threshold
        #: the LLM curator (CURATOR_MODEL); ``classifier_client`` kept as an alias.
        self.client = client or classifier_client
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ public
    def curate(self, seeds: Iterable[SeedInput], lang: str = "en") -> list[CuratedSeed]:
        """Normalise, dedup, classify (LLM, in parallel) and tag the provided seeds.

        Contentless / skip seeds are dropped; the rest are returned as
        :class:`CuratedSeed` (seed + directive) in input order.
        """
        # 1) extract + dedup (sequential, cheap)
        kept: list[dict[str, Any]] = []
        seen_norms: list[set[str]] = []
        for raw in seeds:
            rec = {"query": raw} if isinstance(raw, str) else dict(raw)
            query = self._extract_query(rec)
            if not query:
                continue
            norm = self._normalise_tokens(query)
            if self._is_near_duplicate(norm, seen_norms):
                continue
            seen_norms.append(norm)
            rec["__query"] = query
            kept.append(rec)

        # 2) classify (the LLM curator call is independent per seed -> parallel)
        queries = [rec["__query"] for rec in kept]
        directives = self._classify_many(queries, lang)

        # 3) build, dropping skips
        out: list[CuratedSeed] = []
        for rec, directive in zip(kept, directives):
            if rec.get("category"):
                directive.domain = rec["category"]
            if directive.action == "skip":
                continue
            query = rec["__query"]
            seed = Seed(
                dataset=rec.get("dataset", "seeds"),
                first_query=query,
                category=directive.domain,
                domain=directive.domain,
                lang=rec.get("lang") or lang,
                is_haystack=bool(rec.get("is_haystack")) or len(query) >= HAYSTACK_MIN_CHARS,
            )
            out.append(CuratedSeed(seed=seed, directive=directive))
        return out

    def _classify_many(self, queries: list[str], lang: str) -> list[SeedDirective]:
        if not queries:
            return []
        if self.client is None or len(queries) == 1:
            return [self.classify_seed(q, lang=lang) for q in queries]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(queries), 16)) as ex:
            return list(ex.map(lambda q: self.classify_seed(q, lang=lang), queries))

    def load_prompt_pack(self, seed: Seed) -> PromptPack:
        return get_prompt_pack(seed.category, seed.lang)

    # --------------------------------------------------------------- classify
    def classify_seed(self, query: str, lang: str = "en") -> SeedDirective:
        """Return a :class:`SeedDirective` for ``query`` (LLM, heuristic fallback)."""
        directive = None
        if self.client is not None:
            directive = self._classify_llm(query)
        if directive is None:
            directive = self._classify_heuristic(query)
        # allowed_intents is derived deterministically from sensitivity so the
        # sampler restriction never depends on the LLM enumerating intents.
        if directive.sensitivity == "sensitive":
            directive.allowed_intents = sorted(SENSITIVE_ALLOWED_INTENTS)
        else:
            directive.allowed_intents = None
        return directive

    def infer_category(self, query: str) -> str:
        """Back-compat: infer an eval category for ``query`` (heuristic)."""
        return self._classify_heuristic(query).domain

    # ----------------------------------------------------------------- LLM
    def _classify_llm(self, query: str) -> Optional[SeedDirective]:
        cats = ", ".join(EVAL_CATEGORIES)
        user = (
            "Classify this opening user message for a conversation dataset.\n\n"
            f"USER MESSAGE:\n{query}\n\n"
            "Return ONLY a JSON object with keys:\n"
            f"  \"domain\": one short word for the subject area (one of: {cats}, or general),\n"
            "  \"topic\": a 2-6 word noun phrase naming what the message is about "
            "(used to keep follow-ups on topic),\n"
            "  \"action\": \"keep\" for a substantive message, \"skip\" for a contentless "
            "greeting / empty / malformed message that cannot start a real conversation,\n"
            "  \"sensitivity\": \"sensitive\" for mental-health, self-harm, crisis, "
            "medical, grief, abuse or minors topics, otherwise \"none\",\n"
            "  \"facts\": a list of EXACTLY 3 personal details that a real person having "
            "THIS conversation might plausibly mention in passing, used later to test the "
            "assistant's memory. Each item is an object with:\n"
            "      \"label\": a short noun phrase completing 'my ___' (e.g. \"pet's name\", "
            "\"home town\", \"the project I'm building\", \"favorite cuisine\"),\n"
            "      \"value\": a SPECIFIC, DISTINCTIVE value (a proper noun, name, place, or "
            "exact number) - invent a fresh, uncommon one, never a generic word,\n"
            "      \"new_value\": a DIFFERENT specific value of the same kind (used if the "
            "user later changes it).\n"
            "    Make the details varied and natural for this topic. They are personal "
            "details a user shares - NEVER account numbers, subscription tiers, passwords, "
            "billing, or any system/account data."
        )
        try:
            resp = self.client.generate(
                user, system_prompt=CURATOR_SYSTEM, temperature=0.7,
                max_tokens=self.max_tokens)
            data = _extract_json(resp.text or "")
        except Exception:
            return None
        if not data:
            return None
        domain = str(data.get("domain", "general")).strip().lower() or "general"
        topic = str(data.get("topic", "") or "").strip() or self._topic_fallback(query)
        action = str(data.get("action", "keep")).strip().lower()
        action = "skip" if action.startswith("skip") else "keep"
        sens = str(data.get("sensitivity", "none")).strip().lower()
        sensitivity = "sensitive" if sens.startswith("sens") else "none"
        return SeedDirective(
            domain=domain, topic=topic, action=action, sensitivity=sensitivity,
            facts=_clean_facts(data.get("facts")))

    # ----------------------------------------------------------- heuristic
    def _classify_heuristic(self, query: str) -> SeedDirective:
        if CONTENTLESS_RE.match(query.strip()) or len(query.strip()) < 6:
            return SeedDirective(domain="general", topic=self._topic_fallback(query),
                                 action="skip")
        sensitivity = "sensitive" if SENSITIVE_RE.search(query) else "none"
        return SeedDirective(
            domain=self._infer_domain(query), topic=self._topic_fallback(query),
            action="keep", sensitivity=sensitivity)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _extract_query(rec: dict[str, Any]) -> Optional[str]:
        for key in ("query", "prompt", "question", "instruction", "text"):
            if isinstance(rec.get(key), str) and rec[key].strip():
                return rec[key].strip()
        msgs = rec.get("messages") or rec.get("conversation")
        if isinstance(msgs, list) and msgs:
            first = msgs[0]
            if isinstance(first, dict) and isinstance(first.get("content"), str):
                return first["content"].strip()
        return None

    @staticmethod
    def _normalise_tokens(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def _is_near_duplicate(self, norm: set[str], seen: list[set[str]]) -> bool:
        if not norm:
            return True
        for other in seen:
            if not other:
                continue
            inter = len(norm & other)
            union = len(norm | other)
            if union and inter / union >= self.dedup_threshold:
                return True
        return False

    @staticmethod
    def _infer_domain(query: str) -> str:
        low = query.lower()
        best, best_hits = "general", 0
        for domain, kws in _DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in low)
            if hits > best_hits:
                best, best_hits = domain, hits
        return best

    # Words stripped when deriving a fallback "running topic" phrase.
    _TOPIC_STOPWORDS = {
        "explain", "describe", "write", "what", "why", "how", "outline", "give",
        "tell", "summarize", "summarise", "compute", "calculate", "reverse", "is",
        "are", "the", "a", "an", "of", "to", "me", "please", "could", "would", "you",
        "your", "about", "and", "or", "in", "on", "for", "with", "do", "does", "can",
    }

    @classmethod
    def _topic_fallback(cls, query: str) -> str:
        words = re.findall(r"[A-Za-z0-9']+", query or "")
        kept = [w for w in words if w.lower() not in cls._TOPIC_STOPWORDS]
        phrase = " ".join(kept[:6]).strip()
        return phrase or (" ".join(words[:6]).strip() or "this topic")


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from ``text`` (tolerant of fences / prose)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# Account / subscription / billing / system framing that must never be a personal
# fact (it correctly triggers "I can't access your account" refusals).
_BAD_FACT_VALUE_RE = re.compile(
    r"\b(account|subscription|billing|invoice|payment|password|api[ _-]?key|"
    r"access[ _-]?token|membership|premium|tier[- ]?\d|order[ _-]?number|"
    r"credit[ _-]?card|ssn|social security|license[ _-]?key)\b",
    re.IGNORECASE,
)


def _clean_facts(raw: Any) -> list[dict[str, str]]:
    """Validate/sanitise the curator's facts into ``{label,value,new_value}`` items."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("kind") or "").strip()
        # the label completes "my ___", so drop a leading article/possessive the
        # curator sometimes adds ("the recipe book" / "my home town" -> ...).
        label = re.sub(r"^(?:the|my|your|a|an)\s+", "", label, flags=re.IGNORECASE).strip()
        value = str(item.get("value") or "").strip()
        new_value = str(item.get("new_value") or "").strip()
        if not label or len(value) < 2:
            continue
        blob = f"{label} {value} {new_value}"
        if _BAD_FACT_VALUE_RE.search(blob):
            continue  # not a personal detail
        if value.lower() in seen_values:
            continue
        seen_values.add(value.lower())
        out.append({"label": label, "value": value, "new_value": new_value})
        if len(out) >= 4:
            break
    return out
