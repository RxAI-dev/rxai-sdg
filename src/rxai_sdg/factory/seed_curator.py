"""Seed curation (spec §5.1).

The :class:`SeedCurator` turns the raw seeds you provide - a list of prompt
strings, or dicts with a ``"query"`` field - into typed :class:`Seed` objects.
It extracts the first user query, deduplicates near-identical seeds, **infers**
the category/domain (a keyword heuristic with an optional LLM fallback), flags
long "haystack" seeds, and loads the per-category :class:`PromptPack`.

There is intentionally no dataset-loading abstraction here: load your seeds
however you like (e.g. ``list(load_dataset(...)["query"])``) and pass them in.
"""

from __future__ import annotations

import random
import re
from typing import Any, Iterable, Optional, Union

from .schemas import Seed
from .prompts import PromptPack, get_prompt_pack

EVAL_CATEGORIES = [
    "writing", "math", "coding", "extraction", "stem", "humanities",
    "reasoning", "roleplay",
]

# A seed whose first query is at least this many characters is treated as a
# "haystack" seed (interaction 1 is a long document) to preserve needle-retrieval
# coverage. A dict seed may also set ``"is_haystack": True`` explicitly.
HAYSTACK_MIN_CHARS = 1500

# Keyword-based domain tagging. Heuristic; an explicit ``"category"`` on a dict
# seed always wins, and the optional LLM classifier handles the rest.
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

SeedInput = Union[str, dict]


class SeedCurator:
    def __init__(
        self,
        rng: Optional[random.Random] = None,
        dedup_threshold: float = 0.9,
        classifier_client: Any = None,
    ):
        self.rng = rng or random.Random()
        self.dedup_threshold = dedup_threshold
        self.classifier_client = classifier_client

    # ------------------------------------------------------------------ public
    def curate(self, seeds: Iterable[SeedInput], lang: str = "en") -> list[Seed]:
        """Normalise, dedup and tag the provided seeds."""
        out: list[Seed] = []
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
            category = rec.get("category") or self.infer_category(query)
            out.append(Seed(
                dataset=rec.get("dataset", "seeds"),
                first_query=query,
                category=category,
                domain=category,
                lang=rec.get("lang") or lang,
                is_haystack=bool(rec.get("is_haystack")) or len(query) >= HAYSTACK_MIN_CHARS,
            ))
        return out

    def load_prompt_pack(self, seed: Seed) -> PromptPack:
        return get_prompt_pack(seed.category, seed.lang)

    def infer_category(self, query: str) -> str:
        """Infer an eval category. Keyword heuristic, then optional LLM fallback."""
        category = self._infer_domain(query)
        if category == "general" and self.classifier_client is not None:
            category = self._classify_llm(query)
        return category

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

    def _classify_llm(self, query: str) -> str:
        options = ", ".join(EVAL_CATEGORIES)
        prompt = (
            f"Classify the following user request into exactly one of these "
            f"categories: {options}. Reply with only the single category word.\n\n"
            f"Request: {query}")
        try:
            resp = self.classifier_client.generate(
                prompt, system_prompt="You are a precise text classifier.",
                temperature=0.0, max_tokens=8)
            text = (resp.text or "").strip().lower()
        except Exception:
            return "general"
        for cat in EVAL_CATEGORIES:
            if cat in text:
                return cat
        return "general"
