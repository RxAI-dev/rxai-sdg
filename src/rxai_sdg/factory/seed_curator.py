"""Seed curation (spec §5.1).

The :class:`SeedCurator` turns raw dataset records into typed :class:`Seed`
objects: it extracts the first user query, deduplicates near-identical seeds,
tags ``category``/``domain``/``lang``, flags ``is_haystack`` seeds, and loads the
relevant :class:`PromptPack`. It also supports balancing domain coverage toward
the eval categories and injecting extra math/reasoning seeds.

Dataset loading is abstracted behind ``dataset_spec`` so the curator works with
plain Python iterables (used in tests) or, lazily, with a HuggingFace dataset.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from .schemas import Seed
from .prompts import PromptPack, get_prompt_pack

EVAL_CATEGORIES = [
    "writing", "math", "coding", "extraction", "stem", "humanities",
    "reasoning", "roleplay",
]

# Lightweight keyword-based domain tagging. This is a heuristic fallback; an
# explicit ``category`` field on a record always wins.
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


@dataclass
class DatasetSpec:
    """How to obtain seeds.

    Exactly one source is used, checked in order: ``records`` (an in-memory
    iterable of dicts), then ``hf_dataset`` (loaded lazily via ``datasets``).
    """

    name: str = "in_memory"
    records: Optional[Iterable[dict[str, Any]]] = None
    hf_dataset: Optional[str] = None
    hf_split: str = "train"
    #: name of the field holding the first user query
    query_field: str = "query"
    #: optional explicit category field on each record
    category_field: Optional[str] = "category"
    lang: str = "en"
    haystack_fraction: float = 0.0
    #: a record is a haystack seed when its query length exceeds this (chars)
    haystack_min_chars: int = 1500


class SeedCurator:
    def __init__(
        self,
        rng: Optional[random.Random] = None,
        prompt_pack_loader: Callable[[str, str], PromptPack] = get_prompt_pack,
        dedup_threshold: float = 0.9,
    ):
        self.rng = rng or random.Random()
        self.prompt_pack_loader = prompt_pack_loader
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------ public
    def load_seeds(self, dataset_spec: DatasetSpec) -> list[Seed]:
        raw = list(self._iter_records(dataset_spec))
        seeds: list[Seed] = []
        seen_norms: list[set[str]] = []
        for rec in raw:
            query = self._extract_query(rec, dataset_spec.query_field)
            if not query:
                continue
            norm = self._normalise_tokens(query)
            if self._is_near_duplicate(norm, seen_norms):
                continue
            seen_norms.append(norm)
            seeds.append(self._build_seed(rec, query, dataset_spec))
        return seeds

    def load_prompt_pack(self, seed: Seed) -> PromptPack:
        return self.prompt_pack_loader(seed.category, seed.lang)

    def balance_domains(
        self,
        seeds: list[Seed],
        domain_mix: dict[str, float],
        rng: Optional[random.Random] = None,
    ) -> list[Seed]:
        """Resample ``seeds`` to approximate the configured ``domain_mix``.

        Domains with no seeds are skipped. The result preserves the total count
        where possible by sampling (with replacement when a domain is short).
        """
        rng = rng or self.rng
        by_domain: dict[str, list[Seed]] = {}
        for s in seeds:
            by_domain.setdefault(s.domain, []).append(s)
        present = {d: w for d, w in domain_mix.items() if by_domain.get(d)}
        if not present:
            return list(seeds)
        total = len(seeds)
        weight_sum = sum(present.values())
        out: list[Seed] = []
        for domain, weight in present.items():
            n = max(1, round(total * weight / weight_sum))
            pool = by_domain[domain]
            for _ in range(n):
                out.append(rng.choice(pool))
        rng.shuffle(out)
        return out

    def inject_seeds(self, seeds: list[Seed], extra: list[Seed]) -> list[Seed]:
        """Append explicitly provided extra seeds (e.g. math/reasoning)."""
        return list(seeds) + list(extra)

    # ----------------------------------------------------------------- helpers
    def _iter_records(self, spec: DatasetSpec) -> Iterator[dict[str, Any]]:
        if spec.records is not None:
            yield from spec.records
            return
        if spec.hf_dataset is not None:  # pragma: no cover - exercised only with datasets installed
            from datasets import load_dataset  # lazy import
            ds = load_dataset(spec.hf_dataset, split=spec.hf_split)
            for rec in ds:
                yield dict(rec)
            return
        raise ValueError("DatasetSpec must provide either records or hf_dataset")

    @staticmethod
    def _extract_query(rec: dict[str, Any], field_name: str) -> Optional[str]:
        if field_name in rec and isinstance(rec[field_name], str):
            return rec[field_name].strip()
        # common conversational layouts
        for key in ("query", "prompt", "question", "instruction", "text"):
            if isinstance(rec.get(key), str):
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

    def _build_seed(self, rec: dict[str, Any], query: str, spec: DatasetSpec) -> Seed:
        category = None
        if spec.category_field and isinstance(rec.get(spec.category_field), str):
            category = rec[spec.category_field]
        if not category:
            category = self._infer_domain(query)
        lang = rec.get("lang") or spec.lang
        is_haystack = bool(rec.get("is_haystack"))
        if not is_haystack and spec.haystack_fraction > 0:
            if len(query) >= spec.haystack_min_chars or self.rng.random() < spec.haystack_fraction:
                is_haystack = len(query) >= spec.haystack_min_chars
        return Seed(
            dataset=spec.name,
            first_query=query,
            category=category,
            domain=category,
            lang=lang,
            is_haystack=is_haystack,
        )

    @staticmethod
    def _infer_domain(query: str) -> str:
        low = query.lower()
        best, best_hits = "general", 0
        for domain, kws in _DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in low)
            if hits > best_hits:
                best, best_hits = domain, hits
        return best
