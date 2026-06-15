"""High-level orchestration: :class:`DataFactory` (spec §2, §11.10).

Workflow::

    factory = DataFactory(config, responder_client, simulator_client)
    seeds = ["Explain entropy.", {"query": "Reverse a linked list."}]  # str or dict
    records = factory.generate(seeds)        # one conversation per seed
    factory.save_to_hub("org/rxt-factory", token="hf_...")

``generate`` curates the seeds (rule-based category tagging), then runs the
conversation loop **concurrently** across seeds with a ``ThreadPoolExecutor``
(conversations are independent; the loop inside one conversation stays
sequential). Each conversation gets its **own** ``Random(seed + index)`` so output
is reproducible regardless of thread scheduling, and per-conversation stats are
merged under a lock. Throughput is bounded by the saturated inference replica, not
the session count: a single process at high ``config.concurrency`` should saturate
the endpoint.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .clients import LLMClient
from .config import FactoryConfig
from .dataset import FactoryDatasetPostprocessor
from .holistic import HolisticJudge
from .loop import ConversationLoop, LoopStats
from .responder import Responder
from .sampler import IntentPolicySampler
from .schemas import ConversationRecord, Seed
from .seed_curator import SeedCurator, SeedInput
from .writer import SegmentWriter


@dataclass
class FactoryRunStats:
    seeds_used: int = 0
    conversations_built: int = 0
    conversations_discarded: int = 0
    records_emitted: int = 0
    loop: LoopStats = field(default_factory=LoopStats)


class DataFactory:
    def __init__(
        self,
        config: FactoryConfig,
        responder_client: LLMClient,
        simulator_client: Optional[LLMClient] = None,
        holistic_client: Optional[LLMClient] = None,
        rng: Optional[random.Random] = None,
    ):
        self.config = config
        self.rng = rng or random.Random(config.seed)
        self.taxonomy = config.build_taxonomy()
        self.sampler = IntentPolicySampler(
            self.taxonomy, config.intent_weights, config.policy_weights, rng=self.rng)
        self.responder = Responder(
            responder_client, capture_logits=config.capture_logits,
            max_tokens=config.max_tokens, temperature=config.temperature)
        self.holistic = None
        if config.holistic_judge_enabled and holistic_client is not None:
            self.holistic = HolisticJudge(
                holistic_client, rng=self.rng,
                sample_rate=config.holistic_judge_sample_rate,
                gate_on_programmatic=config.holistic_judge_gate_on_programmatic)
        # Rule-based category tagging by default; the LLM classifier is only wired
        # in when explicitly enabled (it is off the critical path otherwise).
        classifier = simulator_client if config.seed_classifier_enabled else None
        self.curator = SeedCurator(rng=self.rng, classifier_client=classifier)
        self.writer = SegmentWriter()
        self.loop = ConversationLoop(
            self.responder, self.sampler, config,
            simulator_client=simulator_client, holistic=self.holistic, rng=self.rng)
        self.stats = FactoryRunStats(loop=self.loop.stats)
        self._stats_lock = threading.Lock()
        #: records collected by the most recent ``generate`` call
        self.records: list[ConversationRecord] = []

    # ------------------------------------------------------------------ public
    def generate(
        self,
        seeds: Iterable[SeedInput],
        band: Optional[str] = None,
    ) -> list[ConversationRecord]:
        """Generate one conversation record per (curated) seed, concurrently.

        ``band`` selects the conversation-length band (``"basic"`` /
        ``"generalization"`` / ``"short"``); defaults to ``config.default_band``.
        Records are returned in **seed order** (independent of thread scheduling).
        """
        curated = self.curator.curate(seeds, lang=self.config.lang)
        self.stats.seeds_used = len(curated)
        length_band = self.config.band(band)

        # Build per-conversation work items up front (deterministic per index).
        results: dict[int, ConversationRecord] = {}
        max_workers = max(1, min(self.config.concurrency, len(curated))) if curated else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_one, idx, seed, length_band): idx
                for idx, seed in enumerate(curated)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                record, local_stats = fut.result()
                with self._stats_lock:
                    self.stats.loop.merge(local_stats)
                    if record is None:
                        self.stats.conversations_discarded += 1
                    else:
                        self.stats.conversations_built += 1
                        results[idx] = record

        out = [results[i] for i in sorted(results)]
        self.stats.records_emitted = len(out)
        self.records = out
        return out

    def _run_one(
        self, index: int, seed: Seed, length_band,
    ) -> tuple[Optional[ConversationRecord], LoopStats]:
        """Run a single conversation with its own reproducible RNG."""
        base = self.config.seed if self.config.seed is not None else 0
        rng = random.Random(base + index)
        pack = self.curator.load_prompt_pack(seed)
        target = rng.randint(length_band.min, length_band.max)
        return self.loop.run(seed, pack, target_length=target, rng=rng)

    # ------------------------------------------------------------- HF / output
    def save_to_hub(
        self,
        dataset_id: str,
        config_name: Optional[str] = None,
        split: str = "train",
        token: Optional[str] = None,
        append: bool = True,
    ):
        """Append the generated records to (or create) a HuggingFace dataset."""
        post = FactoryDatasetPostprocessor(
            self.records, dataset_id=dataset_id, config_name=config_name,
            split=split, token=token)
        return post.push_to_hf_hub(append=append)

    def to_dataset(self):
        """Build a :class:`datasets.Dataset` from the collected records."""
        return FactoryDatasetPostprocessor(self.records).to_dataset()

    def write_jsonl(self, path: str) -> int:
        return self.writer.write_jsonl(self.records, path)
