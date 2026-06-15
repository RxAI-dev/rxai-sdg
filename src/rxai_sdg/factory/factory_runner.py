"""High-level orchestration: :class:`DataFactory` (spec §2, §11.10).

Ties the six components together behind a single entry point:

    factory = DataFactory(config, responder_client, simulator_client)
    records = factory.generate(dataset_spec, n_conversations=100)

``generate`` curates seeds, runs the conversation loop per seed, derives the
configured reasoning/instruct/mixed training variants, and returns the flat list
of emitted :class:`ConversationRecord` objects (also writable to JSONL).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .clients import LLMClient
from .config import FactoryConfig
from .holistic import HolisticJudge
from .loop import ConversationLoop, LoopStats
from .responder import Responder
from .sampler import IntentPolicySampler
from .schemas import ConversationRecord, Seed
from .seed_curator import DatasetSpec, SeedCurator
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
            max_tokens=config.responder.max_tokens,
            temperature=config.responder.temperature)
        self.simulator_client = simulator_client
        self.holistic = None
        if config.holistic_judge_enabled and holistic_client is not None:
            self.holistic = HolisticJudge(
                holistic_client, rng=self.rng,
                sample_rate=config.holistic_judge_sample_rate,
                gate_on_programmatic=config.holistic_judge_gate_on_programmatic)
        self.curator = SeedCurator(rng=self.rng)
        self.writer = SegmentWriter(
            rng=self.rng, mixed_mode_keep_ratio=config.mixed_mode_keep_ratio)
        self.loop = ConversationLoop(
            self.responder, self.sampler, config,
            simulator_client=simulator_client, holistic=self.holistic, rng=self.rng)
        self.stats = FactoryRunStats(loop=self.loop.stats)

    # ------------------------------------------------------------------ public
    def generate_from_seeds(
        self,
        seeds: list[Seed],
        n_conversations: Optional[int] = None,
        band: Optional[str] = None,
    ) -> list[ConversationRecord]:
        out: list[ConversationRecord] = []
        n = n_conversations or len(seeds)
        for i in range(n):
            seed = seeds[i % len(seeds)] if seeds else None
            if seed is None:
                break
            self.stats.seeds_used += 1
            pack = self.curator.load_prompt_pack(seed)
            length_band = self.config.band(band)
            target = self.rng.randint(length_band.min, length_band.max)
            record = self.loop.run(seed, pack, target_length=target)
            if record is None:
                self.stats.conversations_discarded += 1
                continue
            self.stats.conversations_built += 1
            variants = self.writer.derive_variants(record, self.config.derived_variants)
            out.extend(variants)
        self.stats.records_emitted = len(out)
        return out

    def generate(
        self,
        dataset_spec: DatasetSpec,
        n_conversations: Optional[int] = None,
        band: Optional[str] = None,
        balance_domains: bool = True,
        extra_seeds: Optional[list[Seed]] = None,
    ) -> list[ConversationRecord]:
        seeds = self.curator.load_seeds(dataset_spec)
        if extra_seeds:
            seeds = self.curator.inject_seeds(seeds, extra_seeds)
        if balance_domains:
            seeds = self.curator.balance_domains(seeds, self.config.domain_mix)
        return self.generate_from_seeds(seeds, n_conversations=n_conversations, band=band)

    def write_jsonl(self, records: list[ConversationRecord], path: str) -> int:
        return self.writer.write_jsonl(records, path)
