"""High-level orchestration: :class:`DataFactory` (spec §2, §11.10).

Typical workflow (e.g. one of many parallel notebook sessions)::

    factory = DataFactory(config, responder_client, simulator_client)

    seeds = ["Explain entropy.", {"query": "Reverse a linked list."}]  # str or dict
    records = factory.generate(seeds)                 # one record per conversation

    factory.save_to_hub("org/rxt-factory", token="hf_...", append=True)

``generate`` curates seeds (inferring category/domain - with an optional LLM
fallback classifier), runs the conversation loop per seed, and returns the flat
list of reasoning-mode :class:`ConversationRecord` objects. Deriving instruct /
mixed training variants is a separate post-processing step
(:mod:`rxai_sdg.factory.variants`), intentionally not run here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from .clients import LLMClient
from .config import FactoryConfig
from .dataset import FactoryDatasetPostprocessor
from .holistic import HolisticJudge
from .loop import ConversationLoop, LoopStats
from .responder import Responder
from .sampler import IntentPolicySampler
from .schemas import ConversationRecord, Seed
from .seed_curator import DatasetSpec, SeedCurator
from .writer import SegmentWriter

#: accepted seed inputs: a DatasetSpec, or a list of prompt strings / dicts
SeedInput = Union[DatasetSpec, Iterable[Union[str, dict]]]


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
        curator_client: Optional[LLMClient] = None,
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
        # Curator uses an LLM classifier fallback when one is available; default
        # to the (cheaper) simulator client so a bare prompt list still gets a
        # sensible category when the keyword heuristic is inconclusive.
        self.curator = SeedCurator(
            rng=self.rng,
            classifier_client=curator_client if curator_client is not None else simulator_client)
        self.writer = SegmentWriter()
        self.loop = ConversationLoop(
            self.responder, self.sampler, config,
            simulator_client=simulator_client, holistic=self.holistic, rng=self.rng)
        self.stats = FactoryRunStats(loop=self.loop.stats)
        #: records collected by the most recent ``generate`` call
        self.records: list[ConversationRecord] = []

    # ------------------------------------------------------------------ public
    def generate(
        self,
        seeds: SeedInput,
        n_conversations: Optional[int] = None,
        band: Optional[str] = None,
        balance_domains: bool = False,
        extra_seeds: Optional[list[Seed]] = None,
    ) -> list[ConversationRecord]:
        """Generate one conversation record per seed (or ``n_conversations``)."""
        curated = self.curator.load_seeds(self._to_dataset_spec(seeds))
        if extra_seeds:
            curated = self.curator.inject_seeds(curated, extra_seeds)
        if balance_domains:
            curated = self.curator.balance_domains(curated, self.config.domain_mix)
        return self.generate_from_seeds(curated, n_conversations=n_conversations, band=band)

    def generate_from_seeds(
        self,
        seeds: list[Seed],
        n_conversations: Optional[int] = None,
        band: Optional[str] = None,
    ) -> list[ConversationRecord]:
        out: list[ConversationRecord] = []
        n = n_conversations if n_conversations is not None else len(seeds)
        for i in range(n):
            if not seeds:
                break
            seed = seeds[i % len(seeds)]
            self.stats.seeds_used += 1
            pack = self.curator.load_prompt_pack(seed)
            length_band = self.config.band(band)
            target = self.rng.randint(length_band.min, length_band.max)
            record = self.loop.run(seed, pack, target_length=target)
            if record is None:
                self.stats.conversations_discarded += 1
                continue
            self.stats.conversations_built += 1
            out.append(record)  # one reasoning-mode record per conversation
        self.stats.records_emitted = len(out)
        self.records = out
        return out

    # ------------------------------------------------------------- HF / output
    def to_postprocessor(
        self,
        records: Optional[list[ConversationRecord]] = None,
        dataset_id: Optional[str] = None,
        config_name: Optional[str] = None,
        split: str = "train",
        token: Optional[str] = None,
    ) -> FactoryDatasetPostprocessor:
        return FactoryDatasetPostprocessor(
            records if records is not None else self.records,
            dataset_id=dataset_id or self.config.hf_dataset_id,
            config_name=config_name if config_name is not None else self.config.hf_config_name,
            split=split if split != "train" else self.config.hf_split,
            token=token,
        )

    def save_to_hub(
        self,
        dataset_id: Optional[str] = None,
        config_name: Optional[str] = None,
        split: str = "train",
        token: Optional[str] = None,
        append: bool = True,
        records: Optional[list[ConversationRecord]] = None,
    ):
        """Append the generated records to (or create) a HuggingFace dataset."""
        post = self.to_postprocessor(
            records=records, dataset_id=dataset_id, config_name=config_name,
            split=split, token=token)
        return post.push_to_hf_hub(append=append)

    def to_dataset(self, records: Optional[list[ConversationRecord]] = None):
        return self.to_postprocessor(records=records).to_dataset()

    def write_jsonl(self, records: list[ConversationRecord], path: str) -> int:
        return self.writer.write_jsonl(records, path)

    # ----------------------------------------------------------------- helpers
    def _to_dataset_spec(self, seeds: SeedInput) -> DatasetSpec:
        if isinstance(seeds, DatasetSpec):
            return seeds
        records: list[dict] = []
        for s in seeds:
            if isinstance(s, str):
                records.append({"query": s})
            elif isinstance(s, dict):
                records.append(s)
            else:
                raise TypeError(
                    f"seed must be a str or dict with a 'query' field, got {type(s)}")
        return DatasetSpec(
            name="in_memory", records=records, lang=self.config.lang,
            haystack_fraction=self.config.haystack_fraction)
