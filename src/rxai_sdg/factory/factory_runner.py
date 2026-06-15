"""High-level orchestration: :class:`DataFactory` (spec §2, §11.10).

Workflow (e.g. one of many parallel notebook sessions)::

    factory = DataFactory(config, responder_client, simulator_client)

    seeds = ["Explain entropy.", {"query": "Reverse a linked list."}]  # str or dict
    records = factory.generate(seeds)        # one conversation per seed

    factory.save_to_hub("org/rxt-factory", token="hf_...")   # append or create

``generate`` curates the seeds (inferring category/domain), runs the conversation
loop once per seed, and returns the list of reasoning-mode
:class:`ConversationRecord` objects (also stored on ``factory.records``). Deriving
instruct / mixed training variants is a separate post-processing step
(:mod:`rxai_sdg.factory.variants`).
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
from .schemas import ConversationRecord
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
        # The simulator model doubles as the category classifier fallback.
        self.curator = SeedCurator(rng=self.rng, classifier_client=simulator_client)
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
        seeds: Iterable[SeedInput],
        band: Optional[str] = None,
    ) -> list[ConversationRecord]:
        """Generate one conversation record per (curated) seed.

        ``band`` selects the conversation-length band (``"basic"`` /
        ``"generalization"`` / ``"short"``); defaults to ``config.default_band``.
        """
        curated = self.curator.curate(seeds, lang=self.config.lang)
        self.stats.seeds_used = len(curated)
        out: list[ConversationRecord] = []
        length_band = self.config.band(band)
        for seed in curated:
            pack = self.curator.load_prompt_pack(seed)
            target = self.rng.randint(length_band.min, length_band.max)
            record = self.loop.run(seed, pack, target_length=target)
            if record is None:
                self.stats.conversations_discarded += 1
                continue
            self.stats.conversations_built += 1
            out.append(record)
        self.stats.records_emitted = len(out)
        self.records = out
        return out

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
