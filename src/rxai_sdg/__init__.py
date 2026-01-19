"""
rxai-sdg: Synthetic Dataset Generators for Reactive Transformer Models

Modules:
- rxai_sdg.base: Base classes for dataset generators
- rxai_sdg.mrl: Memory Reinforcement Learning dataset generators
- rxai_sdg.sft: Supervised Fine-Tuning dataset generators
- rxai_sdg.hybrid: Hybrid Reasoning and DMPO dataset generators for RxT-Beta
"""

__version__ = "0.1.33"

from .base import BaseDatasetGenerator, default_additional_config

__all__ = [
    "BaseDatasetGenerator",
    "default_additional_config",
]
