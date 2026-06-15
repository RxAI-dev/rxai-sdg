"""
rxai-sdg: Synthetic Dataset Generators for Reactive Transformer Models

Modules:
- rxai_sdg.base: Base classes for dataset generators
- rxai_sdg.mrl: Memory Reinforcement Learning dataset generators
- rxai_sdg.sft: Supervised Fine-Tuning dataset generators
- rxai_sdg.hybrid: Hybrid Reasoning and DMPO dataset generators for RxT-Beta
"""

__version__ = "0.1.33"

# The base module pulls in optional heavy dependencies (openai / ollama). Keep
# the top-level import resilient so dependency-light subpackages such as
# ``rxai_sdg.factory`` remain importable (and unit-testable) even when those
# providers are not installed.
try:
    from .base import BaseDatasetGenerator, default_additional_config
except ImportError:  # pragma: no cover - exercised only without openai/ollama
    BaseDatasetGenerator = None  # type: ignore[assignment]
    default_additional_config = None  # type: ignore[assignment]

__all__ = [
    "BaseDatasetGenerator",
    "default_additional_config",
]
