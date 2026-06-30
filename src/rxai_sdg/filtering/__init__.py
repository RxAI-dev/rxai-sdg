"""rxai_sdg.filtering - conversational dataset quality scoring and filtering.

This package scores existing conversational datasets with an LLM-as-a-judge and
writes the resulting quality scores back so the data can be filtered downstream.

Two entry points:

- :func:`score_conversational_dataset` - notebook-friendly, threaded scoring of a
  HuggingFace ``Dataset`` with optional periodic upload back to the Hub.
- The CLI modules (``evaluator``, ``sampling``, ``analysis``,
  ``subset_quality_report``, ``run_judge_models``, ``list_models``) for the
  config-driven, file-based workflow. Run them with ``python -m`` e.g.
  ``python -m rxai_sdg.filtering.evaluator --help``.

The judge expects the standard interaction format: each example is either a list
of ``{"query", "think", "answer"}`` interactions, or a dict with an
``interactions`` list and an optional ``system`` prompt.
"""

from __future__ import annotations

from .judge import JudgeOutputError, extract_conversation
from .scoring import (
    ScoringResult,
    ScoringSettings,
    UploadSettings,
    score_conversational_dataset,
)

__all__ = [
    "score_conversational_dataset",
    "ScoringResult",
    "ScoringSettings",
    "UploadSettings",
    "extract_conversation",
    "JudgeOutputError",
]
