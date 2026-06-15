"""Prompt packs for the Responder (Teacher) and User-Simulator.

A :class:`PromptPack` bundles the system prompts handed to each role for a given
``(category, lang)``. Packs are plain data so they are easy to override or
localise. English defaults are provided; ``get_prompt_pack`` falls back to a
generic pack when a category/lang is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPack:
    category: str
    lang: str
    responder_system: str
    simulator_system: str


_RESPONDER_BASE = (
    "You are an expert teaching assistant generating high-quality reference "
    "answers for a training dataset. Always think step by step inside a single "
    "<think>...</think> block, then give the final answer after it. The final "
    "answer must stand on its own and must NOT reference the reasoning (avoid "
    "phrases like 'as computed above' or 'from step 2'). When the user imposes a "
    "formatting or lexical constraint, satisfy it EXACTLY -- correctness of the "
    "constraint matters more than verbosity."
)

_SIMULATOR_BASE = (
    "You are simulating a curious, demanding human user in a multi-turn "
    "conversation. Produce a single natural follow-up message that matches the "
    "requested intent. Stay in character, be concise, and never reveal that you "
    "are an AI or mention the underlying intent labels."
)

_CATEGORY_FLAVOR = {
    "math": "Favor numerically precise, verifiable reasoning.",
    "coding": "Prefer correct, runnable code and clear explanations.",
    "writing": "Prioritise vivid, well-structured prose.",
    "extraction": "Be precise and faithful to the source; do not invent facts.",
    "stem": "Be rigorous and cite mechanisms where relevant.",
    "humanities": "Be balanced, nuanced and well-sourced.",
    "reasoning": "Make each inferential step explicit and checkable.",
    "roleplay": "Stay immersive and consistent with the persona.",
    "general": "",
}


def get_prompt_pack(category: str, lang: str = "en") -> PromptPack:
    flavor = _CATEGORY_FLAVOR.get(category, "")
    responder = _RESPONDER_BASE + ((" " + flavor) if flavor else "")
    if lang != "en":
        responder += f" Respond natively in language code '{lang}'."
    simulator = _SIMULATOR_BASE
    if lang != "en":
        simulator += f" Write the follow-up natively in language code '{lang}'."
    return PromptPack(
        category=category, lang=lang,
        responder_system=responder, simulator_system=simulator)
