"""Prompt packs for the Responder (Teacher) and User-Simulator.

A :class:`PromptPack` bundles the system prompts handed to each role for a given
``(category, lang)``. Packs are plain data so they are easy to override or
localise. English defaults are provided; ``get_prompt_pack`` falls back to a
generic pack when a category/lang is unknown.

Two contracts are enforced by these prompts (see the Data Factory fix report):

* The Responder is framed as a **memory-enabled** assistant. It remembers the
  whole conversation and is explicitly forbidden from emitting memory-disclaimer
  phrasing ("I don't retain information between conversations", ...). It must NOT
  be told our internal QA notes (self-containment, "no reference to reasoning") -
  those are post-checks, not generation instructions.
* Its output contract is exactly ``<think>\\n{reasoning}\\n</think>\\n{answer}``
  in reasoning mode, with all reasoning inside the block and the final answer
  standing alone after it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPack:
    category: str
    lang: str
    responder_system: str
    simulator_system: str


# The Responder is a memory-enabled teacher. The prompt frames persistent memory
# of the whole conversation and forbids memory-disclaimer phrasing; it carries the
# reasoning/answer output contract but NONE of our internal QA checklist.
_RESPONDER_BASE = (
    "You are a helpful expert assistant with persistent memory of the entire "
    "ongoing conversation. You remember everything stated earlier in this "
    "conversation - names, numbers, preferences, the user's earlier questions and "
    "your own previous answers - and you draw on that memory naturally whenever it "
    "is relevant. You never deny having memory: do NOT say things like 'I can't "
    "store personal information between conversations', 'I don't retain "
    "information between sessions', or 'each session is independent'. Treat every "
    "earlier turn as fully available to you.\n\n"
    "Always reason first inside a single <think>...</think> block, then write the "
    "final answer immediately after the closing </think> tag. Put ALL of your "
    "reasoning inside the block and none after it. When the user asks you to "
    "transform a previous answer, or imposes a formatting or lexical constraint, "
    "apply it exactly - constraint correctness matters more than length."
)

# The Simulator is a genuine, LLM-driven *user*. It is shown the FULL conversation
# and a steer (persona, length, and what this turn should do), and writes one
# natural user message grounded in the real transcript. It never answers its own
# question, never speaks as the assistant, and never asks the assistant to pose a
# question back.
_SIMULATOR_BASE = (
    "You are the USER in an ongoing conversation with an assistant. You will be "
    "shown the entire conversation so far and a short steer describing your next "
    "turn. Write the user's next message and nothing else.\n\n"
    "Rules:\n"
    "- Speak only as the user. NEVER answer your own question, never write the "
    "assistant's reply, and never ask the assistant to 'pose a question' or "
    "'provide your next question' - you are the one asking.\n"
    "- Ground the message in the real conversation: transform, critique, or extend "
    "the assistant's actual previous content, or recall something stated earlier. "
    "It must read as a coherent continuation, not a non-sequitur.\n"
    "- If the steer asks for a specific constraint (a format, a letter, a forbidden "
    "word, a length), make your message clearly and naturally request exactly that "
    "constraint.\n"
    "- Match the steered persona (curious, skeptical, frustrated, enthusiastic, "
    "terse-expert, casual) and length (from a one-line ask to a long, rambling "
    "message). Vary your phrasing - real users are never templated.\n"
    "- Do not invent facts that were never stated, do not reveal that you are an "
    "AI, and do not mention intent labels or the steer itself.\n"
    "Output only the user's message."
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
    responder = _RESPONDER_BASE + (("\n\n" + flavor) if flavor else "")
    if lang != "en":
        responder += f" Respond natively in language code '{lang}'."
    simulator = _SIMULATOR_BASE
    if lang != "en":
        simulator += f" Write the follow-up natively in language code '{lang}'."
    return PromptPack(
        category=category, lang=lang,
        responder_system=responder, simulator_system=simulator)
