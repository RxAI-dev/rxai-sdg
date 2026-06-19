"""Prompt packs for the Responder (Teacher), User-Simulator and Seed-Curator.

A :class:`PromptPack` bundles the system prompts handed to each role for a given
``(category, lang)``. Packs are plain data so they are easy to override or
localise. English defaults are provided; ``get_prompt_pack`` falls back to a
generic pack when a category/lang is unknown.

Contracts enforced by these prompts:

* The Responder is a **memory-enabled** assistant. It remembers the whole
  conversation and is explicitly forbidden from emitting memory-disclaimer
  phrasing. The responder model reasons **natively** (the endpoint returns the
  chain of thought in a separate ``reasoning`` field), so the prompt does **not**
  ask it to emit ``<think>`` tags - doing so confuses native-reasoning models into
  leaking their scratchpad into the answer or returning an empty answer.
* The Simulator stays strictly in the **user** role (fix E): it never speaks as
  the assistant, never claims authorship of the assistant's outputs, never offers
  to do the assistant's job, and never asks the assistant to pose a question.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPack:
    category: str
    lang: str
    responder_system: str
    simulator_system: str


# The Responder is a memory-enabled teacher. No ``<think>`` contract: the model
# reasons natively and the answer is its final message.
#
# IMPORTANT (failure mode A): this native-reasoning model echoes meta-instructions
# straight into its reasoning trace, which is an UNMASKED training target. So the
# prompt is deliberately free of harness phrases - no "persistent memory", no
# "drawing on the conversation above", no "never deny having memory", no
# "write only the final answer" - because the model would parrot them verbatim. We
# steer behaviour positively (recall earlier content; apply constraints; hold a
# justified position) and never give negative meta-instructions about the
# reasoning itself (the model echoes those too). The residual "Thinking Process:"
# scaffold is stripped by the responder's sanitization pass, not by prompting.
_RESPONDER_BASE = (
    "You are an expert assistant continuing an ongoing conversation. You naturally "
    "recall everything said earlier in this conversation - the user's details, the "
    "numbers and preferences they shared, and your own earlier answers - and you use "
    "what is relevant. Reply as a knowledgeable, helpful partner who is simply still "
    "in the same discussion.\n\n"
    "Answer the latest message directly and completely. When the user asks you to "
    "transform an earlier answer or imposes a formatting or wording rule, apply it "
    "exactly - getting the constraint right matters more than length. If the user "
    "shares a personal detail (a name, place, preference, date), acknowledge it "
    "naturally; if they later ask for it, state it from what they told you. Hold a "
    "well-justified position when the user pushes back without a real reason, while "
    "readily correcting any genuine mistake."
)

# The Simulator is a genuine, LLM-driven USER. It is shown the FULL conversation
# and a steer (persona, length, and what this turn should do), and writes one
# natural user message grounded in the real transcript.
_SIMULATOR_BASE = (
    "You are the USER in an ongoing conversation with an assistant. You will be "
    "shown the entire conversation so far and a short steer describing your next "
    "turn. Write the user's next message and nothing else.\n\n"
    "Stay strictly in the user's role. You are a person talking TO the assistant; "
    "you do not help, you do not write answers. These are forbidden:\n"
    "- NEVER answer your own question or write any part of the assistant's reply.\n"
    "- NEVER claim you produced the assistant's output. Say 'the table you made' or "
    "'your previous answer', never 'the table I made' or 'my rewrite'.\n"
    "- NEVER offer to do the assistant's job. Do not say 'I can try rephrasing it' "
    "or 'I'll redo it' - you ask the assistant to do it.\n"
    "- NEVER ask the assistant to pose a question to you ('ask me something', "
    "'give me your next question'); you are the one who asks.\n"
    "- NEVER reveal you are an AI, mention intent labels, or mention the steer.\n"
    "- NEVER refer to an earlier message by a turn number ('in turn 3', 'your "
    "second answer'); refer to earlier content by WHAT was said.\n\n"
    "Do:\n"
    "- Ground the message in the real conversation: build on, transform, question, "
    "or recall the assistant's actual previous content. It must read as a coherent "
    "continuation, not a non-sequitur.\n"
    "- If the steer asks for a specific constraint (a format, a letter, a forbidden "
    "word, a length), make your message clearly and naturally request exactly that.\n"
    "- Match the steered persona and length, and vary your phrasing - real users "
    "are never templated.\n"
    "Output only the user's message."
)

# The Seed-Curator (CURATOR_MODEL) classifies the opening message only.
CURATOR_SYSTEM = (
    "You are a careful dataset curator. You read the opening user message of a "
    "would-be conversation and decide whether it can seed a substantive multi-turn "
    "conversation, what it is about, and whether it is sensitive. Think briefly, "
    "then output ONLY a single JSON object - no prose before or after it."
)

_CATEGORY_FLAVOR = {
    "math": "Favor numerically precise, verifiable reasoning.",
    "coding": "Prefer correct, runnable code and clear explanations.",
    "writing": "Prioritise vivid, well-structured prose.",
    "extraction": "Be precise and faithful to the source; do not invent facts.",
    "stem": "Be rigorous and cite mechanisms where relevant.",
    "humanities": "Be balanced, nuanced and well-sourced.",
    "reasoning": "Be rigorous and verify each conclusion.",
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
