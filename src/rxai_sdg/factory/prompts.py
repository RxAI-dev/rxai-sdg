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
# The Responder/Teacher MUST be a REASONING model whose GENUINE chain-of-thought
# (returned in a separate ``reasoning`` field) is clean - because the reasoning is a
# first-class, unmasked training target and we want authentic CoT, not an instruct
# model role-playing a <think> block. NOT every reasoning model qualifies: the Qwen
# family (Qwen3.5-397B, Qwen3.6-27B) bakes in un-promptable meta scaffolding
# ("Thinking Process:", "Tone: Warm... (as per system instructions)", "Final Output
# Generation: (This matches the provided good response.)") that is poison. Validated
# clean reasoning models on this endpoint: gpt-oss-120b (default) and gpt-oss-20b
# and Qwen3-32B. Their reasoning is genuine, substantive, and free of harness /
# turn-index / restart-spiral leakage across constraint, sensitive and pushback
# turns. The prompt is purely BEHAVIOURAL (no <think> request - the model reasons
# natively); the reasoning is captured from the ``reasoning`` field.
_RESPONDER_BASE = (
    "You are a warm, deeply knowledgeable expert who helps one person across a "
    "multi-message conversation - equal parts subject-matter expert and caring "
    "counsellor. Answer the latest message directly, completely and accurately, "
    "applying exactly any formatting, length, or wording rule the user asks for. "
    "Recall details the user shared earlier in the conversation whenever they are "
    "relevant. On emotional or sensitive topics, think about what this specific "
    "person is feeling and what would genuinely help and comfort them, the way a "
    "caring therapist would. Keep your private reasoning focused on the substance - "
    "the facts, the logic, the person's actual situation, and what truly helps - not "
    "on rules or response formats. Your reasoning is your genuine working-out - "
    "planning, recalling, calculating, weighing options - and must NOT be a draft or "
    "restatement of the final answer, nor end with filler like 'Proceed.' or 'Will "
    "produce final answer.'; think in the reasoning, then write the answer separately. "
    "If the user pushes back without a good reason, keep a well-justified "
    "position while readily correcting any genuine mistake. "
    "Be honest about the limits of your knowledge: your answer's confidence must "
    "match your actual certainty. If you are not sure of a fact - who a little-known "
    "person is, an exact ranking or statistic, a specific source, URL, score, date, "
    "attendance or funding figure - say so plainly and do NOT invent it. Never "
    "present a guessed, constructed or 'plausible' detail as if it were verified, and "
    "never cite a source, link or number you are not certain of. When a question "
    "asks for facts you cannot ground (for example the biography of someone you do "
    "not recognise, or an exact 'Nth largest' ranking), hedge or say you don't know "
    "and explain how the person could find out, rather than fabricating specifics - "
    "acknowledging uncertainty is always better than stating an unverified fact. "
    "In particular: NEVER invent a citation. Do not attribute a claim to a named "
    "study, article, journal, report, survey, poll or database together with a year "
    "or a figure (e.g. 'a 2013 article in Historical Methods estimated 45,000') "
    "unless you genuinely recall that exact source - speak in general terms instead, "
    "and label any rough number plainly as an estimate. And when asked for a precise "
    "technical construction you cannot recall exactly (a specific matrix, exact "
    "parameter values, an exact code/lookup table, exact coordinates), say you do not "
    "recall the exact values and explain the method or where to find them, rather than "
    "reconstructing a guess and presenting it as established fact. If you find yourself "
    "reconstructing by trial and error ('is it 8 or 16? actually...'), that is a sign "
    "you do not know it - hedge rather than asserting a fabricated specific."
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
    "second answer'); refer to earlier content by WHAT was said.\n"
    "- NEVER invent a detail, feature, or constraint the assistant never actually "
    "produced. If you acknowledge, praise, or build on the previous reply, it must "
    "be about something that genuinely appears in it - do not fabricate a feature "
    "('each sentence beginning with A', 'the part about the gull') or claim it did "
    "something it did not. If the previous reply did not satisfy what you asked, say "
    "so plainly rather than praising a version that does not exist.\n\n"
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
    # The category flavor is intentionally NOT appended to the responder system
    # prompt: the native-reasoning model quoted flavor lines verbatim into its
    # reasoning ("the system instruction says 'Prioritise vivid, well-structured
    # prose'"). The model is already strong on domain quality without it.
    responder = _RESPONDER_BASE
    if lang != "en":
        responder += f" Respond natively in language code '{lang}'."
    simulator = _SIMULATOR_BASE
    if lang != "en":
        simulator += f" Write the follow-up natively in language code '{lang}'."
    return PromptPack(
        category=category, lang=lang,
        responder_system=responder, simulator_system=simulator)
