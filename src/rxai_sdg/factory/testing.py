"""Deterministic test helpers.

``constraint_satisfying_handler`` is a :class:`MockLLMClient` handler that reads
the templated follow-up phrasings produced by
:mod:`rxai_sdg.factory.constraints` and returns answers that *satisfy* the common
machine-checkable constraints. This lets the integration / smoke tests exercise
the full loop deterministically (no network) while producing mostly-passing
verifications, fact recalls and clean reasoning/answer splits.

It is best-effort and approximates what a strong teacher would do: it satisfies
the turn's own constraint and, where compatible, also honours active
standing/cumulative form constraints (wrapping content in a value-preserving
way). When constraints genuinely conflict, the loop's regeneration / resample
machinery handles the residue.
"""

from __future__ import annotations

import json
import re
from typing import Any

_RECALL_RE = re.compile(r"[Ww]hat is my ([^?]+)\?")

# The active-constraints note is now rendered to natural language (fix F), so the
# mock teacher detects standing/cumulative rules from their NL phrasing.
_NL_RULE_PATTERNS: list[tuple[str, str]] = [
    (r"single valid JSON object", "json_valid"),
    (r"valid YAML", "yaml_valid"),
    (r"markdown table", "markdown_table"),
    (r"markdown \(headings", "markdown_format"),
    (r"Never use the word '([^']+)'", "forbidden_token"),
    (r"Never use gendered pronouns", "no_gendered_pronouns"),
    (r"every sentence to at most (\d+) words", "max_words_per_sentence"),
    (r"whole response to at most (\d+) words", "length_tokens"),
    (r"exactly (\d+) bullet", "n_bullets"),
    (r"Start every sentence with the letter '([A-Za-z])'", "first_letter"),
    (r"alphabetical order", "alphabetical_sentence_starts"),
]
_NL_RULE_RES = [(re.compile(p), t) for p, t in _NL_RULE_PATTERNS]

# --- user-simulator side -----------------------------------------------------
# The simulator embeds a machine-readable STEER block in its prompt; this handler
# realises the same turn deterministically (no LLM) for the integration / quality
# / concurrency tests, producing user messages that the responder handler below
# can satisfy. Verbosity is reflected so generated lengths vary across a batch.

_STEER_RE = re.compile(r"=== STEER ===\n(.*?)\n=== END STEER ===", re.DOTALL)

_MEDIUM_FILLER = "Take your time and be precise about it."
_LONG_FILLER = (
    "I have been mulling this over for a while and I want to get it exactly right, "
    "so please do not rush the explanation and feel free to add any nuance that "
    "you think genuinely matters here, even if it makes the reply longer.")


def _parse_steer(prompt: str) -> dict[str, str]:
    m = _STEER_RE.search(prompt)
    if not m:
        return {}
    steer: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            steer[key.strip()] = val.strip()
    return steer


def _lengthen(base: str, length: str) -> str:
    if length == "short":
        return base
    if length == "long":
        return f"{base} {_LONG_FILLER}"
    return f"{base} {_MEDIUM_FILLER}"


def simulator_user_turn_handler(prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
    """Realise the simulator's STEER into a natural, responder-parseable user turn.

    The steer is passed via the ``steer=`` kwarg (it is no longer embedded in the
    prompt, which leaked into real model output); fall back to parsing the prompt
    for any legacy caller.
    """
    steer = kwargs.get("steer")
    if not isinstance(steer, dict):
        steer = _parse_steer(prompt)
    op = steer.get("op", "request_constraint")
    length = steer.get("length", "medium")

    if op == "plant_fact":
        label, value = steer.get("fact_label", "note"), steer.get("fact_value", "")
        base = (f"Before we go on, there is one detail worth recording: my {label} "
                f"is {value}. Anyway, back to the matter at hand.")
    elif op == "recall_fact":
        label = steer.get("fact_label", "note")
        base = f"Quick memory check before we continue. What is my {label}?"
    elif op == "update_fact":
        label, value = steer.get("fact_label", "note"), steer.get("fact_value", "")
        base = (f"One change on my side: please update my {label} to {value}. "
                f"What is my {label}?")
    elif op == "continue_topic":
        topic = steer.get("topic", "this")
        base = f"Staying on {topic}, what else really matters here?"
    elif op == "recall_content":
        topic = steer.get("topic", "this")
        base = (f"Earlier you mentioned something about {topic}; could you expand on "
                f"that point you made?")
    else:  # request_constraint / continue_topic
        # ``say`` is already a direct second-person request ("reformat your previous
        # answer as ...") naming the exact constraint; phrase it as a polite ask.
        say = steer.get("say", "revise your previous answer")
        base = f"Could you {say}?"
    return _lengthen(base, length)


def _last_fact_value(prompt: str, readable: str) -> str | None:
    pat = re.compile(rf"my {re.escape(readable)} (?:is|to) ([^.?!\n]+)", re.IGNORECASE)
    matches = pat.findall(prompt)
    if matches:
        return matches[-1].strip()
    return None


def _sentences_starting_with(letter: str, n: int = 3) -> str:
    return " ".join(f"{letter}lpha point {i} stands clear." for i in range(1, n + 1))


def constraint_satisfying_handler(prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
    """Return a ``<think>...</think> answer`` string satisfying the query's constraint."""
    think = "<think>Working through the request carefully.</think>"
    body, kind = _answer_body(prompt)
    if kind != "form":
        # Honour active standing/cumulative constraints a strong teacher would
        # keep satisfying (form wrapping is value-preserving so recalls survive).
        body = _apply_active_constraint(body, prompt, kind)
    return f"{think} {body}"


def _active_constraints(prompt: str) -> list[tuple[str, str]]:
    """Detect active standing/cumulative rules from their NL rendering (fix F)."""
    out: list[tuple[str, str]] = []
    for rx, ctype in _NL_RULE_RES:
        m = rx.search(prompt)
        if m:
            param = m.group(1) if m.groups() else ""
            out.append((ctype, param))
    return out


def _wrap_form(body: str, ctype: str) -> str | None:
    one_line = " ".join(body.split())
    if ctype == "json_valid":
        return json.dumps({"answer": one_line})
    if ctype == "yaml_valid":
        return f"answer: {json.dumps(one_line)}"
    if ctype == "markdown_table":
        return f"| Field | Value |\n| --- | --- |\n| answer | {one_line} |"
    if ctype == "markdown_format":
        return f"# Result\n\n- {one_line}"
    return None


def _apply_active_constraint(body: str, prompt: str, kind: str) -> str:
    # Active-constraint note lines ("- (standing) ...") only ever appear in the
    # responder's standing-instructions block, so scanning the whole prompt is
    # safe and avoids fragile region slicing.
    for ctype, params_raw in _active_constraints(prompt):
        # Form wrapping preserves any recalled value, but would clobber an own
        # lexical answer (e.g. first-letter sentences), so only wrap content
        # whose own request is plain text or a recall.
        if kind in ("recall", "plain"):
            wrapped = _wrap_form(body, ctype)
            if wrapped is not None:
                return wrapped
        if kind == "plain" and ctype == "first_letter" and params_raw:
            return _sentences_starting_with(params_raw.upper())
        if kind == "plain" and ctype == "alphabetical_sentence_starts":
            return "Apples open it. Bananas build it. Cats close it."
    return body


def _answer_body(prompt: str) -> tuple[str, str]:
    """Return ``(body, kind)``.

    ``kind`` is one of ``"form"`` (the turn's own request is a structural format),
    ``"recall"``, ``"lexical"`` or ``"plain"`` - used to decide which active
    constraints can be layered on without breaking the own constraint.
    """
    user_idx = prompt.rfind("User:")
    query = prompt[user_idx + 5:] if user_idx != -1 else prompt

    # -- fact recall / update --------------------------------------------------
    m = _RECALL_RE.search(query)
    if m:
        readable = m.group(1).strip()
        value = _last_fact_value(prompt, readable)
        if value is not None:
            return f"Your {readable} is {value}.", "recall"
        return f"Your {readable} is unknown.", "recall"

    low = query.lower()

    # -- reformat (own form constraint) ---------------------------------------
    if "json object" in low or "valid json" in low:
        return '{"summary": "reformatted answer", "items": ["a", "b"]}', "form"
    if "valid yaml" in low or "as yaml" in low:
        return "summary: reformatted answer\nitems:\n  - a\n  - b", "form"
    if "markdown table" in low:
        return "| Key | Value |\n| --- | --- |\n| summary | reformatted |", "form"
    if "markdown formatting" in low:
        return "# Summary\n\n- first point\n- second point", "form"

    # -- lexical ---------------------------------------------------------------
    fl = re.search(r"starts with the letter '([A-Za-z])'", query)
    if fl:
        return _sentences_starting_with(fl.group(1).upper()), "lexical"
    if "alphabetical order" in low:
        return "Apples open the case. Bananas build on it. Cats close it out.", "lexical"
    mw = re.search(r"at most (\d+) words", query)
    if mw and "sentence" in low:
        return "Short clear point. Another short point. Final short point.", "lexical"
    fb = re.search(r"without ever using the word '([\w-]+)'", query)
    if fb:
        return "This rewritten response avoids that particular term entirely throughout.", "lexical"
    if "no gendered pronouns" in low:
        return "The engineer finished the project and documented the result thoroughly.", "lexical"

    # -- compress --------------------------------------------------------------
    cb = re.search(r"exactly (\d+) bullet", query)
    if cb:
        n = int(cb.group(1))
        return "\n".join(f"- point {i}" for i in range(1, n + 1)), "form"
    cw = re.search(r"at most (\d+) words", query)
    if cw:
        return "A concise compressed summary of the prior answer in few words.", "plain"

    # -- genre -----------------------------------------------------------------
    if "as a limerick" in low:
        return ("There once was an answer so neat,\n"
                "Whose structure was tidy and fleet,\n"
                "It rhymed on each line,\n"
                "In a pattern divine,\n"
                "And finished the verse nice and sweet."), "form"

    # -- default (non-programmatic turns: chained_compute, open_chat, ...) -----
    return ("This is a clear point. Here is a concrete example. That covers it.", "plain")
