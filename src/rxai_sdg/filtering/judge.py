"""Shared LLM-as-a-judge primitives for the dataset quality filtering pipeline.

This module is provider- and orchestration-agnostic: it knows how to turn a
conversational dataset example into the judge prompt payload, how to call the
judge schema validator, and how to robustly parse a judgment object out of a
model response. Both the CLI evaluator (async) and the notebook-facing
``score_conversational_dataset`` orchestrator (threaded) build on these helpers
so they share a single source of truth for the conversation format and prompts.

It depends only on the standard library plus ``jsonschema`` and ``openai`` types
(no YAML / dotenv), so importing it never pulls in the CLI-only dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = PACKAGE_ROOT / "prompts"
DEFAULT_SYSTEM_PROMPT_PATH = DEFAULT_PROMPTS_DIR / "judge_prompt.txt"
DEFAULT_USER_TEMPLATE_PATH = DEFAULT_PROMPTS_DIR / "judge_user_template.txt"
DEFAULT_OUTPUT_SCHEMA_PATH = DEFAULT_PROMPTS_DIR / "judge_output_schema.json"

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# Conversation/payload keys we forward to the judge. Everything else on a
# dataset example (ids, source metadata, prior scores, ...) is dropped.
CONVERSATION_KEYS = ("system", "interactions")


class JudgeOutputError(ValueError):
    """Raised when the judge response cannot be parsed into a valid judgment."""


def load_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def build_validator(schema_path: Path | str = DEFAULT_OUTPUT_SCHEMA_PATH) -> Draft202012Validator:
    return Draft202012Validator(load_json(schema_path))


def extract_conversation(example: Any) -> dict[str, Any]:
    """Reduce a dataset example to the generic conversation payload.

    Supported shapes:
    - ``[{"query", "think", "answer"}, ...]`` -- a bare interactions list.
    - ``{"interactions": [...], "system": "..."}`` -- the standard row format,
      where ``system`` is optional and any other metadata fields are ignored.

    Returns a dict containing ``interactions`` and, when present and non-empty,
    ``system`` -- nothing else. ``system`` is ordered first for readability in
    the prompt.
    """
    if isinstance(example, list):
        return {"interactions": example}

    if isinstance(example, dict):
        interactions = example.get("interactions")
        if interactions is None:
            raise ValueError(
                "Conversation example dict is missing the required 'interactions' field. "
                f"Available keys: {sorted(example.keys())}"
            )
        payload: dict[str, Any] = {}
        system = example.get("system")
        if isinstance(system, str) and system.strip():
            payload["system"] = system
        payload["interactions"] = interactions
        return payload

    raise ValueError(
        "Unsupported conversation format: expected a list of interactions or a dict "
        f"with an 'interactions' field, got {type(example).__name__}."
    )


def build_messages(system_prompt: str, user_template: str, conversation: dict[str, Any]) -> list[dict[str, str]]:
    conversation_json = json.dumps(conversation, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.replace("{conversation_json}", conversation_json)},
    ]


def conversation_stats(conversation: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    interactions = conversation.get("interactions")
    n_interactions = len(interactions) if isinstance(interactions, list) else None
    return {
        "n_interactions": n_interactions,
        "has_system": "system" in conversation,
        "prompt_chars": sum(len(message["content"]) for message in messages),
    }


def parse_and_validate_judgment(raw_content: str, validator: Draft202012Validator) -> dict[str, Any]:
    for candidate in json_object_candidates(raw_content):
        try:
            judgment = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        errors = sorted(validator.iter_errors(judgment), key=lambda error: list(error.path))
        if not errors:
            return judgment

    preview = raw_content.strip()[:500]
    raise JudgeOutputError(f"Judge returned no schema-valid JSON object: {preview}")


def json_object_candidates(raw_content: str) -> list[str]:
    stripped = raw_content.strip()
    candidates: list[str] = []
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    required_key_pos = stripped.find('"memory_score"')
    if required_key_pos != -1:
        start = stripped.rfind("{", 0, required_key_pos)
        end = balanced_json_end(stripped, start)
        if start != -1 and end != -1:
            candidates.append(stripped[start : end + 1])

    for start, char in enumerate(stripped):
        if char != "{":
            continue
        end = balanced_json_end(stripped, start)
        if end != -1:
            candidates.append(stripped[start : end + 1])

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped or [stripped]


def balanced_json_end(text: str, start: int) -> int:
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def response_debug_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json", exclude_none=True)
    else:
        dumped = {"repr": repr(response)}

    choices = dumped.get("choices") if isinstance(dumped, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    return {
        "finish_reason": first_choice.get("finish_reason"),
        "message": first_choice.get("message"),
        "usage": dumped.get("usage") if isinstance(dumped, dict) else None,
    }


def response_metadata(response: Any) -> dict[str, Any]:
    dumped = response.model_dump(mode="json", exclude_none=True) if hasattr(response, "model_dump") else {}
    choices = dumped.get("choices") if isinstance(dumped, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    usage = dumped.get("usage", {}) if isinstance(dumped, dict) else {}
    details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
    return {
        "finish_reason": first_choice.get("finish_reason"),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
        },
    }


def extract_message_content(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()

    dumped = message.model_dump(mode="json", exclude_none=True) if hasattr(message, "model_dump") else {}
    for key in ("content", "reasoning_content", "reasoning"):
        value = dumped.get(key) if isinstance(dumped, dict) else None
        if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
            return value.strip()
    return ""


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, JudgeOutputError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False
