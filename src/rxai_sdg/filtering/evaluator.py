from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI
from tqdm import tqdm

from .judge import (
    JudgeOutputError,
    build_messages,
    build_validator,
    conversation_stats,
    extract_conversation,
    extract_message_content,
    is_retryable_error,
    load_text,
    parse_and_validate_judgment,
    response_debug_payload,
    response_metadata,
)
from .utils import append_jsonl, read_jsonl, safe_filename, stable_example_id


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class EvalConfig:
    base_url: str
    api_key_env: str
    input_path: Path
    output_dir: Path
    system_prompt_path: Path
    user_template_path: Path
    output_schema_path: Path
    model: str
    concurrency: int
    request_timeout_seconds: float
    max_retries: int
    max_output_tokens: int
    temperature: float
    json_mode: bool
    extra_body: dict[str, Any] | None


def load_eval_config(
    path: Path,
    model_override: str | None,
    temperature_override: float | None,
    concurrency_override: int | None,
) -> EvalConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict) or "evaluation" not in raw:
        raise ValueError(f"Missing evaluation config in {path}")

    cfg = raw["evaluation"]
    model = model_override or cfg["default_model"]
    temperature = temperature_override if temperature_override is not None else float(cfg["temperature"])
    concurrency = concurrency_override if concurrency_override is not None else int(cfg["concurrency"])
    input_path = Path(cfg["input_path"])
    output_dir = Path(cfg["output_dir"])
    system_prompt_path = Path(cfg["system_prompt_path"])
    user_template_path = Path(cfg["user_template_path"])
    output_schema_path = Path(cfg["output_schema_path"])

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not system_prompt_path.is_absolute():
        system_prompt_path = PROJECT_ROOT / system_prompt_path
    if not user_template_path.is_absolute():
        user_template_path = PROJECT_ROOT / user_template_path
    if not output_schema_path.is_absolute():
        output_schema_path = PROJECT_ROOT / output_schema_path

    return EvalConfig(
        base_url=cfg["base_url"],
        api_key_env=cfg["api_key_env"],
        input_path=input_path,
        output_dir=output_dir,
        system_prompt_path=system_prompt_path,
        user_template_path=user_template_path,
        output_schema_path=output_schema_path,
        model=model,
        concurrency=concurrency,
        request_timeout_seconds=float(cfg["request_timeout_seconds"]),
        max_retries=int(cfg["max_retries"]),
        max_output_tokens=int(cfg["max_output_tokens"]),
        temperature=temperature,
        json_mode=bool(cfg["json_mode"]),
        extra_body=cfg.get("extra_body"),
    )


def get_api_key(env_name: str) -> str:
    load_dotenv()
    api_key = os.getenv(env_name) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {env_name} in your environment or .env. "
            "Do not commit secrets to the repository."
        )
    return api_key


def output_path_for(config: EvalConfig) -> Path:
    model_name = safe_filename(config.model)
    return config.output_dir / f"judge_results__{model_name}.jsonl"


def existing_eval_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path):
        eval_id = record.get("_eval_id")
        if isinstance(eval_id, str) and record.get("_status") == "ok":
            ids.add(eval_id)
    return ids


async def judge_one(
    client: AsyncOpenAI,
    config: EvalConfig,
    validator: Draft202012Validator,
    system_prompt: str,
    user_template: str,
    example: dict[str, Any],
) -> dict[str, Any]:
    eval_id = stable_example_id(example)
    conversation = extract_conversation(example)
    messages = build_messages(system_prompt, user_template, conversation)
    stats = conversation_stats(conversation, messages)

    last_error: Exception | None = None
    last_metadata: dict[str, Any] | None = None
    started_at = time.time()
    for attempt in range(config.max_retries + 1):
        try:
            request: dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_output_tokens,
                "timeout": config.request_timeout_seconds,
            }
            if config.json_mode:
                request["response_format"] = {"type": "json_object"}
            if config.extra_body:
                request["extra_body"] = config.extra_body

            response = await client.chat.completions.create(**request)
            metadata = response_metadata(response)
            last_metadata = metadata
            content = extract_message_content(response)
            if not content:
                debug_payload = json.dumps(response_debug_payload(response), ensure_ascii=False)[:1500]
                raise JudgeOutputError(f"Judge returned an empty response. Response debug: {debug_payload}")

            judgment = parse_and_validate_judgment(content, validator)
            return {
                "_eval_id": eval_id,
                "_model": config.model,
                "_status": "ok",
                "_latency_seconds": round(time.time() - started_at, 3),
                "_finish_reason": metadata["finish_reason"],
                "_usage": metadata["usage"],
                "_example_stats": stats,
                "_source_repo": example.get("_source_repo"),
                "_subset": example.get("_subset"),
                "_split": example.get("_split"),
                "_sample_index": example.get("_sample_index"),
                "judgment": judgment,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= config.max_retries or not is_retryable_error(exc):
                break
            sleep_seconds = min(60.0, (2**attempt) + random.uniform(0.0, 1.0))
            await asyncio.sleep(sleep_seconds)

    return {
        "_eval_id": eval_id,
        "_model": config.model,
        "_status": "error",
        "_latency_seconds": round(time.time() - started_at, 3),
        "_finish_reason": last_metadata.get("finish_reason") if last_metadata else None,
        "_usage": last_metadata.get("usage") if last_metadata else None,
        "_example_stats": stats,
        "_source_repo": example.get("_source_repo"),
        "_subset": example.get("_subset"),
        "_split": example.get("_split"),
        "_sample_index": example.get("_sample_index"),
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": str(last_error) if last_error else "Unknown error",
    }


def select_examples(
    examples: list[dict[str, Any]],
    limit: int | None,
    per_subset: int | None,
) -> list[dict[str, Any]]:
    if limit is not None and per_subset is not None:
        raise ValueError("Use either --limit or --per-subset, not both.")
    if per_subset is None:
        return examples[:limit] if limit is not None else examples

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for example in examples:
        subset = str(example.get("_subset", "unknown_subset"))
        current_count = counts.get(subset, 0)
        if current_count >= per_subset:
            continue
        selected.append(example)
        counts[subset] = current_count + 1
    return selected


async def run_evaluation(
    config: EvalConfig,
    limit: int | None,
    per_subset: int | None,
    overwrite: bool,
    rerun_ok: bool,
) -> None:
    api_key = get_api_key(config.api_key_env)
    client = AsyncOpenAI(api_key=api_key, base_url=config.base_url, timeout=config.request_timeout_seconds)

    system_prompt = load_text(config.system_prompt_path)
    user_template = load_text(config.user_template_path)
    validator = build_validator(config.output_schema_path)

    output_path = output_path_for(config)
    if overwrite and output_path.exists():
        output_path.unlink()

    examples = read_jsonl(config.input_path)
    examples = select_examples(examples, limit=limit, per_subset=per_subset)

    completed_ids = set() if overwrite or rerun_ok else existing_eval_ids(output_path)
    pending = [example for example in examples if stable_example_id(example) not in completed_ids]

    print(f"Model: {config.model}")
    print(f"Input examples: {len(examples)}")
    print(f"Already completed: {len(examples) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(f"Output: {output_path.resolve()}")

    semaphore = asyncio.Semaphore(config.concurrency)

    async def bounded_judge(example: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await judge_one(client, config, validator, system_prompt, user_template, example)

    tasks = [asyncio.create_task(bounded_judge(example)) for example in pending]
    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging examples")
    for completed in progress:
        result = await completed
        await asyncio.to_thread(append_jsonl, output_path, [result])
        progress.set_postfix(status=result["_status"])

    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run async LLM-as-a-judge evaluation via OVH AI Endpoints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--per-subset", type=int, default=None, help="Evaluate N examples from each dataset subset.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Do not skip existing _eval_id records.")
    parser.add_argument("--rerun-ok", action="store_true", help="Append new attempts even for examples already judged ok.")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_eval_config(config_path, args.model, args.temperature, args.concurrency)
    asyncio.run(
        run_evaluation(
            config,
            limit=args.limit,
            per_subset=args.per_subset,
            overwrite=args.overwrite,
            rerun_ok=args.rerun_ok,
        )
    )


if __name__ == "__main__":
    main()
