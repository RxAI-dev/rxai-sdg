"""Notebook-facing orchestration for scoring conversational datasets.

``score_conversational_dataset`` runs the LLM-as-a-judge over every example of a
HuggingFace :class:`~datasets.Dataset`, writes the judge scores into a new column
and (optionally) periodically pushes the updated dataset back to the Hub so a
long run is not lost on an environment break.

It is intentionally threaded (not asyncio) so it composes cleanly inside Jupyter
notebooks, where a running event loop otherwise makes ``asyncio.run`` awkward.
The CLI ``evaluator`` keeps the async path; both share :mod:`rxai_sdg.filtering.judge`.

Example
-------
>>> from datasets import load_dataset
>>> from rxai_sdg.filtering import score_conversational_dataset
>>> ds = load_dataset("ReactiveAI/beta-reasoning", "Dolci-Think-SFT-32B-completed", split="train")
>>> scored = score_conversational_dataset(
...     ds,
...     judge_config={"api_url": OVH_API_URL, "api_key": OVH_API_KEY, "model_name": "Qwen3.5-397B-A17B"},
...     upload_config={
...         "dataset_name": "ReactiveAI/beta-reasoning",
...         "subset": "Dolci-Think-SFT-32B-completed",
...         "upload_every_n_examples": 1000,
...         "hf_token": HUGGINGFACE_TOKEN,
...     },
...     verbose=True,
...     concurrency=8,
... )
"""

from __future__ import annotations

import random
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI

from .judge import (
    DEFAULT_OUTPUT_SCHEMA_PATH,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_USER_TEMPLATE_PATH,
    JudgeOutputError,
    build_messages,
    build_validator,
    extract_conversation,
    extract_message_content,
    is_retryable_error,
    load_text,
    parse_and_validate_judgment,
)

try:  # Progress bar is optional; fall back to plain prints when missing.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is a declared dependency
    tqdm = None


DEFAULT_SCORES_FIELD = "scores"
DEFAULT_SPLIT = "train"


@dataclass
class ScoringSettings:
    """Runtime knobs for :func:`score_conversational_dataset` (``**other_config``)."""

    concurrency: int = 8
    temperature: float = 0.0
    max_output_tokens: int = 12000
    request_timeout_seconds: float = 420.0
    max_retries: int = 5
    json_mode: bool = True
    extra_body: dict[str, Any] | None = None
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH
    user_template_path: Path = DEFAULT_USER_TEMPLATE_PATH
    output_schema_path: Path = DEFAULT_OUTPUT_SCHEMA_PATH


@dataclass
class UploadSettings:
    dataset_name: str
    hf_token: str
    subset: str | None = None
    split: str = DEFAULT_SPLIT
    upload_every_n_examples: int | None = None
    scores_field: str = DEFAULT_SCORES_FIELD
    private: bool | None = None

    @property
    def config_name(self) -> str:
        return self.subset or "default"


@dataclass
class ScoringResult:
    """Summary returned by :func:`score_conversational_dataset`."""

    dataset: Dataset
    scores_field: str
    n_total: int
    n_ok: int
    n_error: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    uploaded: bool = False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScoringResult(n_total={self.n_total}, n_ok={self.n_ok}, "
            f"n_error={self.n_error}, uploaded={self.uploaded}, scores_field={self.scores_field!r})"
        )


def _build_settings(other_config: dict[str, Any]) -> ScoringSettings:
    known = {f for f in ScoringSettings.__dataclass_fields__}
    unknown = set(other_config) - known
    if unknown:
        raise TypeError(
            f"Unexpected scoring option(s): {sorted(unknown)}. "
            f"Supported options: {sorted(known)}."
        )
    settings = ScoringSettings(**{k: v for k, v in other_config.items() if k in known})
    settings.system_prompt_path = Path(settings.system_prompt_path)
    settings.user_template_path = Path(settings.user_template_path)
    settings.output_schema_path = Path(settings.output_schema_path)
    return settings


def _build_upload_settings(upload_config: dict[str, Any] | None) -> UploadSettings | None:
    if upload_config is None:
        return None
    config = dict(upload_config)
    if not config.get("dataset_name"):
        raise ValueError("upload_config requires 'dataset_name'.")
    if not config.get("hf_token"):
        raise ValueError("upload_config requires 'hf_token'.")
    known = {f for f in UploadSettings.__dataclass_fields__}
    unknown = set(config) - known
    if unknown:
        raise TypeError(
            f"Unexpected upload_config key(s): {sorted(unknown)}. "
            f"Supported keys: {sorted(known)}."
        )
    return UploadSettings(**config)


def _make_client(judge_config: dict[str, Any], timeout: float) -> OpenAI:
    missing = [key for key in ("api_url", "api_key", "model_name") if not judge_config.get(key)]
    if missing:
        raise ValueError(f"judge_config is missing required key(s): {missing}.")
    return OpenAI(api_key=judge_config["api_key"], base_url=judge_config["api_url"], timeout=timeout)


def _judge_conversation(
    client: OpenAI,
    model: str,
    conversation: dict[str, Any],
    system_prompt: str,
    user_template: str,
    validator: Any,
    settings: ScoringSettings,
) -> dict[str, Any]:
    """Call the judge for a single conversation, returning the validated judgment."""
    messages = build_messages(system_prompt, user_template, conversation)

    last_error: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            request: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": settings.temperature,
                "max_tokens": settings.max_output_tokens,
                "timeout": settings.request_timeout_seconds,
            }
            if settings.json_mode:
                request["response_format"] = {"type": "json_object"}
            if settings.extra_body:
                request["extra_body"] = settings.extra_body

            response = client.chat.completions.create(**request)
            content = extract_message_content(response)
            if not content:
                raise JudgeOutputError("Judge returned an empty response.")
            return parse_and_validate_judgment(content, validator)
        except Exception as exc:  # noqa: BLE001 - we classify via is_retryable_error
            last_error = exc
            if attempt >= settings.max_retries or not is_retryable_error(exc):
                break
            time.sleep(min(60.0, (2**attempt) + random.uniform(0.0, 1.0)))

    assert last_error is not None
    raise last_error


def _attach_scores(dataset: Dataset, scores_column: list[Any], scores_field: str) -> Dataset:
    base = dataset
    if scores_field in base.column_names:
        base = base.remove_columns([scores_field])
    return base.add_column(scores_field, list(scores_column))


def _data_dir_for(config_name: str) -> str:
    # Mirrors datasets.push_to_hub: the default config lives under data/, named
    # configs live under a directory named after the config.
    return config_name if config_name != "default" else "data"


def _existing_split_parquet_files(api, repo_id: str, data_dir: str, split: str) -> list[str] | None:
    """Return existing parquet shard paths for ``{data_dir}/{split}-*`` on the
    Hub, or ``None`` if the repo/config does not exist yet."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        return None
    prefix = f"{data_dir}/{split}-"
    shards = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    return shards if shards else None


def _commit_data_only(scored: Dataset, upload: UploadSettings, existing_shards: list[str], completed: int, total: int) -> None:
    """Upload the scored data for an existing config by committing parquet shards
    directly, WITHOUT rewriting the dataset card.

    ``datasets.push_to_hub`` regenerates ``README.md`` on every push; if the
    existing card is not clean UTF-8 that regeneration can drop other configs and
    write a garbled body. For a config that already exists we only need to
    refresh its parquet, so we commit the shards to ``{data_dir}/{split}-*`` and
    delete stale shards, leaving the human-written card and every other config
    untouched. (The card's cached row/byte stats for this split may read stale
    until the Hub reindexes; the parquet itself carries the new ``scores``.)
    """
    import math
    import os
    import shutil
    import tempfile

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi(token=upload.hf_token)
    data_dir = _data_dir_for(upload.config_name)
    split = upload.split

    try:
        nbytes = int(scored._estimate_nbytes())
    except Exception:
        nbytes = int(getattr(scored.data, "nbytes", 0) or 0)
    max_shard = 500 * 1024 * 1024  # 500 MB, matching datasets' default target
    num_shards = max(1, math.ceil(nbytes / max_shard)) if nbytes else 1

    tmpdir = tempfile.mkdtemp(prefix="rxai_scores_")
    try:
        additions = []
        for index in range(num_shards):
            shard = scored.shard(num_shards=num_shards, index=index, contiguous=True)
            name = f"{split}-{index:05d}-of-{num_shards:05d}.parquet"
            local_path = os.path.join(tmpdir, name)
            shard.to_parquet(local_path)
            additions.append(
                CommitOperationAdd(path_in_repo=f"{data_dir}/{name}", path_or_fileobj=local_path)
            )

        new_paths = {addition.path_in_repo for addition in additions}
        deletions = [
            CommitOperationDelete(path_in_repo=path) for path in existing_shards if path not in new_paths
        ]

        api.create_commit(
            repo_id=upload.dataset_name,
            repo_type="dataset",
            operations=additions + deletions,
            commit_message=f"Add '{upload.scores_field}' judge scores ({completed}/{total} examples)",
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _push_built(scored: Dataset, upload: UploadSettings, completed: int, total: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=upload.hf_token)
    data_dir = _data_dir_for(upload.config_name)
    existing_shards = _existing_split_parquet_files(api, upload.dataset_name, data_dir, upload.split)

    if existing_shards is not None:
        # Config/split already exists: refresh its parquet only, never touch the
        # card (avoids dropping other configs / corrupting the README body).
        _commit_data_only(scored, upload, existing_shards, completed, total)
        return

    # New repo or new config: let datasets register the config in the card. We do
    # NOT patch its card reads here, so a non-UTF-8 existing card fails safe
    # (aborts and preserves the card) rather than being silently rewritten.
    kwargs: dict[str, Any] = {
        "repo_id": upload.dataset_name,
        "config_name": upload.config_name,
        "split": upload.split,
        "token": upload.hf_token,
        "commit_message": f"Add '{upload.scores_field}' judge scores ({completed}/{total} examples)",
    }
    if upload.private is not None:
        kwargs["private"] = upload.private
    scored.push_to_hub(**kwargs)


def _safe_push(scored: Dataset, upload: UploadSettings, completed: int, total: int, verbose: bool) -> bool:
    """Push to the Hub, returning success. A failed upload is logged but never
    raised: in-memory scores are preserved so a transient Hub/network error
    cannot discard a long run's progress (the next upload simply retries)."""
    try:
        _push_built(scored, upload, completed, total)
        return True
    except Exception as exc:  # noqa: BLE001 - upload must not crash scoring
        if verbose:
            print(
                f"  ! upload to '{upload.dataset_name}' ({completed}/{total}) failed: "
                f"{type(exc).__name__}: {exc}. Scores are kept in memory; will retry on the next upload."
            )
            # Full traceback so the failing frame (card read vs. data/commit) is
            # diagnosable rather than hidden behind the one-line message.
            print(textwrap.indent(traceback.format_exc().rstrip(), "    "))
        return False


class _ScoresBuffer:
    """Thread-safe holder for the per-example scores being accumulated."""

    def __init__(self, size: int) -> None:
        self._lock = threading.Lock()
        self._scores: list[Any] = [None] * size

    def set(self, index: int, value: Any) -> None:
        with self._lock:
            self._scores[index] = value

    def snapshot(self) -> list[Any]:
        with self._lock:
            return list(self._scores)


def score_conversational_dataset(
    dataset: Dataset,
    judge_config: dict[str, Any],
    upload_config: dict[str, Any] | None = None,
    *,
    verbose: bool = True,
    **other_config: Any,
) -> ScoringResult:
    """Score every conversation in ``dataset`` and write judge scores to a column.

    Parameters
    ----------
    dataset:
        A HuggingFace ``Dataset`` whose examples follow the standard interaction
        format: each row has an ``interactions`` list (``{query, think, answer}``)
        and an optional ``system`` prompt.
    judge_config:
        ``{"api_url", "api_key", "model_name"}`` for the OpenAI-compatible judge
        endpoint.
    upload_config:
        Optional ``{"dataset_name", "hf_token", "subset"?, "split"?,
        "upload_every_n_examples"?, "scores_field"?, "private"?}``. When given,
        the dataset is pushed to the Hub every ``upload_every_n_examples``
        completed examples and once more at the end. When omitted, scoring runs
        purely in-memory and the scored dataset is returned.
    verbose:
        Show ``N done / N total`` progress and upload events.
    **other_config:
        Any field of :class:`ScoringSettings` (``concurrency``, ``temperature``,
        ``max_output_tokens``, ``request_timeout_seconds``, ``max_retries``,
        ``json_mode``, ``extra_body``, and prompt/schema path overrides).

    Returns
    -------
    ScoringResult
        Holds the scored ``dataset`` plus ok/error counts and per-example errors.
    """
    settings = _build_settings(other_config)
    upload = _build_upload_settings(upload_config)
    scores_field = upload.scores_field if upload else DEFAULT_SCORES_FIELD

    client = _make_client(judge_config, settings.request_timeout_seconds)
    model = judge_config["model_name"]
    system_prompt = load_text(settings.system_prompt_path)
    user_template = load_text(settings.user_template_path)
    validator = build_validator(settings.output_schema_path)

    total = len(dataset)
    scores_buffer = _ScoresBuffer(total)
    read_lock = threading.Lock()
    errors: list[dict[str, Any]] = []
    counts = {"ok": 0, "error": 0}

    upload_every = upload.upload_every_n_examples if upload else None

    def work(index: int) -> dict[str, Any]:
        # HuggingFace Dataset random access is guarded so concurrent worker
        # threads never read the underlying Arrow table at the same time; the
        # slow judge API call happens outside the lock.
        with read_lock:
            example = dataset[index]
        conversation = extract_conversation(example)
        return _judge_conversation(
            client, model, conversation, system_prompt, user_template, validator, settings
        )

    if verbose:
        print(
            f"Scoring {total} examples with judge '{model}' "
            f"(concurrency={settings.concurrency}, scores_field='{scores_field}')."
        )

    progress = None
    if verbose and tqdm is not None:
        progress = tqdm(total=total, desc="Scoring conversations")

    with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
        future_to_index = {executor.submit(work, i): i for i in range(total)}
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            completed += 1
            try:
                judgment = future.result()
                scores_buffer.set(index, judgment)
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["error"] += 1
                errors.append({"index": index, "error_type": type(exc).__name__, "error": str(exc)})
                if verbose and progress is None:
                    print(f"  ! example {index} failed: {type(exc).__name__}: {exc}")

            if progress is not None:
                progress.update(1)
                progress.set_postfix(ok=counts["ok"], err=counts["error"])
            elif verbose:
                print(f"  scored {completed}/{total} (ok={counts['ok']}, errors={counts['error']})")

            if upload and upload_every and completed % upload_every == 0 and completed < total:
                if verbose:
                    print(f"\nUploading partial scores to '{upload.dataset_name}' ({completed}/{total})...")
                partial = _attach_scores(dataset, scores_buffer.snapshot(), scores_field)
                _safe_push(partial, upload, completed, total, verbose)

    if progress is not None:
        progress.close()

    scored_dataset = _attach_scores(dataset, scores_buffer.snapshot(), scores_field)

    uploaded = False
    if upload:
        if verbose:
            print(f"\nUploading final scores to '{upload.dataset_name}' (config '{upload.config_name}', split '{upload.split}')...")
        uploaded = _safe_push(scored_dataset, upload, total, total, verbose)
        if verbose and uploaded:
            print(f"Uploaded scored dataset to '{upload.dataset_name}' (config '{upload.config_name}').")

    if verbose:
        print(f"\nDone. ok={counts['ok']}, errors={counts['error']}, total={total}, uploaded={uploaded}.")

    return ScoringResult(
        dataset=scored_dataset,
        scores_field=scores_field,
        n_total=total,
        n_ok=counts["ok"],
        n_error=counts["error"],
        errors=errors,
        uploaded=uploaded,
    )
