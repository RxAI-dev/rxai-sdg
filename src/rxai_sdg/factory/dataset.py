"""HuggingFace dataset integration for the Data Factory.

After conversations for all provided seeds are generated, the collected
:class:`~rxai_sdg.factory.schemas.ConversationRecord` list is turned into a
HuggingFace ``Dataset`` and either **appended** to an existing dataset (resolved
by ``dataset_id`` + ``config_name`` + ``split``) or used to **create** a new one.

This mirrors the postprocessor pattern used by the other ``rxai_sdg`` generators
(``MrlGeneratorPostprocessor`` / ``HybridGeneratorPostprocessor``).

Row schema
----------
To keep a stable Arrow schema across many independent generation sessions (e.g.
10-20 parallel notebooks all appending to the same dataset), the deeply-nested
and variable-keyed parts of a record (``turns``, ``fact_ledger``,
``cross_turn_checks``, ``holistic_score``) are stored as **JSON strings**.
Scalar/seed fields are stored as native columns for easy filtering. Consumers
``json.loads`` the JSON columns. Use :func:`row_to_record` to reconstruct a full
:class:`ConversationRecord`.

``datasets`` / ``huggingface_hub`` are imported lazily so importing the factory
package does not require them.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from .schemas import ConversationRecord


def _ensure_utf8_card(raw: bytes) -> Optional[bytes]:
    """Return UTF-8-clean bytes for a dataset card, or ``None`` if already valid.

    An existing dataset card (``README.md``) that contains a non-UTF-8 byte (e.g.
    a stray cp1252 ``0x85``) makes ``datasets.push_to_hub`` crash with
    ``UnicodeDecodeError`` while it reads the card to merge split/config metadata
    on append. We re-decode leniently (preserving all valid UTF-8 content, the
    YAML config block included) and strip any leading BOM / replacement / control
    junk so the YAML frontmatter still parses, then re-encode as clean UTF-8.
    """
    try:
        raw.decode("utf-8")
        return None  # already valid -> no repair needed
    except UnicodeDecodeError:
        pass
    text = raw.decode("utf-8", errors="replace")
    text = text.lstrip("\ufeff\ufffd\x85\x00\r\n\t ")
    return text.encode("utf-8")


#: native (non-JSON) columns
SCALAR_COLUMNS = [
    "conversation_id", "dataset", "first_query", "category", "domain", "lang",
    "is_haystack", "mode", "length",
]
#: columns stored as JSON strings for a stable, append-safe schema
JSON_COLUMNS = ["turns", "fact_ledger", "cross_turn_checks", "holistic_score",
                "factory_models"]


def record_to_row(record: ConversationRecord) -> dict[str, Any]:
    """Flatten a record into a single, append-safe dataset row."""
    d = record.to_dict()
    seed = d["source_seed"]
    row: dict[str, Any] = {
        "conversation_id": d["conversation_id"],
        "dataset": seed["dataset"],
        "first_query": seed["first_query"],
        "category": seed["category"],
        "domain": seed["domain"],
        "lang": seed["lang"],
        "is_haystack": seed["is_haystack"],
        "mode": d["mode"],
        "length": d["length"],
    }
    for col in JSON_COLUMNS:
        row[col] = json.dumps(d[col], ensure_ascii=False)
    return row


def row_to_record(row: dict[str, Any]) -> ConversationRecord:
    """Reconstruct a :class:`ConversationRecord` from a dataset row."""
    d: dict[str, Any] = {
        "conversation_id": row["conversation_id"],
        "source_seed": {
            "dataset": row["dataset"],
            "first_query": row["first_query"],
            "category": row["category"],
            "domain": row["domain"],
            "lang": row["lang"],
            "is_haystack": row["is_haystack"],
        },
        "mode": row["mode"],
        "length": row["length"],
    }
    for col in JSON_COLUMNS:
        d[col] = json.loads(row[col]) if isinstance(row[col], str) else row[col]
    return ConversationRecord.from_dict(d)


class FactoryDatasetPostprocessor:
    """Collects records and writes them to a HuggingFace dataset.

    Example::

        post = FactoryDatasetPostprocessor(records, dataset_id="org/rxt-factory",
                                           token="hf_...")
        post.push_to_hf_hub(append=True)   # append to existing or create new
    """

    def __init__(
        self,
        records: Optional[Iterable[ConversationRecord]] = None,
        dataset_id: Optional[str] = None,
        config_name: Optional[str] = None,
        split: str = "train",
        token: Optional[str] = None,
    ):
        self.records: list[ConversationRecord] = list(records or [])
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.token = token

    # ------------------------------------------------------------------ build
    def add(self, records: Iterable[ConversationRecord]) -> "FactoryDatasetPostprocessor":
        self.records.extend(records)
        return self

    def to_rows(self) -> list[dict[str, Any]]:
        return [record_to_row(r) for r in self.records]

    def to_dataset(self):
        """Build a (new) :class:`datasets.Dataset` from the collected records."""
        from datasets import Dataset  # lazy import
        return Dataset.from_list(self.to_rows())

    # ------------------------------------------------------------- hub access
    def _load_existing(self):
        """Return the existing remote dataset split, or ``None`` if absent."""
        from datasets import load_dataset  # lazy import
        try:
            if self.config_name is not None:
                return load_dataset(self.dataset_id, self.config_name,
                                    split=self.split, token=self.token)
            return load_dataset(self.dataset_id, split=self.split, token=self.token)
        except Exception:
            return None  # dataset/config/split does not exist yet -> create

    def build(self, append: bool = True):
        """Build the final dataset, optionally concatenating with the existing one."""
        new_ds = self.to_dataset()
        if append and self.dataset_id is not None:
            existing = self._load_existing()
            if existing is not None and len(existing) > 0:
                from datasets import concatenate_datasets  # lazy import
                new_ds = concatenate_datasets([existing, new_ds])
        return new_ds

    def push_to_hf_hub(
        self,
        dataset_id: Optional[str] = None,
        config_name: Optional[str] = None,
        split: Optional[str] = None,
        token: Optional[str] = None,
        append: bool = True,
    ):
        """Create or append-then-push the dataset to the Hub."""
        if dataset_id is not None:
            self.dataset_id = dataset_id
        if config_name is not None:
            self.config_name = config_name
        if split is not None:
            self.split = split
        if token is not None:
            self.token = token
        if self.dataset_id is None:
            raise ValueError("dataset_id is required to push to the Hub")

        ds = self.build(append=append)
        # Repair the remote dataset card BEFORE pushing: push_to_hub reads the
        # existing README.md as strict UTF-8 to merge split/config metadata, and a
        # non-UTF-8 byte in it (e.g. a stray cp1252 0x85) otherwise crashes the
        # append with UnicodeDecodeError before any data is written.
        self._repair_remote_dataset_card()
        kwargs: dict[str, Any] = {"repo_id": self.dataset_id, "split": self.split,
                                  "token": self.token}
        if self.config_name is not None:
            kwargs["config_name"] = self.config_name
        ds.push_to_hub(**kwargs)
        return ds

    def _repair_remote_dataset_card(self) -> bool:
        """Rewrite the remote ``README.md`` as clean UTF-8 if it is not already.

        Returns ``True`` if a repair was uploaded. No-ops (returns ``False``) when
        the repo/card does not exist yet or the card is already valid UTF-8. Best
        effort: any error here is swallowed so a transient Hub issue does not mask
        the real push error.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except Exception:
            return False
        try:
            path = hf_hub_download(self.dataset_id, "README.md",
                                   repo_type="dataset", token=self.token)
            with open(path, "rb") as fh:
                raw = fh.read()
            fixed = _ensure_utf8_card(raw)
            if fixed is None:
                return False  # already valid (or no bytes) -> nothing to do
            HfApi(token=self.token).upload_file(
                path_or_fileobj=fixed, path_in_repo="README.md",
                repo_id=self.dataset_id, repo_type="dataset", token=self.token,
                commit_message="Normalise dataset card encoding to UTF-8")
            return True
        except Exception:
            return False

    def save_to_disk(self, path: str):
        ds = self.to_dataset()
        ds.save_to_disk(path)
        return ds
