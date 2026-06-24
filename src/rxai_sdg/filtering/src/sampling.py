from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from tqdm import tqdm


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return config


def resolve_dataset(repo: str, subset: str, preferred_split: str) -> Dataset:
    configs = get_dataset_config_names(repo)

    if subset in configs:
        loaded = load_dataset(repo, subset)
        return pick_split(loaded, preferred_split, repo, subset)

    loaded = load_dataset(repo)
    if isinstance(loaded, DatasetDict):
        if subset in loaded:
            return loaded[subset]
        if preferred_split in loaded and subset == preferred_split:
            return loaded[preferred_split]

    available = ", ".join(configs) if configs else "no named configs detected"
    loaded_keys = ", ".join(loaded.keys()) if isinstance(loaded, DatasetDict) else "single Dataset"
    raise ValueError(
        f"Could not resolve subset '{subset}' in '{repo}'. "
        f"Available configs: {available}. Available splits in default config: {loaded_keys}."
    )


def pick_split(loaded: Dataset | DatasetDict, preferred_split: str, repo: str, subset: str) -> Dataset:
    if isinstance(loaded, Dataset):
        return loaded
    if preferred_split in loaded:
        return loaded[preferred_split]
    if len(loaded) == 1:
        return next(iter(loaded.values()))
    available = ", ".join(loaded.keys())
    raise ValueError(
        f"Config '{subset}' in '{repo}' has no split '{preferred_split}'. "
        f"Available splits: {available}."
    )


def sample_dataset(dataset: Dataset, n: int, seed: int) -> Dataset:
    if len(dataset) == 0:
        raise ValueError("Cannot sample from an empty dataset.")

    sample_size = min(n, len(dataset))
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(sample_size))


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def materialize_subset(
    repo: str,
    subset: str,
    split: str,
    samples_per_subset: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset = resolve_dataset(repo, subset, split)
    sampled = sample_dataset(dataset, samples_per_subset, seed)

    records: list[dict[str, Any]] = []
    for idx, record in enumerate(sampled):
        enriched = dict(record)
        enriched["_source_repo"] = repo
        enriched["_subset"] = subset
        enriched["_split"] = split
        enriched["_sample_index"] = idx
        enriched["_sampling_seed"] = seed
        records.append(enriched)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample SFT examples from ReactiveAI/beta-reasoning.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config["dataset"]
    sampling_config = config["sampling"]

    repo = dataset_config["hf_repo"]
    subsets = dataset_config["subsets"]
    split = dataset_config.get("split", "train")
    samples_per_subset = int(sampling_config["samples_per_subset"])
    seed = int(sampling_config["random_seed"])
    output_dir = Path(sampling_config["output_dir"])
    formats = set(sampling_config.get("formats", ["jsonl"]))

    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for subset in tqdm(subsets, desc="Sampling subsets"):
        records = materialize_subset(repo, subset, split, samples_per_subset, seed)
        all_records.extend(records)

        subset_stem = subset.replace("/", "__")
        if "jsonl" in formats:
            write_jsonl(records, output_dir / f"{subset_stem}.jsonl")
        if "parquet" in formats:
            pd.DataFrame(records).to_parquet(output_dir / f"{subset_stem}.parquet", index=False)

    if "jsonl" in formats:
        write_jsonl(all_records, output_dir / "all_sampled.jsonl")
    if "parquet" in formats:
        pd.DataFrame(all_records).to_parquet(output_dir / "all_sampled.parquet", index=False)

    summary = pd.DataFrame(all_records).groupby("_subset").size().reset_index(name="n_samples")
    summary.to_csv(output_dir / "sampling_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {len(all_records)} total records to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
