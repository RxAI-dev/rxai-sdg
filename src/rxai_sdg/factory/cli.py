"""Command-line / batch runner for the Data Factory (spec §11.10).

Examples
--------
Smoke test (no network; deterministic mock that satisfies constraints)::

    python -m rxai_sdg.factory.cli --smoke -n 5 --out out.jsonl

Real run from a JSONL seed file with an OpenAI-compatible endpoint::

    python -m rxai_sdg.factory.cli \
        --seeds seeds.jsonl --n 100 --config factory.json \
        --responder-model gpt-4 --simulator-model gpt-4o-mini \
        --api-key $OPENAI_API_KEY --out conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Optional

from .clients import MockLLMClient, OpenAILLMClient
from .config import FactoryConfig
from .factory_runner import DataFactory
from .schemas import validate_record
from .seed_curator import DatasetSpec
from .testing import constraint_satisfying_handler


def _load_seed_records(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return records
    # Support both JSONL and a single JSON array.
    if text.lstrip().startswith("["):
        return list(json.loads(text))
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data Factory batch runner")
    p.add_argument("--seeds", help="path to a JSONL/JSON seed file")
    p.add_argument("-n", "--n", type=int, default=5, help="number of conversations")
    p.add_argument("--config", help="path to a JSON/YAML FactoryConfig file")
    p.add_argument("--band", default=None, help="length band (basic|generalization|short)")
    p.add_argument("--out", default="factory_out.jsonl", help="output JSONL path")
    p.add_argument("--smoke", action="store_true",
                   help="run with a deterministic mock client (no network)")
    p.add_argument("--responder-model", default="gpt-4")
    p.add_argument("--simulator-model", default="gpt-4o-mini")
    p.add_argument("--api-url", default="https://api.openai.com/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--use-ollama", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--validate", action="store_true", default=True,
                   help="validate every emitted record against the schema")
    # optional HuggingFace output
    p.add_argument("--hf-dataset", default=None,
                   help="push the generated records to this HuggingFace dataset id")
    p.add_argument("--hf-config", default=None, help="HuggingFace dataset config/subset name")
    p.add_argument("--hf-split", default="train", help="HuggingFace dataset split")
    p.add_argument("--hf-token", default=None, help="HuggingFace auth token")
    p.add_argument("--no-append", action="store_true",
                   help="create/overwrite instead of appending to the existing dataset")
    return p


def run(args: argparse.Namespace) -> int:
    if args.config:
        config = FactoryConfig.from_file(args.config)
    else:
        config = FactoryConfig(seed=args.seed)
    if config.seed is None:
        config.seed = args.seed

    rng = random.Random(config.seed)

    if args.seeds:
        seed_records = _load_seed_records(args.seeds)
    else:
        seed_records = [
            {"query": "Explain how entropy relates to information.", "category": "stem"},
            {"query": "Write a short paragraph about lighthouses.", "category": "writing"},
            {"query": "What is 17 * 23, and why does the method work?", "category": "math"},
            {"query": "Outline a function to reverse a linked list.", "category": "coding"},
        ]
    dataset_spec = DatasetSpec(name="cli_seeds", records=seed_records,
                               lang=config.lang, haystack_fraction=config.haystack_fraction)

    if args.smoke:
        responder_client = MockLLMClient(handler=constraint_satisfying_handler)
        simulator_client = None  # use deterministic templates
    else:
        responder_client = OpenAILLMClient(
            model_name=args.responder_model, api_url=args.api_url,
            api_key=args.api_key, use_ollama=args.use_ollama)
        simulator_client = OpenAILLMClient(
            model_name=args.simulator_model, api_url=args.api_url,
            api_key=args.api_key, use_ollama=args.use_ollama)

    factory = DataFactory(
        config, responder_client, simulator_client=simulator_client, rng=rng)
    records = factory.generate(
        dataset_spec, n_conversations=args.n, band=args.band)

    if args.validate:
        for rec in records:
            validate_record(rec.to_dict())

    n = factory.write_jsonl(records, args.out)

    if args.hf_dataset:
        factory.save_to_hub(
            dataset_id=args.hf_dataset, config_name=args.hf_config,
            split=args.hf_split, token=args.hf_token, append=not args.no_append)
        print(f"pushed {len(records)} records -> {args.hf_dataset} "
              f"(split={args.hf_split}, append={not args.no_append})", file=sys.stderr)

    stats = factory.stats
    print(f"seeds_used={stats.seeds_used} built={stats.conversations_built} "
          f"discarded={stats.conversations_discarded} records={n}", file=sys.stderr)
    print(f"regenerations={stats.loop.total_regenerations} "
          f"intent_resamples={stats.loop.intent_resamples} "
          f"downweighted={stats.loop.downweighted}", file=sys.stderr)
    print(f"wrote {n} records -> {args.out}", file=sys.stderr)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
