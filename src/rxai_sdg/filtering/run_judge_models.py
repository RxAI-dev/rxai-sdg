from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
# Directory that contains the top-level ``rxai_sdg`` package, so spawned
# subprocesses can import the evaluator even without an editable install.
SRC_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_MODULE = "rxai_sdg.filtering.evaluator"


def load_models(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)
    models = config["evaluation"]["judge_models"]
    if not isinstance(models, list) or not models:
        raise ValueError("evaluation.judge_models must be a non-empty list.")
    return [str(model) for model in models]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluator for every configured judge model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-subset", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun-ok", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config

    models = args.models if args.models else load_models(config_path)
    failures: list[tuple[str, int]] = []
    for model in models:
        command = [
            sys.executable,
            "-m",
            EVALUATOR_MODULE,
            "--config",
            str(config_path),
            "--model",
            model,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.per_subset is not None:
            command.extend(["--per-subset", str(args.per_subset)])
        if args.temperature is not None:
            command.extend(["--temperature", str(args.temperature)])
        if args.concurrency is not None:
            command.extend(["--concurrency", str(args.concurrency)])
        if args.overwrite:
            command.append("--overwrite")
        if args.rerun_ok:
            command.append("--rerun-ok")

        print(f"\n=== Running judge model: {model} ===", flush=True)
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_ROOT), existing_pythonpath]))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures.append((model, completed.returncode))
            print(f"Model failed but runner will continue: {model} (exit={completed.returncode})", flush=True)

    if failures:
        print("\nCompleted with model failures:", flush=True)
        for model, returncode in failures:
            print(f"- {model}: exit={returncode}", flush=True)
    else:
        print("\nAll judge models completed successfully.", flush=True)


if __name__ == "__main__":
    main()
