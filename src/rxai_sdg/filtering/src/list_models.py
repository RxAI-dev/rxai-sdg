from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="List available OVH/OpenAI-compatible model IDs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--contains", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    eval_config = config["evaluation"]
    load_dotenv()
    api_key = os.getenv(eval_config["api_key_env"]) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {eval_config['api_key_env']}")

    client = OpenAI(api_key=api_key, base_url=eval_config["base_url"])
    models = sorted(model.id for model in client.models.list().data)
    if args.contains:
        needle = args.contains.lower()
        models = [model for model in models if needle in model.lower()]

    for model in models:
        print(model)


if __name__ == "__main__":
    main()
