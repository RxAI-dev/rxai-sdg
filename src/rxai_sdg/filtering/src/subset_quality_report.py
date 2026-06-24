from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utils import read_jsonl, safe_filename


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_OUT_DIR = Path("data/results/subset_quality_report")
SCORE_COLUMNS = ["memory_score", "instruction_score", "freshness_score", "overall_score"]
PRIMARY_SCORE = "overall_score"


HYPOTHESIS_GROUPS = {
    "dolci": {
        "rank": 1,
        "subsets": ["Dolci-Think-SFT-32B-completed", "Dolci-Think-SFT-7B-completed"],
    },
    "r1_wildchat_middle": {
        "rank": 2,
        "subsets": ["tulu-wildchat-r1-completed", "wildchat-r1-completed"],
    },
    "old_real_wildchat": {
        "rank": 3,
        "subsets": ["real-chat-reasoning", "wild-chat-reasoning"],
    },
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def latest_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        eval_id = record.get("_eval_id")
        if isinstance(eval_id, str):
            latest[eval_id] = record
    return list(latest.values())


def flatten_result(record: dict[str, Any]) -> dict[str, Any]:
    judgment = record.get("judgment") if isinstance(record.get("judgment"), dict) else {}
    usage = record.get("_usage") if isinstance(record.get("_usage"), dict) else {}
    example_stats = record.get("_example_stats") if isinstance(record.get("_example_stats"), dict) else {}
    row = {
        "_eval_id": record.get("_eval_id"),
        "_model": record.get("_model"),
        "_status": record.get("_status"),
        "_subset": record.get("_subset"),
        "_sample_index": record.get("_sample_index"),
        "_latency_seconds": record.get("_latency_seconds"),
        "_finish_reason": record.get("_finish_reason"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "confidence": judgment.get("confidence"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "n_interactions": example_stats.get("n_interactions"),
        "serialized_chars": example_stats.get("serialized_chars"),
    }
    for column in SCORE_COLUMNS:
        row[column] = judgment.get(column)
    return row


def load_results(results_dir: Path, models: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in models:
        path = results_dir / f"judge_results__{safe_filename(model)}.jsonl"
        if not path.exists():
            print(f"Warning: missing result file for model: {model} ({path})")
            continue
        rows = [flatten_result(record) for record in latest_records(path)]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError(f"No result files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def status_summary(df: pd.DataFrame) -> pd.DataFrame:
    status = (
        df.groupby(["_model", "_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["_model", "_status"])
    )
    totals = df.groupby("_model").size().rename("total")
    status = status.merge(totals, on="_model")
    status["pct"] = (100 * status["n"] / status["total"]).round(2)
    return status


def coverage_summary(df: pd.DataFrame, expected_subsets: list[str], expected_per_subset: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(df["_model"].dropna().unique()):
        model_df = df[df["_model"] == model]
        for subset in expected_subsets:
            subset_df = model_df[model_df["_subset"] == subset]
            ok_count = int((subset_df["_status"] == "ok").sum())
            total_count = int(len(subset_df))
            rows.append(
                {
                    "_model": model,
                    "_subset": subset,
                    "n_total": total_count,
                    "n_ok": ok_count,
                    "n_expected": expected_per_subset,
                    "missing_ok": max(expected_per_subset - ok_count, 0),
                    "ok_pct": round(100 * ok_count / expected_per_subset, 2),
                }
            )
    return pd.DataFrame(rows)


def model_subset_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    for column in SCORE_COLUMNS:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")

    aggregations: dict[str, Any] = {
        "_eval_id": "count",
        "n_interactions": "mean",
        "serialized_chars": "mean",
    }
    for column in SCORE_COLUMNS:
        aggregations[column] = ["mean", "std", "median"]

    summary = ok.groupby(["_model", "_subset"]).agg(aggregations)
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    summary = summary.rename(columns={"_eval_id_count": "n_ok"})
    return summary.reset_index().round(3)


def ensemble_by_example(df: pd.DataFrame, primary_model: str | None) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    for column in SCORE_COLUMNS:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")

    grouped = ok.groupby(["_eval_id", "_subset", "_sample_index"], dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        eval_id, subset, sample_index = key
        row: dict[str, Any] = {
            "_eval_id": eval_id,
            "_subset": subset,
            "_sample_index": sample_index,
            "n_models_ok": group["_model"].nunique(),
        }
        for column in SCORE_COLUMNS:
            row[f"{column}_ensemble_mean"] = group[column].mean()
            row[f"{column}_ensemble_std"] = group[column].std()

        primary_rows = group[group["_model"] == primary_model] if primary_model else pd.DataFrame()
        if primary_model and not primary_rows.empty:
            for column in SCORE_COLUMNS:
                row[f"{column}_primary"] = primary_rows.iloc[-1][column]
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def subset_ranking(ensemble: pd.DataFrame, primary_model: str | None) -> pd.DataFrame:
    if ensemble.empty:
        return pd.DataFrame()

    score_suffix = "_primary" if primary_model and f"{PRIMARY_SCORE}_primary" in ensemble.columns else "_ensemble_mean"
    grouped = ensemble.groupby("_subset")
    summary = grouped.agg(
        n_examples=("_eval_id", "count"),
        n_models_ok_avg=("n_models_ok", "mean"),
        overall_mean=(f"{PRIMARY_SCORE}{score_suffix}", "mean"),
        overall_std=(f"{PRIMARY_SCORE}{score_suffix}", "std"),
        memory_mean=(f"memory_score{score_suffix}", "mean"),
        instruction_mean=(f"instruction_score{score_suffix}", "mean"),
        freshness_mean=(f"freshness_score{score_suffix}", "mean"),
    )
    summary["quality_index"] = summary["overall_mean"]
    summary = summary.reset_index()
    summary["rank"] = summary["quality_index"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(["rank", "_subset"]).round(3)


def hypothesis_summary(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_name, group_config in HYPOTHESIS_GROUPS.items():
        subsets = group_config["subsets"]
        group_df = ranking[ranking["_subset"].isin(subsets)]
        if group_df.empty:
            continue
        rows.append(
            {
                "hypothesis_group": group_name,
                "expected_rank": group_config["rank"],
                "n_subsets": len(group_df),
                "mean_quality_index": group_df["quality_index"].mean(),
                "mean_overall": group_df["overall_mean"].mean(),
                "subsets": ", ".join(group_df["_subset"].tolist()),
            }
        )
    result = pd.DataFrame(rows).sort_values("expected_rank").round(3)
    if not result.empty:
        result["observed_rank"] = result["mean_quality_index"].rank(method="dense", ascending=False).astype(int)
        result["matches_expected_order_so_far"] = result["observed_rank"] == result["expected_rank"]
    return result


def disagreement_summary(ensemble: pd.DataFrame) -> pd.DataFrame:
    if ensemble.empty or "overall_score_ensemble_std" not in ensemble.columns:
        return pd.DataFrame()
    return (
        ensemble.sort_values("overall_score_ensemble_std", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )


def frame_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data."
    return "```text\n" + df.to_string(index=False) + "\n```"


def write_markdown_report(
    out_path: Path,
    models: list[str],
    primary_model: str | None,
    status: pd.DataFrame,
    coverage: pd.DataFrame,
    ranking: pd.DataFrame,
    hypothesis: pd.DataFrame,
) -> None:
    lines = [
        "# Subset Quality Report",
        "",
        f"Models: {', '.join(models)}",
        f"Primary model for ranking: {primary_model or 'ensemble mean'}",
        "",
        "## Status Summary",
        "",
        frame_block(status),
        "",
        "## Coverage Summary",
        "",
        frame_block(coverage),
        "",
        "## Subset Ranking",
        "",
        frame_block(ranking),
        "",
        "## Hypothesis Check",
        "",
        frame_block(hypothesis),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final subset-quality report from judge results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--primary-model", type=str, default=None)
    parser.add_argument("--expected-per-subset", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    config = load_config(args.config)
    expected_subsets = config["dataset"]["subsets"]
    results_dir = Path(config["evaluation"]["output_dir"])

    df = load_results(results_dir, args.models)
    args.out.mkdir(parents=True, exist_ok=True)

    status = status_summary(df)
    coverage = coverage_summary(df, expected_subsets, args.expected_per_subset)
    per_model_subset = model_subset_summary(df)
    ensemble = ensemble_by_example(df, args.primary_model)
    ranking = subset_ranking(ensemble, args.primary_model)
    hypothesis = hypothesis_summary(ranking)
    disagreements = disagreement_summary(ensemble)

    df.to_csv(args.out / "judge_results_latest_flat.csv", index=False)
    status.to_csv(args.out / "status_summary.csv", index=False)
    coverage.to_csv(args.out / "coverage_summary.csv", index=False)
    per_model_subset.to_csv(args.out / "model_subset_summary.csv", index=False)
    ensemble.to_csv(args.out / "ensemble_by_example.csv", index=False)
    ranking.to_csv(args.out / "subset_quality_ranking.csv", index=False)
    hypothesis.to_csv(args.out / "hypothesis_summary.csv", index=False)
    disagreements.to_csv(args.out / "top_model_disagreements.csv", index=False)
    write_markdown_report(args.out / "subset_quality_report.md", args.models, args.primary_model, status, coverage, ranking, hypothesis)

    print("\nStatus summary")
    print(status.to_string(index=False))
    print("\nCoverage summary")
    print(coverage.to_string(index=False))
    print("\nSubset ranking")
    print(ranking.to_string(index=False) if not ranking.empty else "No ranking data.")
    print("\nHypothesis summary")
    print(hypothesis.to_string(index=False) if not hypothesis.empty else "No hypothesis data.")
    print(f"\nSaved final subset report to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
