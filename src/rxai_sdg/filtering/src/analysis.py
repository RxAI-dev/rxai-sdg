from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from utils import read_jsonl, safe_filename


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
SCORE_COLUMNS = [
    "memory_score",
    "instruction_score",
    "freshness_score",
    "overall_score",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def latest_records(path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        eval_id = record.get("_eval_id")
        if isinstance(eval_id, str):
            if eval_id in latest and latest[eval_id].get("_status") == "ok" and record.get("_status") != "ok":
                continue
            latest[eval_id] = record
    return list(latest.values())


def flatten_result(record: dict[str, Any]) -> dict[str, Any]:
    judgment = (
        record.get("judgment") if isinstance(record.get("judgment"), dict) else {}
    )
    usage = record.get("_usage") if isinstance(record.get("_usage"), dict) else {}
    example_stats = (
        record.get("_example_stats")
        if isinstance(record.get("_example_stats"), dict)
        else {}
    )
    flattened = {
        "_eval_id": record.get("_eval_id"),
        "_model": record.get("_model"),
        "_status": record.get("_status"),
        "_subset": record.get("_subset"),
        "_sample_index": record.get("_sample_index"),
        "_latency_seconds": record.get("_latency_seconds"),
        "_finish_reason": record.get("_finish_reason"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "n_interactions": example_stats.get("n_interactions"),
        "n_messages": example_stats.get("n_messages"),
        "serialized_chars": example_stats.get("serialized_chars"),
        "prompt_chars": example_stats.get("prompt_chars"),
        "confidence": judgment.get("confidence"),
    }
    for column in SCORE_COLUMNS:
        flattened[column] = judgment.get(column)
    return flattened


def load_model_results(results_dir: Path, model: str) -> pd.DataFrame:
    path = results_dir / f"judge_results__{safe_filename(model)}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [flatten_result(record) for record in latest_records(path)]
    return pd.DataFrame(rows)


def load_model_attempts(results_dir: Path, model: str) -> pd.DataFrame:
    path = results_dir / f"judge_results__{safe_filename(model)}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [flatten_result(record) for record in read_jsonl(path)]
    return pd.DataFrame(rows)


def summarize_status(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
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


def summarize_errors(attempts: pd.DataFrame) -> pd.DataFrame:
    errors = attempts[attempts["_status"] == "error"].copy()
    if errors.empty:
        return pd.DataFrame()
    error_columns = ["_model", "error_type", "error"]
    available_columns = [column for column in error_columns if column in errors.columns]
    return (
        errors.groupby(available_columns, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["_model", "n"], ascending=[True, False])
    )


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    aggregations = {column: ["count", "mean", "std"] for column in SCORE_COLUMNS}
    summary = ok.groupby("_model").agg(aggregations)
    summary.columns = [
        "_".join(column).rstrip("_") for column in summary.columns.to_flat_index()
    ]
    return summary.reset_index().round(3)


def pairwise_agreement(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    models = sorted(ok["_model"].dropna().unique())
    for left_idx, left_model in enumerate(models):
        for right_model in models[left_idx + 1 :]:
            left = ok[ok["_model"] == left_model].set_index("_eval_id")
            right = ok[ok["_model"] == right_model].set_index("_eval_id")
            common_ids = left.index.intersection(right.index)
            if len(common_ids) == 0:
                continue
            row: dict[str, Any] = {
                "left_model": left_model,
                "right_model": right_model,
                "n_common": len(common_ids),
            }
            for column in SCORE_COLUMNS:
                left_scores = pd.to_numeric(
                    left.loc[common_ids, column], errors="coerce"
                )
                right_scores = pd.to_numeric(
                    right.loc[common_ids, column], errors="coerce"
                )
                diffs = (left_scores - right_scores).abs()
                row[f"{column}_mae"] = diffs.mean()
                row[f"{column}_corr"] = left_scores.rank().corr(right_scores.rank())
                row[f"{column}_weighted_kappa"] = weighted_cohen_kappa(
                    left_scores, right_scores
                )
            rows.append(row)
    return pd.DataFrame(rows).round(3)


def reference_bias(df: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty or reference_model not in set(ok["_model"]):
        return pd.DataFrame()

    reference = ok[ok["_model"] == reference_model].set_index("_eval_id")
    rows: list[dict[str, Any]] = []
    for model in sorted(model for model in ok["_model"].dropna().unique() if model != reference_model):
        candidate = ok[ok["_model"] == model].set_index("_eval_id")
        common_ids = reference.index.intersection(candidate.index)
        if len(common_ids) == 0:
            continue
        row: dict[str, Any] = {
            "reference_model": reference_model,
            "model": model,
            "n_common": len(common_ids),
        }
        for column in SCORE_COLUMNS:
            ref_scores = pd.to_numeric(reference.loc[common_ids, column], errors="coerce")
            model_scores = pd.to_numeric(candidate.loc[common_ids, column], errors="coerce")
            diff = model_scores - ref_scores
            row[f"{column}_mean_bias"] = diff.mean()
            row[f"{column}_median_bias"] = diff.median()
            row[f"{column}_pct_exact"] = (diff == 0).mean() * 100
            row[f"{column}_pct_within_1"] = (diff.abs() <= 1).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def top_disagreements(df: pd.DataFrame, reference_model: str, limit: int = 25) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty or reference_model not in set(ok["_model"]):
        return pd.DataFrame()

    reference = ok[ok["_model"] == reference_model].set_index("_eval_id")
    rows: list[dict[str, Any]] = []
    for model in sorted(model for model in ok["_model"].dropna().unique() if model != reference_model):
        candidate = ok[ok["_model"] == model].set_index("_eval_id")
        common_ids = reference.index.intersection(candidate.index)
        for eval_id in common_ids:
            row: dict[str, Any] = {
                "_eval_id": eval_id,
                "_subset": candidate.loc[eval_id, "_subset"],
                "_sample_index": candidate.loc[eval_id, "_sample_index"],
                "reference_model": reference_model,
                "model": model,
            }
            total_abs_diff = 0.0
            for column in SCORE_COLUMNS:
                ref_score = pd.to_numeric(reference.loc[eval_id, column], errors="coerce")
                model_score = pd.to_numeric(candidate.loc[eval_id, column], errors="coerce")
                abs_diff = abs(float(model_score) - float(ref_score))
                row[f"reference_{column}"] = ref_score
                row[f"model_{column}"] = model_score
                row[f"{column}_abs_diff"] = abs_diff
                total_abs_diff += abs_diff
            row["total_abs_diff"] = total_abs_diff
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_abs_diff", ascending=False).head(limit).round(3)


def weighted_cohen_kappa(
    left: pd.Series, right: pd.Series, min_rating: int = 1, max_rating: int = 10
) -> float:
    paired = pd.concat([left, right], axis=1).dropna().astype(int)
    if paired.empty:
        return math.nan

    categories = list(range(min_rating, max_rating + 1))
    k = len(categories)
    index = {rating: idx for idx, rating in enumerate(categories)}
    observed = np.zeros((k, k), dtype=float)

    for left_score, right_score in paired.itertuples(index=False):
        if left_score in index and right_score in index:
            observed[index[left_score], index[right_score]] += 1

    n = observed.sum()
    if n == 0:
        return math.nan

    observed = observed / n
    left_marginal = observed.sum(axis=1)
    right_marginal = observed.sum(axis=0)
    expected = np.outer(left_marginal, right_marginal)

    weights = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) / (k - 1)) ** 2

    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else math.nan
    return 1.0 - observed_disagreement / expected_disagreement


def fleiss_kappa_for_column(
    df: pd.DataFrame, column: str, min_rating: int = 1, max_rating: int = 10
) -> dict[str, Any]:
    ok = df[df["_status"] == "ok"].copy()
    pivot = ok.pivot_table(
        index="_eval_id", columns="_model", values=column, aggfunc="last"
    )
    complete = pivot.dropna()
    if complete.empty or complete.shape[1] < 2:
        return {
            "metric": column,
            "n_items": 0,
            "n_raters": complete.shape[1],
            "fleiss_kappa": math.nan,
        }

    categories = list(range(min_rating, max_rating + 1))
    n_raters = complete.shape[1]
    rating_counts = []
    for _, row in complete.astype(int).iterrows():
        counts = [int((row == rating).sum()) for rating in categories]
        rating_counts.append(counts)

    matrix = np.array(rating_counts, dtype=float)
    n_items = matrix.shape[0]
    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    p_i = ((matrix * matrix).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_e = float((p_j * p_j).sum())
    kappa = (p_bar - p_e) / (1 - p_e) if p_e != 1 else math.nan
    return {
        "metric": column,
        "n_items": n_items,
        "n_raters": n_raters,
        "fleiss_kappa": kappa,
    }


def fleiss_kappa(df: pd.DataFrame) -> pd.DataFrame:
    rows = [fleiss_kappa_for_column(df, column) for column in SCORE_COLUMNS]
    return pd.DataFrame(rows).round(3)


def operational_stats(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    for column in [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "_latency_seconds",
    ]:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")
    stats = (
        ok.groupby("_model")
        .agg(
            n_ok=("_eval_id", "count"),
            avg_latency_seconds=("_latency_seconds", "mean"),
            p50_latency_seconds=("_latency_seconds", "median"),
            p95_latency_seconds=(
                "_latency_seconds",
                lambda values: values.quantile(0.95),
            ),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
            avg_reasoning_tokens=("reasoning_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_n_interactions=("n_interactions", "mean"),
            avg_serialized_chars=("serialized_chars", "mean"),
        )
        .reset_index()
        .round(3)
    )
    return stats


def restrict_to_common_examples(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["_status"] == "ok"].copy()
    if ok.empty:
        return df.iloc[0:0].copy()

    n_models = ok["_model"].nunique()
    common_ids = (
        ok.groupby("_eval_id")["_model"]
        .nunique()
        .loc[lambda counts: counts == n_models]
        .index
    )
    return df[df["_eval_id"].isin(common_ids)].copy()


def dataset_shape_stats(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.drop_duplicates("_eval_id").copy()
    if latest.empty:
        return pd.DataFrame()
    return (
        latest.groupby("_subset")
        .agg(
            n_examples=("_eval_id", "count"),
            avg_n_interactions=("n_interactions", "mean"),
            min_n_interactions=("n_interactions", "min"),
            max_n_interactions=("n_interactions", "max"),
            avg_serialized_chars=("serialized_chars", "mean"),
        )
        .reset_index()
        .round(3)
    )


def intra_rater_reliability(attempts: pd.DataFrame) -> pd.DataFrame:
    ok = attempts[attempts["_status"] == "ok"].copy()
    duplicated = ok.groupby(["_model", "_eval_id"]).filter(lambda group: len(group) > 1)
    if duplicated.empty:
        return pd.DataFrame(
            [
                {
                    "note": (
                        "No repeated ok judgments per model/example. "
                        "Intra-rater reliability requires repeated runs, ideally at temperature > 0."
                    )
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    for model, model_df in duplicated.groupby("_model"):
        row: dict[str, Any] = {
            "_model": model,
            "n_repeated_examples": model_df["_eval_id"].nunique(),
        }
        for column in SCORE_COLUMNS:
            per_example_std = model_df.groupby("_eval_id")[column].std().dropna()
            row[f"{column}_mean_repeat_std"] = per_example_std.mean()
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze judge-model validation results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="Analyze only examples judged ok by every selected model.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/results/judge_validation_summary")
    )
    args = parser.parse_args()
    args.out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    eval_config = config["evaluation"]
    results_dir = Path(eval_config["output_dir"])
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    models = args.models if args.models else eval_config["judge_models"]

    frames = [load_model_results(results_dir, model) for model in models]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError(f"No judge result files found in {results_dir}")

    attempt_frames = [load_model_attempts(results_dir, model) for model in models]
    attempt_frames = [frame for frame in attempt_frames if not frame.empty]

    df = pd.concat(frames, ignore_index=True)
    attempts = pd.concat(attempt_frames, ignore_index=True) if attempt_frames else df
    if args.common_only:
        df = restrict_to_common_examples(df)
        attempts = attempts[attempts["_eval_id"].isin(set(df["_eval_id"]))].copy()
    args.out.mkdir(parents=True, exist_ok=True)

    status = summarize_status(df)
    errors = summarize_errors(attempts)
    scores = summarize_scores(df)
    agreement = pairwise_agreement(df)
    reference_model = eval_config["default_model"]
    bias = reference_bias(df, reference_model)
    disagreements = top_disagreements(df, reference_model)
    fleiss = fleiss_kappa(df)
    ops = operational_stats(df)
    shape = dataset_shape_stats(df)
    intra = intra_rater_reliability(attempts)

    df.to_csv(args.out / "judge_results_latest_flat.csv", index=False)
    attempts.to_csv(args.out / "judge_results_all_attempts_flat.csv", index=False)
    status.to_csv(args.out / "status_summary.csv", index=False)
    errors.to_csv(args.out / "error_summary.csv", index=False)
    scores.to_csv(args.out / "score_summary_by_model.csv", index=False)
    agreement.to_csv(args.out / "pairwise_agreement.csv", index=False)
    bias.to_csv(args.out / "reference_bias.csv", index=False)
    disagreements.to_csv(args.out / "top_disagreements.csv", index=False)
    fleiss.to_csv(args.out / "fleiss_kappa.csv", index=False)
    ops.to_csv(args.out / "operational_stats_by_model.csv", index=False)
    shape.to_csv(args.out / "dataset_shape_stats.csv", index=False)
    intra.to_csv(args.out / "intra_rater_reliability.csv", index=False)

    print("\nStatus summary")
    print(status.to_string(index=False) if not status.empty else "No status data.")
    print("\nError summary")
    print(errors.to_string(index=False) if not errors.empty else "No errors.")
    print("\nScore summary")
    print(scores.to_string(index=False) if not scores.empty else "No score data.")
    print("\nPairwise agreement")
    print(
        agreement.to_string(index=False)
        if not agreement.empty
        else "Need at least two models with overlapping ok records."
    )
    print("\nReference bias")
    print(bias.to_string(index=False) if not bias.empty else "No reference bias data.")
    print("\nFleiss kappa")
    print(
        fleiss.to_string(index=False)
        if not fleiss.empty
        else "Need complete overlapping ratings from at least two models."
    )
    print("\nOperational stats")
    print(ops.to_string(index=False) if not ops.empty else "No operational data.")
    print("\nDataset shape stats")
    print(shape.to_string(index=False) if not shape.empty else "No dataset shape data.")
    print("\nIntra-rater reliability")
    print(intra.to_string(index=False) if not intra.empty else "No repeated judgments.")
    print(f"\nSaved summaries to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
