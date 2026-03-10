from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main_variants import ACTIVITY_COL, CASE_COL, build_df_filtered


EOS_TOKEN = "<EOS>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute prefix statistics and next-activity uncertainty from filtered event logs."
    )
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--lifecycle", action="store_true", default=False)
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of most frequent prefixes to print. Use 0 to print all.",
    )
    parser.add_argument(
        "--max_prefix_len",
        type=int,
        default=None,
        help="Optional cap for prefix length. By default uses full trace length.",
    )
    parser.add_argument(
        "--output_prefix_csv",
        type=str,
        default=None,
        help="Optional CSV path to save per-prefix statistics.",
    )
    parser.add_argument(
        "--output_nextdist_csv",
        type=str,
        default=None,
        help="Optional CSV path to save per-prefix next-activity distributions.",
    )
    return parser.parse_args()


def gini_from_counts(counts: pd.Series) -> float:
    x = np.sort(counts.to_numpy(dtype=np.float64))
    n = x.size
    if n == 0:
        return 0.0
    total = x.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x)) / (n * total) - (n + 1.0) / n)


def top_share(counts: pd.Series, top_n: int) -> float:
    total = int(counts.sum())
    if top_n <= 0 or total == 0:
        return 0.0
    return float(counts.head(top_n).sum()) / float(total)


def extract_prefix_next_pairs(
    df_filtered: pd.DataFrame, max_prefix_len: int | None = None
) -> pd.DataFrame:
    rows: list[tuple[tuple[str, ...], str]] = []

    for _, group in df_filtered.groupby(CASE_COL, sort=False):
        seq = group[ACTIVITY_COL].tolist()
        n = len(seq)
        if n == 0:
            continue

        upper = n if max_prefix_len is None else min(n, max_prefix_len)
        for i in range(1, upper + 1):
            prefix = tuple(seq[:i])
            next_activity = seq[i] if i < n else EOS_TOKEN
            rows.append((prefix, next_activity))

    return pd.DataFrame(rows, columns=["prefix", "next_activity"])


def compute_prefix_outputs(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    prefix_counts = (
        pairs["prefix"]
        .value_counts()
        .rename_axis("prefix")
        .reset_index(name="count")
        .sort_values(by=["count", "prefix"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    prefix_counts["prefix_id"] = prefix_counts.index + 1
    prefix_counts["prefix_length"] = prefix_counts["prefix"].apply(len)
    prefix_counts["prefix_str"] = prefix_counts["prefix"].apply(" > ".join)

    prefix_next_counts = (
        pairs.groupby(["prefix", "next_activity"], sort=False)
        .size()
        .rename("next_count")
        .reset_index()
    )

    totals = (
        prefix_next_counts.groupby("prefix", sort=False)["next_count"]
        .sum()
        .rename("count")
        .reset_index()
    )
    merged = prefix_next_counts.merge(totals, on="prefix", how="left")
    merged["next_prob"] = merged["next_count"] / merged["count"]
    merged["entropy_term"] = -merged["next_prob"] * np.log2(merged["next_prob"])

    entropy_per_prefix = (
        merged.groupby("prefix", sort=False)
        .agg(
            next_entropy_bits=("entropy_term", "sum"),
            next_options=("next_activity", "nunique"),
        )
        .reset_index()
    )

    prefix_stats = prefix_counts.merge(entropy_per_prefix, on="prefix", how="left")
    prefix_stats["next_entropy_bits"] = prefix_stats["next_entropy_bits"].fillna(0.0)
    prefix_stats["next_options"] = (
        prefix_stats["next_options"].fillna(0).astype(int)
    )
    prefix_stats = prefix_stats.loc[
        :,
        [
            "prefix_id",
            "count",
            "prefix_length",
            "next_options",
            "next_entropy_bits",
            "prefix",
            "prefix_str",
        ],
    ]

    prefix_to_id = prefix_stats.set_index("prefix")["prefix_id"]
    next_distribution = merged.copy()
    next_distribution["prefix_id"] = next_distribution["prefix"].map(prefix_to_id)
    next_distribution["prefix_str"] = next_distribution["prefix"].apply(" > ".join)
    next_distribution = next_distribution.loc[
        :,
        ["prefix_id", "prefix", "prefix_str", "next_activity", "next_count", "next_prob"],
    ].sort_values(by=["prefix_id", "next_count", "next_activity"], ascending=[True, False, True])

    total_instances = int(prefix_stats["count"].sum())
    weighted_cond_entropy = (
        float((prefix_stats["count"] * prefix_stats["next_entropy_bits"]).sum())
        / float(total_instances)
        if total_instances > 0
        else 0.0
    )

    overall_next_probs = pairs["next_activity"].value_counts(normalize=True)
    overall_next_entropy = float(
        -(overall_next_probs * np.log2(overall_next_probs)).sum()
    )

    return prefix_stats, next_distribution, weighted_cond_entropy, overall_next_entropy


def maybe_save_csv(df: pd.DataFrame, out_path_str: str | None, tuple_col: str | None = None) -> None:
    if not out_path_str:
        return
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = df.copy()
    if tuple_col and tuple_col in to_save.columns:
        to_save[tuple_col] = to_save[tuple_col].apply(list)
    to_save.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(to_save)} rows)")


def main() -> None:
    args = parse_args()

    df_filtered = build_df_filtered(dataset=args.dataset, lifecycle=args.lifecycle)
    df_filtered_lifecycle = build_df_filtered(dataset=args.dataset, lifecycle=True)

    pairs = extract_prefix_next_pairs(df_filtered, max_prefix_len=args.max_prefix_len)
    prefix_stats, next_distribution, weighted_cond_entropy, overall_next_entropy = (
        compute_prefix_outputs(pairs)
    )

    n_cases = df_filtered[CASE_COL].nunique()
    n_activities_current = df_filtered[ACTIVITY_COL].nunique()
    n_activities_lifecycle = df_filtered_lifecycle[ACTIVITY_COL].nunique()
    n_classes_with_eos_current = n_activities_current + 1
    n_classes_with_eos_lifecycle = n_activities_lifecycle + 1

    n_prefix_instances = int(prefix_stats["count"].sum())
    n_unique_prefixes = len(prefix_stats)

    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Cases after filtering: {n_cases}")
    print(f"Possible activities (classes, current setting): {n_activities_current}")
    print(
        f"Possible next-activity classes incl. EOS (current setting): {n_classes_with_eos_current}"
    )
    print(f"Possible activities (classes, lifecycle-aware): {n_activities_lifecycle}")
    print(
        f"Possible next-activity classes incl. EOS (lifecycle-aware): {n_classes_with_eos_lifecycle}"
    )
    print(f"Prefix instances (including EOS-target instances): {n_prefix_instances}")
    print(f"Unique prefixes: {n_unique_prefixes}")
    print(f"Share top 5 prefixes: {top_share(prefix_stats['count'], 5):.4f}")
    print(f"Share top 10 prefixes: {top_share(prefix_stats['count'], 10):.4f}")
    print(f"Gini coefficient (prefix counts): {gini_from_counts(prefix_stats['count']):.4f}")
    print(
        f"Weighted conditional entropy H(next|prefix) [bits]: {weighted_cond_entropy:.4f}"
    )
    print(f"Marginal next entropy H(next) [bits]: {overall_next_entropy:.4f}")
    print("=" * 80)

    to_show = prefix_stats if args.top_k == 0 else prefix_stats.head(args.top_k)
    print(f"Top prefixes (top_n={len(to_show)}):")
    for row in to_show.itertuples(index=False):
        print(
            f"{row.prefix_id:>4}. count={row.count:<6} len={row.prefix_length:<3} "
            f"next_opts={row.next_options:<3} H={row.next_entropy_bits:.3f}  {row.prefix_str}"
        )

    maybe_save_csv(prefix_stats, args.output_prefix_csv, tuple_col="prefix")
    maybe_save_csv(next_distribution, args.output_nextdist_csv, tuple_col="prefix")


if __name__ == "__main__":
    main()
