from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ppm.data_preparation.data_preparation import (
    BENCHMARK_PARAMS,
    EVENT_LOGS,
    filter_cases_by_duration,
    filter_cases_ending_before,
    filter_cases_starting_from,
)


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute trace variants and their occurrence counts from filtered event logs."
    )
    parser.add_argument("--dataset", type=str, default="BPI12")
    parser.add_argument("--lifecycle", action="store_true", default=False)
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of most frequent variants to print. Use 0 to print all.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV path to save all variants and counts.",
    )
    return parser.parse_args()


def build_df_filtered(dataset: str, lifecycle: bool) -> pd.DataFrame:
    if dataset not in EVENT_LOGS:
        valid = ", ".join(sorted(EVENT_LOGS.keys()))
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of: {valid}")

    log = EVENT_LOGS[dataset]()
    df_orig = log.dataframe.copy()

    if lifecycle:
        if "lifecycle:transition" in df_orig.columns:
            df_orig[ACTIVITY_COL] = (
                df_orig[ACTIVITY_COL] + "_" + df_orig["lifecycle:transition"]
            )
        else:
            print("Column 'lifecycle:transition' not found. Disabling lifecycle.")

    cols_to_keep = [CASE_COL, ACTIVITY_COL, TIME_COL]
    if "org:resource" in df_orig.columns:
        cols_to_keep.append("org:resource")
    if "case:AMOUNT_REQ" in df_orig.columns:
        cols_to_keep.append("case:AMOUNT_REQ")

    df_orig = df_orig.loc[:, cols_to_keep].copy()
    df_orig[TIME_COL] = pd.to_datetime(df_orig[TIME_COL], utc=True)

    cases_to_keep = df_orig.groupby(CASE_COL).size() > 2
    cases_to_keep = cases_to_keep[cases_to_keep].index
    df_orig = df_orig[df_orig[CASE_COL].isin(cases_to_keep)]
    df_orig = df_orig.sort_values(by=[CASE_COL, TIME_COL]).reset_index(drop=True)

    params = BENCHMARK_PARAMS.get(dataset, {})
    df_filtered = filter_cases_starting_from(df_orig, params.get("start_date"))
    df_filtered = filter_cases_ending_before(df_filtered, params.get("end_date"))
    df_filtered = df_filtered.drop_duplicates()

    if params.get("max_days"):
        df_filtered, _ = filter_cases_by_duration(df_filtered, params["max_days"])

    return df_filtered.sort_values(by=[CASE_COL, TIME_COL]).reset_index(drop=True)


def compute_variant_counts(df_filtered: pd.DataFrame) -> pd.DataFrame:
    case_variants = (
        df_filtered.groupby(CASE_COL, sort=False)[ACTIVITY_COL].agg(tuple).rename("variant")
    )
    variant_counts = (
        case_variants.value_counts()
        .rename_axis("variant")
        .reset_index(name="count")
        .sort_values(by=["count", "variant"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    variant_counts["variant_id"] = variant_counts.index + 1
    variant_counts["trace_length"] = variant_counts["variant"].apply(len)
    variant_counts["variant_str"] = variant_counts["variant"].apply(" > ".join)
    return variant_counts.loc[:, ["variant_id", "count", "trace_length", "variant", "variant_str"]]


def top_share(variant_counts: pd.DataFrame, top_n: int, total_cases: int) -> float:
    if top_n <= 0 or total_cases == 0:
        return 0.0
    return float(variant_counts["count"].head(top_n).sum()) / float(total_cases)


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


def main() -> None:
    args = parse_args()
    df_filtered = build_df_filtered(dataset=args.dataset, lifecycle=args.lifecycle)
    # Always report a lifecycle-aware class count for comparability across settings.
    df_filtered_lifecycle = build_df_filtered(dataset=args.dataset, lifecycle=True)
    variant_counts = compute_variant_counts(df_filtered)

    n_cases = df_filtered[CASE_COL].nunique()
    n_variants = len(variant_counts)
    n_activities_current = df_filtered[ACTIVITY_COL].nunique()
    n_activities_lifecycle = df_filtered_lifecycle[ACTIVITY_COL].nunique()
    n_classes_with_eos_current = n_activities_current + 1
    n_classes_with_eos_lifecycle = n_activities_lifecycle + 1

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
    print(f"Unique variants: {n_variants}")
    print(f"Share top 5 variants: {top_share(variant_counts, 5, n_cases):.4f}")
    print(f"Share top 10 variants: {top_share(variant_counts, 10, n_cases):.4f}")
    print(f"Gini coefficient (variant counts): {gini_from_counts(variant_counts['count']):.4f}")
    print("=" * 80)

    to_show = variant_counts if args.top_k == 0 else variant_counts.head(args.top_k)
    print(f"Top variants (top_n={len(to_show)}):")
    for row in to_show.itertuples(index=False):
        print(
            f"{row.variant_id:>4}. count={row.count:<6} len={row.trace_length:<3}  {row.variant_str}"
        )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = variant_counts.copy()
        to_save["variant"] = to_save["variant"].apply(list)
        to_save.to_csv(out_path, index=False)
        print("=" * 80)
        print(f"Saved {len(to_save)} variants to: {out_path}")


if __name__ == "__main__":
    main()
