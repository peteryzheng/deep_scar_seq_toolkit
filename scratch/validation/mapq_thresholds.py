#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "read_name",
    "chr1",
    "breakpoint1",
    "chr2",
    "breakpoint2",
    "clip_side1",
    "clip_side2",
    "sv_type",
    "mapping_quality1",
    "mapping_quality2",
    "largest_match_length1",
    "largest_match_length2",
]

# Columns that define a unique event (per-sample uniqueness).
DEFAULT_EVENT_COLUMNS = [
    "sample",
    "aliquot",
    "chr1",
    "breakpoint1",
    "chr2",
    "breakpoint2",
    "clip_side1",
    "clip_side2",
    "sv_type",
    "mh_seq",
    "ins_seq",
]


@dataclass(frozen=True)
class RowThresholdResult:
    threshold: int
    total_events: int
    pass_both: int
    pass_side1: int
    pass_side2: int


@dataclass(frozen=True)
class UniqueEventThresholdResult:
    threshold: int
    total_unique_events: int
    pass_unique_events: int


@dataclass(frozen=True)
class FailedRowsBreakdown:
    threshold: int
    failed_rows_total: int
    failed_rows_event_retained: int
    failed_rows_event_lost: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MAPQ distributions and threshold impacts for annotated SV events."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a TSV file or a glob (e.g., data/*.annotated.tsv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write plots (histograms + CDF).",
    )
    parser.add_argument(
        "--min-match-length",
        type=int,
        default=None,
        help="If set, only include events with largest_match_length1/2 >= this value.",
    )
    parser.add_argument(
        "--thresholds",
        default="0,5,10,20,30,40,50,60",
        help="Comma-separated MAPQ thresholds to evaluate.",
    )
    return parser.parse_args()


def parse_aliquot_and_sample(filename: str) -> tuple[str, str]:
    """Parse aliquot and sample from a filename like NA05_H.*.annotated.tsv."""
    aliquot = filename.split(".")[0]
    if "_" in aliquot:
        sample = aliquot.split("_")[0]
    else:
        sample = aliquot
    return aliquot, sample


def load_events(paths: list[Path]) -> pd.DataFrame:
    records = []
    for path in paths:
        df = pd.read_csv(path, sep="\t", dtype={"sv_type": "string"})
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        aliquot, sample = parse_aliquot_and_sample(path.name)
        if "aliquot" not in df.columns:
            df["aliquot"] = aliquot
        else:
            df["aliquot"] = df["aliquot"].fillna(aliquot)
        if "sample" not in df.columns:
            df["sample"] = sample
        else:
            df["sample"] = df["sample"].fillna(sample)
        if "mh_seq" not in df.columns or "ins_seq" not in df.columns:
            raise ValueError(f"{path} missing mh_seq/ins_seq columns for event key.")
        df["mh_seq"] = df["mh_seq"].fillna("")
        df["ins_seq"] = df["ins_seq"].fillna("")
        df["source"] = path.name
        records.append(df)
    if not records:
        raise ValueError("No input files found to load.")
    return pd.concat(records, ignore_index=True)


def resolve_paths(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.exists():
        if path.is_dir():
            return sorted(path.glob("*.tsv"))
        return [path]
    return sorted(Path().glob(pattern))


def summarize_mapq(events: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    summary = {
        "metric": ["mapping_quality1", "mapping_quality2"],
        "mean": [
            events["mapping_quality1"].mean(),
            events["mapping_quality2"].mean(),
        ],
        "std": [
            events["mapping_quality1"].std(),
            events["mapping_quality2"].std(),
        ],
    }
    for q in quantiles:
        label = f"q{int(q * 100)}"
        summary[label] = [
            events["mapping_quality1"].quantile(q),
            events["mapping_quality2"].quantile(q),
        ]
    return pd.DataFrame(summary)


def attach_event_key(events: pd.DataFrame) -> pd.DataFrame:
    event_columns = [col for col in DEFAULT_EVENT_COLUMNS if col in events.columns]
    missing = [col for col in DEFAULT_EVENT_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(f"Missing event columns for grouping: {missing}")
    event_key = events[event_columns].astype(str).agg("|".join, axis=1)
    events = events.copy()
    events["event_key"] = event_key
    return events


def evaluate_row_thresholds(events: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    total = len(events)
    results: list[RowThresholdResult] = []
    for threshold in thresholds:
        pass_side1_mask = events["mapping_quality1"] >= threshold
        pass_side2_mask = events["mapping_quality2"] >= threshold
        pass_both_mask = pass_side1_mask & pass_side2_mask

        results.append(
            RowThresholdResult(
                threshold=threshold,
                total_events=total,
                pass_both=int(pass_both_mask.sum()),
                pass_side1=int(pass_side1_mask.sum()),
                pass_side2=int(pass_side2_mask.sum()),
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    df["pass_both_frac"] = df["pass_both"] / df["total_events"]
    df["pass_side1_frac"] = df["pass_side1"] / df["total_events"]
    df["pass_side2_frac"] = df["pass_side2"] / df["total_events"]
    return df


def evaluate_unique_event_thresholds(events: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    total_unique_events = events["event_key"].nunique()
    results: list[UniqueEventThresholdResult] = []
    for threshold in thresholds:
        pass_both_mask = (events["mapping_quality1"] >= threshold) & (
            events["mapping_quality2"] >= threshold
        )
        pass_unique_events = events.loc[pass_both_mask, "event_key"].nunique()
        results.append(
            UniqueEventThresholdResult(
                threshold=threshold,
                total_unique_events=total_unique_events,
                pass_unique_events=pass_unique_events,
            )
        )
    df = pd.DataFrame([r.__dict__ for r in results])
    df["pass_unique_events_frac"] = df["pass_unique_events"] / df["total_unique_events"]
    return df


def evaluate_failed_rows_breakdown(events: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    results: list[FailedRowsBreakdown] = []
    for threshold in thresholds:
        pass_both_mask = (events["mapping_quality1"] >= threshold) & (
            events["mapping_quality2"] >= threshold
        )
        failed_mask = ~pass_both_mask

        pass_events = events.loc[pass_both_mask, "event_key"].unique()
        failed_events = events.loc[failed_mask, "event_key"].unique()

        failed_events_event_retained = np.intersect1d(failed_events, pass_events)
        failed_events_event_lost = np.setdiff1d(failed_events, pass_events)

        failed_rows_total = int(failed_mask.sum())
        failed_rows_event_retained = int(
            events.loc[failed_mask & events["event_key"].isin(failed_events_event_retained)].shape[0]
        )
        failed_rows_event_lost = int(
            events.loc[failed_mask & events["event_key"].isin(failed_events_event_lost)].shape[0]
        )

        results.append(
            FailedRowsBreakdown(
                threshold=threshold,
                failed_rows_total=failed_rows_total,
                failed_rows_event_retained=failed_rows_event_retained,
                failed_rows_event_lost=failed_rows_event_lost,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    df["failed_rows_event_retained_frac"] = df["failed_rows_event_retained"] / df["failed_rows_total"].replace(0, np.nan)
    df["failed_rows_event_lost_frac"] = df["failed_rows_event_lost"] / df["failed_rows_total"].replace(0, np.nan)
    return df


def plot_distributions(events: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(events["mapping_quality1"], bins=50, color="#2A6F9B", alpha=0.9)
    axes[0].set_title("MAPQ distribution (side 1)")
    axes[0].set_xlabel("MAPQ")
    axes[0].set_ylabel("Count")
    axes[1].hist(events["mapping_quality2"], bins=50, color="#F28E2B", alpha=0.9)
    axes[1].set_title("MAPQ distribution (side 2)")
    axes[1].set_xlabel("MAPQ")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "mapq_histograms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, col in [("side1", "mapping_quality1"), ("side2", "mapping_quality2")]:
        values = np.sort(events[col].to_numpy())
        cdf = np.linspace(0, 1, len(values), endpoint=False)
        ax.plot(values, cdf, label=label)
    ax.set_title("MAPQ empirical CDF")
    ax.set_xlabel("MAPQ")
    ax.set_ylabel("Fraction ≤ MAPQ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mapq_cdf.png", dpi=150)
    plt.close(fig)


def plot_threshold_summaries(
    row_table: pd.DataFrame,
    unique_table: pd.DataFrame,
    failed_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    x_labels = row_table["threshold"].astype(str).tolist()
    x_positions = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        x_positions,
        row_table["pass_both_frac"],
        marker="o",
        color="#2A6F9B",
        linewidth=2,
    )
    ax.set_title("Fraction of reads passing MAPQ filter")
    ax.set_xlabel("MAPQ threshold (both sides)")
    ax.set_ylabel("Reads passing filter (fraction)")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions, x_labels)
    fig.tight_layout()
    fig.savefig(output_dir / "mapq_pass_fraction_reads.png")
    plt.close(fig)

    unique_x_labels = unique_table["threshold"].astype(str).tolist()
    unique_x_positions = np.arange(len(unique_x_labels))
    retained = unique_table["pass_unique_events_frac"]
    lost = 1 - retained

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        unique_x_positions,
        retained,
        color="#4E79A7",
        label="Retained",
    )
    ax.bar(
        unique_x_positions,
        lost,
        bottom=retained,
        color="#E15759",
        label="Lost",
    )
    ax.set_title("Unique events retained after MAPQ filter")
    ax.set_xlabel("MAPQ threshold (both sides)")
    ax.set_ylabel("Fraction of unique events")
    ax.set_ylim(0, 1)
    ax.set_xticks(unique_x_positions, unique_x_labels)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mapq_pass_fraction_events.png")
    plt.close(fig)

    failed_table = failed_table.set_index("threshold").reindex(row_table["threshold"]).fillna(0).reset_index()
    failed_x_labels = failed_table["threshold"].astype(str).tolist()
    failed_x_positions = np.arange(len(failed_x_labels))

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(
        failed_x_positions,
        failed_table["failed_rows_event_retained_frac"],
        color="#4E79A7",
        label="Retained",
    )
    ax.bar(
        failed_x_positions,
        failed_table["failed_rows_event_lost_frac"],
        bottom=failed_table["failed_rows_event_retained_frac"],
        color="#E15759",
        label="Lost",
    )
    ax.set_title("Which Events Do Filtered-Out Reads Come From?")
    ax.set_xlabel("MAPQ threshold (both sides)")
    ax.set_ylabel("Fraction of filtered-out reads")
    ax.set_ylim(0, 1)
    ax.set_xticks(failed_x_positions, failed_x_labels)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mapq_failed_rows_composition.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.input)
    if not paths:
        raise SystemExit("No input files found for --input.")
    events = load_events(paths)

    if args.min_match_length is not None:
        events = events[
            (events["largest_match_length1"] >= args.min_match_length)
            & (events["largest_match_length2"] >= args.min_match_length)
        ]

    thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    print("MAPQ summary statistics:")
    print(summarize_mapq(events).to_string(index=False))

    events = attach_event_key(events)

    print("\nRow-level threshold impact (counts + fractions):")
    row_table = evaluate_row_thresholds(events, thresholds)
    row_table = row_table.rename(columns={"total_events": "total_reads"})
    print(row_table.to_string(index=False))

    print("\nUnique-event threshold impact (counts + fractions):")
    unique_table = evaluate_unique_event_thresholds(events, thresholds)
    print(unique_table.to_string(index=False))

    print("\nFailed-rows breakdown (rows + events):")
    failed_table = evaluate_failed_rows_breakdown(events, thresholds)
    print(failed_table.to_string(index=False))

    if args.output_dir:
        plot_distributions(events, Path(args.output_dir))
        plot_threshold_summaries(row_table, unique_table, failed_table, Path(args.output_dir))
        print(f"\nPlots written to {args.output_dir}")


if __name__ == "__main__":
    main()
