#!/usr/bin/env python3
"""Plot SV distributions for a batch of annotated TSVs (batch-specific)."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_aliquot_and_sample(filename):
    """Parse aliquot and sample from a filename like NA05_H.*.annotated.tsv."""
    aliquot = filename.split(".")[0]
    if "_" in aliquot:
        sample = aliquot.split("_")[0]
    else:
        sample = aliquot
    return aliquot, sample


def load_events(files):
    records = []
    for path in files:
        aliquot, sample = parse_aliquot_and_sample(path.name)
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=[
                    "read_name",
                    "chr1",
                    "breakpoint1",
                    "chr2",
                    "breakpoint2",
                    "clip_side1",
                    "clip_side2",
                    "sv_type",
                    "mh_seq",
                    "ins_seq",
                    "mh_length",
                    "ins_length",
                ],
                dtype={
                    "sv_type": "string",
                    "mh_seq": "string",
                    "ins_seq": "string",
                    "mh_length": "string",
                    "ins_length": "string",
                },
            )
        except ValueError as exc:
            raise ValueError(f"Missing expected columns in {path}: {exc}")
        df["aliquot"] = aliquot
        df["sample"] = sample
        records.append(df)
    if not records:
        raise ValueError("No annotated TSV files found. Check input directory/pattern.")

    events = pd.concat(records, ignore_index=True)
    events["sv_type"] = events["sv_type"].fillna("NA")
    events["mh_seq"] = events["mh_seq"].fillna("")
    events["ins_seq"] = events["ins_seq"].fillna("")
    events["mh_length"] = pd.to_numeric(events["mh_length"], errors="coerce").fillna(0).astype(int)
    events["ins_length"] = pd.to_numeric(events["ins_length"], errors="coerce").fillna(0).astype(int)

    unique_events = (
        events.fillna("")
        .groupby(
            [
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
        )
        .agg(
            read_count=("read_name", "count"),
            frag_count=("read_name", lambda x: x.nunique()),
            mh_length=("mh_length", "first"),
            ins_length=("ins_length", "first"),
        )
        .reset_index()
    )
    return unique_events


def order_aliquots(events):
    order = (
        events[["sample", "aliquot"]]
        .drop_duplicates()
        .sort_values(["sample", "aliquot"])
    )
    return order


def add_sample_separators(ax, order, axis="x", text_offset=1.02):
    sample_positions = {}
    for idx, row in enumerate(order.itertuples(index=False)):
        sample_positions.setdefault(row.sample, []).append(idx)

    for sample, positions in sample_positions.items():
        start = positions[0]
        end = positions[-1]
        mid = (start + end) / 2
        if axis == "x":
            ax.axvline(end + 0.5, color="black", linewidth=0.5)
            ax.text(
                mid,
                text_offset,
                sample,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.axhline(end + 0.5, color="black", linewidth=0.5)
            ax.text(
                text_offset,
                mid,
                sample,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )


def plot_sv_type_distribution(events, order, outpath):
    counts = events.groupby(["aliquot", "sv_type"]).size().reset_index(name="count")
    totals = counts.groupby("aliquot")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals

    pivot = counts.pivot(index="aliquot", columns="sv_type", values="fraction").fillna(0)
    pivot = pivot.loc[order["aliquot"]]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.5), 5))
    bottom = np.zeros(len(pivot))
    for sv_type in pivot.columns:
        ax.bar(pivot.index, pivot[sv_type], bottom=bottom, label=sv_type)
        bottom += pivot[sv_type].values

    ax.set_ylabel("Fraction of SVs")
    ax.set_xlabel("Aliquot (grouped by sample)")
    ax.set_title("SV Type Distribution by Aliquot", pad=30)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    add_sample_separators(ax, order, text_offset=1.01)
    ax.legend(title="SV Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_sv_type_counts(events, order, outpath):
    counts = events.groupby(["aliquot", "sv_type"]).size().reset_index(name="count")
    pivot = counts.pivot(index="aliquot", columns="sv_type", values="count").fillna(0)
    pivot = pivot.loc[order["aliquot"]]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.5), 5))
    bottom = np.zeros(len(pivot))
    for sv_type in pivot.columns:
        ax.bar(pivot.index, pivot[sv_type], bottom=bottom, label=sv_type)
        bottom += pivot[sv_type].values

    ax.set_ylabel("SV Count")
    ax.set_xlabel("Aliquot (grouped by sample)")
    ax.set_title("SV Type Counts by Aliquot", pad=30)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    add_sample_separators(ax, order, text_offset=1.01)
    ax.legend(title="SV Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_length_frequency(events, order, value_col, title, outpath, x_max=None):
    sample_to_aliquots = order.groupby("sample")["aliquot"].apply(list).to_dict()
    samples = list(sample_to_aliquots.keys())

    nrows = 2
    ncols = int(np.ceil(len(samples) / nrows)) if samples else 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(8, ncols * 4.5), max(4.5, nrows * 2.6)),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, sample in zip(axes, samples):
        aliquots = sample_to_aliquots[sample]
        subset = events[events["aliquot"].isin(aliquots)]
        if subset.empty:
            continue
        max_len = int(subset[value_col].max())
        if x_max is not None:
            max_len = min(max_len, x_max)
        x_vals = np.arange(0, max_len + 1)
        for aliquot in aliquots:
            data = subset[subset["aliquot"] == aliquot][value_col]
            if len(data) == 0:
                continue
            counts = data.value_counts().sort_index()
            counts = counts.reindex(x_vals, fill_value=0)
            ax.plot(x_vals, counts.values, label=aliquot, linewidth=1.5)
        ax.set_ylabel("Count")
        ax.set_title(sample, loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[len(samples) :]:
        ax.axis("off")

    for ax in axes:
        ax.set_xlabel("Length (bp)")
    if x_max is not None:
        for ax in axes[: len(samples)]:
            ax.set_xlim(0, x_max)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Aliquot", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot SV distributions from annotated TSVs.")
    parser.add_argument(
        "--input-dir", default="data", help="Directory to search for annotated TSVs"
    )
    parser.add_argument(
        "--pattern", default="*.annotated.tsv", help="Glob pattern for annotated TSVs"
    )
    parser.add_argument(
        "--output-dir", default="sv_plots", help="Output directory for plots and tables"
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.rglob(args.pattern))
    if not files:
        print(f"No files found under {input_dir} matching {args.pattern}", file=sys.stderr)
        sys.exit(1)

    events = load_events(files)
    order = order_aliquots(events)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sv_counts = events.groupby(["sample", "aliquot", "sv_type"]).size().reset_index(name="count")
    sv_counts.to_csv(output_dir / "sv_type_counts.tsv", sep="\t", index=False)

    mh_counts = events.groupby(["sample", "aliquot", "mh_length"]).size().reset_index(name="count")
    mh_counts.to_csv(output_dir / "mh_length_counts.tsv", sep="\t", index=False)

    ins_counts = events.groupby(["sample", "aliquot", "ins_length"]).size().reset_index(name="count")
    ins_counts.to_csv(output_dir / "ins_length_counts.tsv", sep="\t", index=False)

    order.to_csv(output_dir / "sample_aliquot_map.tsv", sep="\t", index=False)

    plot_sv_type_distribution(events, order, output_dir / "sv_type_distribution_by_aliquot.png")
    plot_sv_type_counts(events, order, output_dir / "sv_type_counts_by_aliquot.png")

    plot_length_frequency(
        events,
        order,
        value_col="mh_length",
        title="Microhomology Length Frequency by Aliquot",
        outpath=output_dir / "mh_length_frequency_by_aliquot.png",
        x_max=20,
    )

    plot_length_frequency(
        events,
        order,
        value_col="ins_length",
        title="Insertion Length Frequency by Aliquot",
        outpath=output_dir / "insertion_length_frequency_by_aliquot.png",
        x_max=150,
    )

    print(f"Wrote plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
