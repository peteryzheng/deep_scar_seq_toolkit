#!/usr/bin/env python3
"""Assess unique SV discovery as aliquots are cumulatively included."""

#%%
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from deep_scar_seq_toolkit.aggregate import EXPECTED_EVENT_COLUMNS, aggregate_reads

INPUT_DIR = Path("data/dss_020226")
PATTERN = "*.annotated.tsv"
OUT_PATH = Path("scratch/analysis/aliquot_depth_summary.tsv")

MIN_MAPQ = 5
MIN_MATCH_LENGTH = 30
MAX_EXTRA_BASES = 0


def parse_filename(path: Path) -> dict[str, str]:
    """Parse sample, aliquot, and cut site from a filename."""
    stem = path.name.replace(".annotated.tsv", "")
    aliquot = stem.split(".")[0]
    sample = aliquot.split("_")[0]
    cut_site = next((seg for seg in stem.split(".") if seg.startswith("chr")), "")
    cut_site = cut_site.split("_w")[0] if cut_site else ""
    if not cut_site:
        raise ValueError(f"Could not parse cut site from {path.name}")
    return {"sample": sample, "aliquot": aliquot, "cut_site": cut_site}


def read_events(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        meta = parse_filename(path)
        events = pd.read_csv(
            path,
            sep="\t",
            usecols=EXPECTED_EVENT_COLUMNS,
            dtype={
                "sv_type": "string",
                "mh_seq": "string",
                "ins_seq": "string",
                "mh_length": "string",
                "ins_length": "string",
            },
        )
        events["sample"] = meta["sample"]
        events["aliquot"] = meta["aliquot"]
        events["cut_site"] = meta["cut_site"]
        rows.append(events)
    if not rows:
        raise SystemExit(f"No files found in {INPUT_DIR} matching {PATTERN}")
    return pd.concat(rows, ignore_index=True)

#%%
def main() -> None:
    paths = sorted(INPUT_DIR.rglob(PATTERN))
    events = read_events(paths)

    summary_rows: list[dict[str, object]] = []
    for (sample, cut_site), group in events.groupby(["sample", "cut_site"]):
        aliquots = sorted(group["aliquot"].unique())
        prev_count = None
        for k in range(1, len(aliquots) + 1):
            included = aliquots[:k]
            subset = group[group["aliquot"].isin(included)].copy()

            aggregated = aggregate_reads(
                subset, MIN_MAPQ, MIN_MATCH_LENGTH, MAX_EXTRA_BASES
            )
            unique_svs = len(aggregated)
            new_svs = None if prev_count is None else unique_svs - prev_count
            summary_rows.append(
                {
                    "sample": sample,
                    "cut_site": cut_site,
                    "depth_aliquots": k,
                    "unique_svs": unique_svs,
                    "new_svs": new_svs,
                    "aliquots_included": ",".join(included),
                }
            )
            prev_count = unique_svs

    summary = pd.DataFrame(summary_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {len(summary)} rows to {OUT_PATH}")

    for cut_site, group in summary.groupby("cut_site"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for sample, sample_group in group.groupby("sample"):
            ax.plot(
                sample_group["depth_aliquots"],
                sample_group["unique_svs"],
                marker="o",
                label=sample,
            )
        ax.set_xlabel("Aliquots included (depth proxy)")
        ax.set_ylabel("Unique SVs")
        ax.set_title(f"{cut_site}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sample", fontsize=8, title_fontsize=9)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

# %%
