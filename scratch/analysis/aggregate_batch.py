#!/usr/bin/env python3
"""Batch-aggregate annotated SV TSVs in a directory (scratch use)."""

#%%
from pathlib import Path

import pandas as pd

from deep_scar_seq_toolkit.aggregate import READ_EVENT_COLUMNS, aggregate_reads


#%%
INPUT_DIR = Path("data/dss_020226")
OUTPUT_DIR = Path("data/dss_020226_events")
PATTERN = "*.annotated.tsv"

MIN_MAPQ = 5
MIN_MATCH_LENGTH = 30
MAX_EXTRA_BASES = 0


#%%
def parse_aliquot_and_sample(filename):
    """Parse aliquot and sample from a filename like NA05_H.*.annotated.tsv."""
    aliquot = filename.split(".")[0]
    if "_" in aliquot:
        sample = aliquot.split("_")[0]
    else:
        sample = aliquot
    return aliquot, sample


#%%
def aggregate_file(path, output_dir, min_mapq, min_match_length, max_extra_bases):
    events = pd.read_csv(
        path,
        sep="\t",
        usecols=lambda col: col in READ_EVENT_COLUMNS,
        dtype={
            "sv_type": "string",
            "mh_seq": "string",
            "ins_seq": "string",
            "mh_length": "string",
            "ins_length": "string",
            "umi": "string",
            "umi_source_tag": "string",
            "umi_missing": "string",
            "umi_mi": "string",
        },
    )
    aggregated = aggregate_reads(events, min_mapq, min_match_length, max_extra_bases)
    aliquot, sample = parse_aliquot_and_sample(path.name)
    aggregated["aliquot"] = aliquot
    aggregated["sample"] = sample
    out_name = path.name.replace(".annotated.tsv", ".aggregated.tsv")
    out_path = output_dir / out_name
    aggregated.to_csv(out_path, sep="\t", index=False)
    return out_path, len(aggregated)


#%%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paths = sorted(INPUT_DIR.rglob(PATTERN))
if not paths:
    raise SystemExit(f"No files found in {INPUT_DIR} matching {PATTERN}")

total = 0
for path in paths:
    out_path, count = aggregate_file(
        path,
        OUTPUT_DIR,
        MIN_MAPQ,
        MIN_MATCH_LENGTH,
        MAX_EXTRA_BASES,
    )
    total += 1
    print(f"Wrote {count} aggregated events to {out_path}")

print(f"Done. Aggregated {total} files into {OUTPUT_DIR}")

# %%
