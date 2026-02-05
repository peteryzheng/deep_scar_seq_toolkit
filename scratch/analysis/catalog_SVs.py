#%%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as pn

# import load_events from scratch.plotting.load_events
from scratch.plotting.load_events import load_events

# %%
input_dir = Path('../../data/dss_020226/')
files = sorted(input_dir.rglob('*chr2_1152626_w5*.tsv'))
# files = sorted(input_dir.rglob('*chr2_1153052_w5*.tsv'))
# %%
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
                    "mapping_quality1",
                    "mapping_quality2",
                    'largest_match_length1',
                    'largest_match_length2',
                    "largest_clip_length1",
                    "largest_clip_length2",
                    "is_supplementary1",
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
    # return events
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
            mapq1_mean=("mapping_quality1", "mean"),
            match1_mean=("largest_clip_length1", "mean"),
            mapq2_mean=("mapping_quality2", "mean"),
            match2_mean=("largest_clip_length2", "mean"),
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

# %%
events = load_events(files)
print(f"Loaded {len(events)} unique SV events from {len(files)} files.")


# %%
events

# %% 
# distribution of mapping qualities to each other
sns.jointplot(
    data=events,
    x="mapq1_mean",
    y="mapq2_mean",
    kind="hex",
    height=8,
    marginal_kws=dict(bins=50, fill=True),
)
plt.suptitle("Mapping Quality Side 1 vs. Side 2", y=1.02)
plt.show()

# %%
# %%
# does mapping quality and match length correlate?
sns.jointplot(
    data=events,
    x="mapping_quality1",
    y="largest_match_length1",
    kind="hex",
    height=8,
    marginal_kws=dict(bins=50, fill=True),
)
plt.suptitle("Mapping Quality vs. Largest Match Length (Side 1)", y=1.02)
plt.show()

sns.jointplot(
    data=events,
    x="mapping_quality2",
    y="largest_match_length2",
    kind="hex",
    height=8,
    marginal_kws=dict(bins=50, fill=True),
)
plt.suptitle("Mapping Quality vs. Largest Match Length (Side 2)", y=1.02)
plt.show()
# mapping quality has little to do with match length
# all alignments have match lengths >= 30bp
events[
    (events['largest_match_length1'] < 30) &
        (events['largest_match_length2'] < 30)
]
# so matching length is not a good filter here

# %%
# the question remains: how many SVs are there that have low mapping quality1 or 2?
# and more importantly, what is going on with those SVs? and can we trust them at all?
print(f"Number of SVs with mapping quality 1 < 20: {len(events[events['mapping_quality1'] < 20])}")
print(f"Number of SVs with mapping quality 2 < 20: {len(events[events['mapping_quality2'] < 20])}")
print(f"Number of SVs with mapping quality 1 < 20 or mapping quality 2 < 20: {len(events[(events['mapping_quality1'] < 20) | (events['mapping_quality2'] < 20)])}")

# all low mapq1 are low mapq2 as well
# there are 421 such events
print(events[events['mapping_quality1'] < 20]['is_supplementary1'].value_counts())
# and almost all low mapq1 have supplementary alignments at the target side

# how about the events that are low mapq2 but high mapq1?
events[(events['mapping_quality1'] >= 20) & (events['mapping_quality2'] < 20)]
# there are 803 such events
events[(events['mapping_quality1'] >= 20) & (events['mapping_quality2'] < 20)]['is_supplementary1'].value_counts()
# they all have supplementary alignments at the non-target side (side 2)
# is low mapq2 associated with low match length at side 2?
print(events[(events['mapping_quality1'] >= 20) & (events['mapping_quality2'] < 20)]['largest_match_length2'].describe())
# or is low mapq2 associated with lower mapq1?

# %%
