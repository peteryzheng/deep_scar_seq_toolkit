#%%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
input_dir = Path('../../data/dss_020226_events/')
files = sorted(input_dir.rglob('*chr2_1152626_w5*.aggregated.tsv'))
# files = sorted(input_dir.rglob('*chr2_1153052_w5*.aggregated.tsv'))
# %%
def load_events(files):
    records = []
    for path in files:
        records.append(pd.read_csv(path, sep="\t"))
    if not records:
        raise ValueError("No aggregated TSV files found. Check input directory/pattern.")

    return pd.concat(records, ignore_index=True)


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

#%%
# what is the average fragment count per event?
# plot count bar plot of fragment counts of events, with x axis being fragment count
# have every aliquot in a separate facet
plt.figure(figsize=(10, 6))
sns.histplot(
    data=events,
    x="frag_count",
    bins=30,
    kde=False,
)
plt.yscale("log")
plt.xlabel("Fragment Count per SV Event")
plt.ylabel("Number of SV Events (log scale)")
plt.title("Distribution of Fragment Counts per SV Event")
plt.tight_layout()
plt.show()

# %%
# violin plot of fragment counts by aliquot, with x axis being aliquot and y axis being fragment count
# violin plot with normal y axis and no variance estimation, just the violins
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=events,
    x="aliquot",
    y="frag_count",
    color="lightgray",
    showfliers=False,
)
# y axis normal scale
plt.yscale("linear")
plt.xlabel("Aliquot")
plt.ylabel("Fragment Count per SV Event")
plt.title("Fragment Count Distribution by Aliquot")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#%%
# is there other interesting visualization we can do with these events? I am interested in seeing if any aliquots have weird frag count patterns
# maybe a linear model of frag count by aliquot, or a heatmap of frag count by aliquot and sv type
# heatmap of average fragment count by aliquot and sv type
pivot = events.pivot_table(
    index="aliquot",
    columns="sv_type",
    values="frag_count",
    aggfunc="mean",
    fill_value=0
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu")
plt.xlabel("SV Type")
plt.ylabel("Aliquot")
plt.title("Average Fragment Count by Aliquot and SV Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
# breakpoint2 is within 1000bp of 9029104 on chr2
# and the event is a tandem duplication (DUP)
events_of_interest = events[
    (events["chr2"] == "chr2") &
    (events["breakpoint2"].between(9028104, 9030104)) &
    (events["sv_type"] == "DUP")
]
events_of_interest

# %%
# Circos-style plots for each aliquot using hg19 chromosome sizes (pycirclize).
from pycirclize import Circos

HG19_CHR_SIZES = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr19": 59128983,
    "chr20": 63025520,
    "chr21": 48129895,
    "chr22": 51304566,
    "chrX": 155270560,
    "chrY": 59373566,
}


def plot_circos_for_aliquot(events, aliquot, chr_sizes=HG19_CHR_SIZES):
    subset = events[events["aliquot"] == aliquot].copy()
    subset["breakpoint1"] = pd.to_numeric(subset["breakpoint1"], errors="coerce")
    subset["breakpoint2"] = pd.to_numeric(subset["breakpoint2"], errors="coerce")
    subset = subset.dropna(subset=["breakpoint1", "breakpoint2"])
    subset = subset[
        subset["chr1"].isin(chr_sizes) &
        subset["chr2"].isin(chr_sizes)
    ]

    if subset.empty:
        return

    sv_type_colors = {
        "DEL": "#1f77b4",
        "DUP": "#ff7f0e",
        "INV": "#2ca02c",
        "TRA": "#d62728",
        "UNK": "#7f7f7f"
    }

    circos = Circos(chr_sizes, space=2)
    for sector in circos.sectors:
        sector.axis()
        sector.text(sector.name.replace("chr", ""), r=120, size=8)

    max_weight = subset["frag_count"].max()
    for _, row in subset.iterrows():
        chr1 = row["chr1"]
        chr2 = row["chr2"]
        bp1 = int(max(0, min(row["breakpoint1"], chr_sizes[chr1] - 1)))
        bp2 = int(max(0, min(row["breakpoint2"], chr_sizes[chr2] - 1)))
        weight = 0.5 + 3.0 * (row["frag_count"] / max_weight)
        sv_type = str(row.get("sv_type", "NA"))
        color = sv_type_colors.get(sv_type, "#7f7f7f")
        circos.link(
            (chr1, bp1, bp1 + 1),
            (chr2, bp2, bp2 + 1),
            lw=weight,
            color=color,
            alpha=0.35,
        )

    fig = circos.plotfig(figsize=(8, 8))
    fig.suptitle(f"Circos links for {aliquot} (hg19)")
    plt.show()


for aliquot in order_aliquots(events)["aliquot"]:
    plot_circos_for_aliquot(events, aliquot)


# %%
# what is the distribution of SV span?
events["breakpoint1"] = pd.to_numeric(events["breakpoint1"], errors="coerce")
events["breakpoint2"] = pd.to_numeric(events["breakpoint2"], errors="coerce")
events["sv_span"] = (events["breakpoint2"] - events["breakpoint1"]).abs()
# print summary stats of sv_span
print(events["sv_span"].describe())

# %%
# print out svs that are smaller than 200bp
events[events["sv_span"] <= events['mh_length']][
    [
        "sample",
        "aliquot",
        "chr1",
        "breakpoint1",
        "chr2",
        "breakpoint2",
        "sv_type",
        "frag_count",
        "sv_span"
    ]
]

# %%

events[events["sv_span"] < 10][
    [
        "sample",
        "aliquot",
        "chr1",
        "breakpoint1",
        "chr2",
        "breakpoint2",
        "sv_type",
        "frag_count",
        "sv_span"
    ]
]
# %%
# load the raw events from NA08_C
raw_events = pd.read_csv(
    '../../data/dss_020226/NA08_E.hg19.dedup.chr2_1152626_w5.annotated.tsv',
    sep='\t'
)
raw_events[(raw_events['chr2'] == 'chr2') & (raw_events['breakpoint2'] == 1152631)][[
    'read_name',
    'chr1',
    'breakpoint1',
    'is_forward1',
    'chr2',
    'breakpoint2',
    'is_forward2',
    'sv_type',
    'clip_side1',
    'clip_side2',
    'cigar1',
    'cigar2',
    'mapping_quality1',
    'mapping_quality2'
]]

# %%
# LH00328:605:2325J5LT3:5:1154:28827:26492
# look for this read in /Users/youyun/Documents/HMS/PhD/beroukhimlab/nti/data/DSS/NA08_E.hg19.dedup.bam
import pysam
bam_path = '/Users/youyun/Documents/HMS/PhD/beroukhimlab/nti/data/DSS/NA08_E.hg19.dedup.bam'
bam = pysam.AlignmentFile(bam_path, "rb")
read_name = 'LH00328:605:2325J5LT3:5:2229:50932:15075'
for read in bam.fetch(until_eof=True):
    if read.query_name == read_name:
        print(read)
        print(read.cigarstring, read.is_forward, read.is_supplementary)
        print(read.reference_name, read.reference_start, read.reference_end)
        print(read.next_reference_name, read.next_reference_start)
        print(read.mapping_quality)
        break
# %%
