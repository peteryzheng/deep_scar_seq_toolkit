#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
SVs = pd.read_csv("read_level_sv_events.tsv", sep="\t")
# %%
# count the total events
# then count the number of unique events by chr1, breakpoint 1, chr2, breakpoint2, clip_side1, clip_side2, sv_type, mh_seq, ins_seq
# for each unique event, count how many times it occurs
# also count the number of unique readnames supporting each unique event
total_events = len(SVs)
unique_events = SVs.fillna('').groupby(
    [
        "chr1",
        "breakpoint1",
        "chr2",
        "breakpoint2",
        "clip_side1",
        "clip_side2",
        "sv_type",
        'mh_seq',
        'ins_seq'
    ]
).agg(
    read_count=("read_name", "count"),
    frag_count=("read_name", lambda x: x.nunique())
).reset_index().sort_values(by="read_count", ascending=False)
unique_events

#%%
# print the read count on y and frag count on x
# color dot by sv_type
# add a line of y = 2x
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=unique_events,
    x="frag_count",
    y="read_count",
    hue="sv_type",
    alpha=0.7
)
x = np.linspace(0, unique_events["frag_count"].max(), 100)
plt.plot(x, 2*x, color='red', linestyle='--', label='y = 2x')
plt.xlabel("Number of Unique Fragments Supporting SV")
plt.ylabel("Number of Reads Supporting SV")
plt.title("SV Support: Reads vs Unique Fragments")
plt.legend(title="SV Type")
plt.grid(True)
plt.show()

# %%
# I want to know whenever there are two rows with the same read name supporting the same SV
# are they representing two reads from the same fragment?
# i.e. is_read1 from the two rows should be different
duplicated_reads = SVs.duplicated(
    subset=[
        "chr1",
        "breakpoint1",
        "chr2",
        "breakpoint2",
        "clip_side1",
        "clip_side2",
        "sv_type",
        'mh_seq',
        'ins_seq',
        'read_name'
    ],
    keep=False
)
duplicated_SVs = SVs[duplicated_reads].copy()
duplicated_SVs['read_number'] = duplicated_SVs['read_number'].astype(str)
dup_summary = duplicated_SVs.fillna('').groupby(
    [
        "chr1",
        "breakpoint1",
        "chr2",
        "breakpoint2",
        "clip_side1",
        "clip_side2",
        "sv_type",
        'mh_seq',
        'ins_seq',
        'read_name'
    ]
).agg(
    read_count=("read_name", "count"),
    read_number_values=("read_number", lambda x: ','.join(sorted(x.unique())))
).reset_index()
dup_summary[dup_summary['read_count'] == 2]['read_number_values'].value_counts()
# %%
# now I will calculate the span of the sv by taking the abs(breakpoint2 - breakpoint1)
# span of the SV will be 0 for translocations
unique_events['sv_span'] = np.where(
    unique_events['chr1'] == unique_events['chr2'],
    abs(unique_events['breakpoint2'] - unique_events['breakpoint1']),
    0
)
# plot a dotplot of sv_span vs read_count
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=unique_events,
    x="sv_span",
    y="frag_count",
    hue="sv_type",
    alpha=0.7
)
plt.xscale('log')
plt.xlabel("SV Span (log scale)")
plt.ylabel("Number of Reads Supporting SV")
plt.title("SV Support: Reads vs SV Span")
plt.legend(title="SV Type")
plt.grid(True)
plt.show()
# %%