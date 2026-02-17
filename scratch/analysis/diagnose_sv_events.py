#!/usr/bin/env python3
"""Comprehensive SV event diagnosis for steps 1-8.

This script builds master event/read summaries and outputs diagnostics for:
1) Master table construction.
2) Event burden normalization.
3) Event support strength (read/frag counts).
4) Alignment confidence differences.
5) SV/breakpoint structure differences.
6) Junction signature differences.
7) Recurrence and sharing across aliquots/groups.
8) Outlier aliquot detection using robust z-scores.

It also writes interactive Plotly HTML figures when Plotly is available.
"""

#%%
from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


#%%
SAMPLE_GROUPS = {
    "NA01": "no_cut",
    "NA02": "no_cut",
    "NA06": "no_cut",
    "NA03": "inner_cut",
    "NA04": "inner_cut",
    "NA05": "inner_cut",
    "NA07": "outer_cut",
    "NA08": "outer_cut",
}

CUT_SITE_LABELS = {
    "chr2_1152626": "outer_cut_site",
    "chr2_1153052": "inner_cut_site",
}
GROUP_ORDER = ["no_cut", "inner_cut", "outer_cut"]

AGG_FILE_GLOB = "*.hg19.dedup.*_w5.aggregated.tsv"
ANN_FILE_GLOB = "*.hg19.dedup.*_w5.annotated.tsv"

AGG_REQUIRED_COLUMNS = [
    "chr1",
    "breakpoint1",
    "chr2",
    "breakpoint2",
    "clip_side1",
    "clip_side2",
    "sv_type",
    "mh_seq",
    "ins_seq",
    "read_count",
    "frag_count",
    "mapq1_mean",
    "match1_mean",
    "mapq2_mean",
    "match2_mean",
    "mh_length",
    "ins_length",
]

ANN_SUMMARY_COLUMNS = [
    "mapping_quality1",
    "mapping_quality2",
    "largest_match_length1",
    "largest_match_length2",
    "largest_clip_length1",
    "largest_clip_length2",
    "extra_bases1",
    "extra_bases2",
]

ANN_EVENT_COLUMNS = [
    "chr1",
    "breakpoint1",
    "chr2",
    "breakpoint2",
    "clip_side1",
    "clip_side2",
    "sv_type",
    "mh_seq",
    "ins_seq",
    "is_forward1",
    "is_forward2",
]


#%%
EVENTS_DIR = Path("data/dss_020226_events")
ANNOTATED_DIR = Path("data/dss_020226")
OUTPUT_DIR = Path("sv_diagnosis")
FOCUS_CUT_SITE = "chr2_1153052"
NO_PLOTS = False
PLOT_BASE_FONT_SIZE = 18
PLOT_TITLE_FONT_SIZE = 24
PLOT_AXIS_TITLE_FONT_SIZE = 20
PLOT_TICK_FONT_SIZE = 16
PLOT_LEGEND_FONT_SIZE = 16
PLOT_LEGEND_TITLE_FONT_SIZE = 18
PLOT_HOVER_FONT_SIZE = 15


#%%
def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    return here


#%%
def resolve_input_dir(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()
    root = find_project_root()
    candidate = (root / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


#%%
def parse_filename_meta(path: Path) -> dict[str, str]:
    name = path.name
    marker = ".hg19.dedup."
    if marker not in name:
        raise ValueError(f"Unexpected filename format (missing '{marker}'): {name}")
    aliquot, rest = name.split(marker, 1)
    sample = aliquot.split("_")[0]
    cut_site = ""
    if "_w5." in rest:
        cut_site = rest.split("_w5.", 1)[0]
    if not cut_site:
        raise ValueError(f"Could not parse cut site from filename: {name}")
    return {
        "sample": sample,
        "aliquot": aliquot,
        "cut_site": cut_site,
        "sample_group": SAMPLE_GROUPS.get(sample, "other"),
        "cut_site_label": CUT_SITE_LABELS.get(cut_site, "other_cut_site"),
    }


#%%
def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


#%%
def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = {}
    for base in seq:
        counts[base] = counts.get(base, 0) + 1
    total = len(seq)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


#%%
def is_low_complexity_ins(seq: str) -> bool:
    if not isinstance(seq, str) or seq == "":
        return False
    unique_bases = len(set(seq))
    max_base_fraction = max(seq.count(b) for b in set(seq)) / len(seq)
    entropy = shannon_entropy(seq)
    return unique_bases <= 2 or max_base_fraction >= 0.8 or entropy <= 1.2


#%%
def robust_z(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series([0.0] * len(x), index=x.index, dtype=float)
    return (x - med) / (1.4826 * mad)


#%%
def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.lower().map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


#%%
def add_event_key(df: pd.DataFrame) -> pd.DataFrame:
    df["mh_seq"] = df["mh_seq"].fillna("")
    df["ins_seq"] = df["ins_seq"].fillna("")
    df["event_key"] = (
        df["chr1"].astype(str)
        + ":"
        + df["breakpoint1"].astype("Int64").astype(str)
        + "|"
        + df["chr2"].astype(str)
        + ":"
        + df["breakpoint2"].astype("Int64").astype(str)
        + "|"
        + df["clip_side1"].astype(str)
        + "|"
        + df["clip_side2"].astype(str)
        + "|"
        + df["sv_type"].astype(str)
        + "|"
        + df["mh_seq"].astype(str)
        + "|"
        + df["ins_seq"].astype(str)
    )
    return df


#%%
def seq_entropy_safe(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(shannon_entropy)


#%%
def comparison_stats_text(
    df: pd.DataFrame,
    value_col: str,
    cut_site: str,
    group_col: str = "sample_group",
    g1: str = "no_cut",
    g2: str = "inner_cut",
) -> str:
    subset = df[df["cut_site"] == cut_site].copy()
    a = pd.to_numeric(subset[subset[group_col] == g1][value_col], errors="coerce").dropna()
    b = pd.to_numeric(subset[subset[group_col] == g2][value_col], errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return f"{g1} vs {g2}: insufficient data"
    med_delta = a.median() - b.median()
    mean_delta = a.mean() - b.mean()
    return (
        f"{g1} vs {g2} | n={len(a)} vs {len(b)} | "
        f"median delta={med_delta:.3f} | mean delta={mean_delta:.3f}"
    )


#%%
def compute_clonality_by_aliquot(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["sample_group", "cut_site", "cut_site_label", "sample", "aliquot"]
    for key_vals, grp in master.groupby(keys):
        frag = pd.to_numeric(grp["frag_count"], errors="coerce").dropna()
        frag = frag[frag > 0].sort_values(ascending=False)
        total = float(frag.sum()) if len(frag) else 0.0
        if total == 0:
            top1 = float("nan")
            top5 = float("nan")
            hhi = float("nan")
            eff = float("nan")
        else:
            share = frag / total
            top1 = float(share.iloc[0])
            top5 = float(share.head(5).sum())
            hhi = float((share**2).sum())
            eff = float(1.0 / hhi) if hhi > 0 else float("nan")
        rows.append(
            {
                "sample_group": key_vals[0],
                "cut_site": key_vals[1],
                "cut_site_label": key_vals[2],
                "sample": key_vals[3],
                "aliquot": key_vals[4],
                "event_count": int(len(grp)),
                "total_fragments": total,
                "top1_frag_share": top1,
                "top5_frag_share": top5,
                "hhi_frag": hhi,
                "effective_event_number": eff,
            }
        )
    return pd.DataFrame(rows)


#%%
def log_ticks_for_mean_frag(values: pd.Series) -> tuple[list[float], list[str]]:
    x = pd.to_numeric(values, errors="coerce")
    x = x[x > 0]
    if x.empty:
        return [1.0, 2.0, 3.0, 4.0, 5.0], ["1", "2", "3", "4", "5"]
    min_v = float(x.min())
    max_v = float(x.max())
    candidates = [0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    lo = max(0.8, min_v * 0.95)
    hi = max_v * 1.05
    ticks = [t for t in candidates if lo <= t <= hi]
    if len(ticks) < 3:
        ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
    return ticks, [f"{t:g}" for t in ticks]

#%%
def load_aggregated_events(events_dir: Path) -> pd.DataFrame:
    events_dir = resolve_input_dir(events_dir)
    paths = sorted(events_dir.rglob(AGG_FILE_GLOB))
    if not paths:
        raise SystemExit(
            f"No aggregated files found in {events_dir} with pattern {AGG_FILE_GLOB}"
        )

    records: list[pd.DataFrame] = []
    empty_file_count = 0
    for path in paths:
        meta = parse_filename_meta(path)
        events = pd.read_csv(path, sep="\t")
        missing = [c for c in AGG_REQUIRED_COLUMNS if c not in events.columns]
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")
        if events.empty:
            empty_file_count += 1
            continue
        for key, value in meta.items():
            events[key] = value
        events["source_file"] = str(path)
        records.append(events)

    if not records:
        raise SystemExit(f"All aggregated files were empty under {events_dir}")

    df = pd.concat(records, ignore_index=True)
    df = ensure_numeric(
        df,
        [
            "breakpoint1",
            "breakpoint2",
            "read_count",
            "frag_count",
            "mapq1_mean",
            "match1_mean",
            "mapq2_mean",
            "match2_mean",
            "mh_length",
            "ins_length",
        ],
    )
    df["read_count"] = pd.to_numeric(df["read_count"], errors="coerce")
    df["frag_count"] = pd.to_numeric(df["frag_count"], errors="coerce")
    df = add_event_key(df)
    df["intrachrom"] = df["chr1"] == df["chr2"]
    df["ins_seq"] = df["ins_seq"].fillna("")
    df["mh_seq"] = df["mh_seq"].fillna("")
    df["ins_nonzero"] = df["ins_length"].fillna(0) > 0
    df["ins_low_complexity"] = df["ins_seq"].map(is_low_complexity_ins)
    if empty_file_count:
        print(f"Skipped {empty_file_count} empty aggregated files.")
    print(f"Loaded {len(df)} unique SV events from {len(records)} files.")
    return df


#%%
def load_read_level_event_orientation(annotated_dir: Path) -> pd.DataFrame:
    annotated_dir = resolve_input_dir(annotated_dir)
    paths = sorted(annotated_dir.rglob(ANN_FILE_GLOB))
    if not paths:
        raise SystemExit(
            f"No annotated files found in {annotated_dir} with pattern {ANN_FILE_GLOB}"
        )

    rows: list[pd.DataFrame] = []
    for path in paths:
        meta = parse_filename_meta(path)
        df = pd.read_csv(path, sep="\t", usecols=ANN_EVENT_COLUMNS)
        if df.empty:
            continue
        df = ensure_numeric(df, ["breakpoint1", "breakpoint2"])
        df = add_event_key(df)
        df["is_forward1"] = to_bool_series(df["is_forward1"])
        df["is_forward2"] = to_bool_series(df["is_forward2"])
        for key, value in meta.items():
            df[key] = value
        rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_group",
                "cut_site",
                "cut_site_label",
                "sample",
                "aliquot",
                "event_key",
                "read_rows",
                "forward1_frac",
                "forward2_frac",
            ]
        )

    all_reads = pd.concat(rows, ignore_index=True)
    orientation = (
        all_reads.groupby(
            ["sample_group", "cut_site", "cut_site_label", "sample", "aliquot", "event_key"],
            as_index=False,
        )
        .agg(
            read_rows=("event_key", "size"),
            forward1_frac=("is_forward1", "mean"),
            forward2_frac=("is_forward2", "mean"),
        )
    )
    return orientation


#%%
def summarize_annotated_by_aliquot(annotated_dir: Path) -> pd.DataFrame:
    annotated_dir = resolve_input_dir(annotated_dir)
    paths = sorted(annotated_dir.rglob(ANN_FILE_GLOB))
    if not paths:
        raise SystemExit(
            f"No annotated files found in {annotated_dir} with pattern {ANN_FILE_GLOB}"
        )

    rows: list[dict[str, object]] = []
    empty_file_count = 0
    for path in paths:
        meta = parse_filename_meta(path)
        df = pd.read_csv(path, sep="\t", usecols=ANN_SUMMARY_COLUMNS)
        df = ensure_numeric(df, ANN_SUMMARY_COLUMNS)
        n = len(df)
        if n == 0:
            empty_file_count += 1
        row: dict[str, object] = {
            **meta,
            "source_file": str(path),
            "annotated_rows": int(n),
        }
        for col in ANN_SUMMARY_COLUMNS:
            s = df[col]
            row[f"{col}_mean"] = float(s.mean()) if n > 0 else float("nan")
            row[f"{col}_median"] = float(s.median()) if n > 0 else float("nan")
            row[f"{col}_p90"] = float(s.quantile(0.9)) if n > 0 else float("nan")
        row["extra_bases1_nonzero_frac"] = (
            float((df["extra_bases1"] > 0).mean()) if n > 0 else float("nan")
        )
        row["extra_bases2_nonzero_frac"] = (
            float((df["extra_bases2"] > 0).mean()) if n > 0 else float("nan")
        )
        rows.append(row)
    if empty_file_count:
        print(f"Observed {empty_file_count} empty annotated files.")
    return pd.DataFrame(rows)


#%%
def add_cut_distance(events: pd.DataFrame) -> pd.DataFrame:
    cut_pos = (
        events["cut_site"]
        .astype(str)
        .str.split("_")
        .str[-1]
        .pipe(pd.to_numeric, errors="coerce")
    )
    events["cut_position"] = cut_pos
    events["abs_bp1_dist_to_cut"] = (events["breakpoint1"] - events["cut_position"]).abs()
    return events


#%%
def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


#%%
def step2_burden(master: pd.DataFrame, ann_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    burden = (
        master.groupby(
            ["sample_group", "cut_site", "cut_site_label", "sample", "aliquot"],
            as_index=False,
        )
        .agg(
            unique_events=("event_key", "count"),
            mean_read_count=("read_count", "mean"),
            median_read_count=("read_count", "median"),
            mean_frag_count=("frag_count", "mean"),
            median_frag_count=("frag_count", "median"),
            singleton_frac=("frag_count", lambda s: (s == 1).mean()),
            frag_ge2_frac=("frag_count", lambda s: (s >= 2).mean()),
            frag_ge5_frac=("frag_count", lambda s: (s >= 5).mean()),
            mean_abs_bp1_dist_to_cut=("abs_bp1_dist_to_cut", "mean"),
        )
    )
    burden = burden.merge(
        ann_summary[
            [
                "sample_group",
                "cut_site",
                "sample",
                "aliquot",
                "annotated_rows",
            ]
        ],
        on=["sample_group", "cut_site", "sample", "aliquot"],
        how="left",
    )
    burden["events_per_read"] = burden["unique_events"] / burden["annotated_rows"]
    burden["events_per_1k_reads"] = burden["events_per_read"] * 1000.0

    burden_group = (
        burden.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            aliquots=("aliquot", "nunique"),
            total_unique_events=("unique_events", "sum"),
            mean_unique_events=("unique_events", "mean"),
            median_unique_events=("unique_events", "median"),
            mean_events_per_1k_reads=("events_per_1k_reads", "mean"),
            mean_frag_count=("mean_frag_count", "mean"),
            mean_singleton_frac=("singleton_frac", "mean"),
        )
    )
    return burden, burden_group


#%%
def step3_support(master: pd.DataFrame) -> pd.DataFrame:
    return (
        master.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            events=("event_key", "count"),
            read_count_mean=("read_count", "mean"),
            read_count_median=("read_count", "median"),
            read_count_p90=("read_count", lambda s: s.quantile(0.9)),
            frag_count_mean=("frag_count", "mean"),
            frag_count_median=("frag_count", "median"),
            frag_count_p90=("frag_count", lambda s: s.quantile(0.9)),
            singleton_frac=("frag_count", lambda s: (s == 1).mean()),
            frag_ge2_frac=("frag_count", lambda s: (s >= 2).mean()),
            frag_ge10_frac=("frag_count", lambda s: (s >= 10).mean()),
        )
    )


#%%
def step4_alignment(master: pd.DataFrame, ann_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_level = (
        master.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            mapq1_mean=("mapq1_mean", "mean"),
            mapq1_median=("mapq1_mean", "median"),
            mapq2_mean=("mapq2_mean", "mean"),
            mapq2_median=("mapq2_mean", "median"),
            match1_mean=("match1_mean", "mean"),
            match2_mean=("match2_mean", "mean"),
            match1_p10=("match1_mean", lambda s: s.quantile(0.1)),
            match2_p10=("match2_mean", lambda s: s.quantile(0.1)),
        )
    )

    read_level = (
        ann_summary.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            aliquots=("aliquot", "nunique"),
            annotated_rows=("annotated_rows", "sum"),
            mapping_quality1_mean=("mapping_quality1_mean", "mean"),
            mapping_quality2_mean=("mapping_quality2_mean", "mean"),
            largest_match_length1_mean=("largest_match_length1_mean", "mean"),
            largest_match_length2_mean=("largest_match_length2_mean", "mean"),
            largest_clip_length1_mean=("largest_clip_length1_mean", "mean"),
            largest_clip_length2_mean=("largest_clip_length2_mean", "mean"),
            extra_bases1_nonzero_frac=("extra_bases1_nonzero_frac", "mean"),
            extra_bases2_nonzero_frac=("extra_bases2_nonzero_frac", "mean"),
        )
    )
    return event_level, read_level


#%%
def step5_structure(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sv_type = (
        master.groupby(["sample_group", "cut_site", "cut_site_label", "sv_type"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    sv_type["fraction"] = sv_type["events"] / sv_type.groupby(
        ["sample_group", "cut_site"]
    )["events"].transform("sum")

    clip_combo = (
        master.assign(clip_combo=master["clip_side1"].astype(str) + "_" + master["clip_side2"].astype(str))
        .groupby(["sample_group", "cut_site", "cut_site_label", "clip_combo"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    clip_combo["fraction"] = clip_combo["events"] / clip_combo.groupby(
        ["sample_group", "cut_site"]
    )["events"].transform("sum")

    partner_chr = (
        master.groupby(["sample_group", "cut_site", "cut_site_label", "chr2"], as_index=False)
        .size()
        .rename(columns={"size": "events", "chr2": "partner_chr"})
    )
    partner_chr["fraction"] = partner_chr["events"] / partner_chr.groupby(
        ["sample_group", "cut_site"]
    )["events"].transform("sum")
    return sv_type, clip_combo, partner_chr


#%%
def step6_junction(master: pd.DataFrame) -> pd.DataFrame:
    return (
        master.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            events=("event_key", "count"),
            mh_length_mean=("mh_length", "mean"),
            mh_length_median=("mh_length", "median"),
            mh_length_p90=("mh_length", lambda s: s.quantile(0.9)),
            ins_length_mean=("ins_length", "mean"),
            ins_length_median=("ins_length", "median"),
            ins_nonzero_frac=("ins_nonzero", "mean"),
            ins_low_complexity_frac=("ins_low_complexity", "mean"),
        )
    )


#%%
def step7_recurrence(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_aliquot = master[
        ["sample_group", "cut_site", "cut_site_label", "aliquot", "event_key"]
    ].drop_duplicates()

    recurrence = (
        event_aliquot.groupby(["sample_group", "cut_site", "cut_site_label", "event_key"], as_index=False)
        .agg(aliquot_count=("aliquot", "nunique"))
    )
    recurrence_summary = (
        recurrence.groupby(["sample_group", "cut_site", "cut_site_label"], as_index=False)
        .agg(
            unique_event_keys=("event_key", "count"),
            single_aliquot_keys=("aliquot_count", lambda s: (s == 1).sum()),
            multi_aliquot_keys=("aliquot_count", lambda s: (s >= 2).sum()),
            max_aliquots_per_key=("aliquot_count", "max"),
        )
    )
    recurrence_summary["multi_aliquot_frac"] = (
        recurrence_summary["multi_aliquot_keys"] / recurrence_summary["unique_event_keys"]
    )

    cross = (
        master[["cut_site", "cut_site_label", "sample_group", "event_key"]]
        .drop_duplicates()
        .pivot_table(
            index=["cut_site", "cut_site_label", "event_key"],
            columns="sample_group",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )
    group_cols = [c for c in cross.columns if c not in ["cut_site", "cut_site_label", "event_key"]]
    for col in group_cols:
        cross[col] = cross[col] > 0
    cross["groups_present"] = cross[group_cols].sum(axis=1)
    cross["shared_across_groups"] = cross["groups_present"] >= 2

    cross_summary = (
        cross.groupby(["cut_site", "cut_site_label"], as_index=False)
        .agg(
            unique_event_keys=("event_key", "count"),
            shared_across_groups=("shared_across_groups", "sum"),
        )
    )
    cross_summary["shared_frac"] = (
        cross_summary["shared_across_groups"] / cross_summary["unique_event_keys"]
    )

    return recurrence, recurrence_summary, cross_summary


#%%
def step8_outliers(master: pd.DataFrame, burden: pd.DataFrame) -> pd.DataFrame:
    features = (
        master.groupby(["sample_group", "cut_site", "cut_site_label", "sample", "aliquot"], as_index=False)
        .agg(
            event_rows=("event_key", "count"),
            mean_frag_count=("frag_count", "mean"),
            singleton_frac=("frag_count", lambda s: (s == 1).mean()),
            mean_mapq1=("mapq1_mean", "mean"),
            mean_mapq2=("mapq2_mean", "mean"),
            mean_match1=("match1_mean", "mean"),
            mean_match2=("match2_mean", "mean"),
            mh_length_mean=("mh_length", "mean"),
            ins_nonzero_frac=("ins_nonzero", "mean"),
            abs_bp1_dist_mean=("abs_bp1_dist_to_cut", "mean"),
        )
        .merge(
            burden[
                [
                    "sample_group",
                    "cut_site",
                    "sample",
                    "aliquot",
                    "annotated_rows",
                    "unique_events",
                    "events_per_1k_reads",
                ]
            ],
            on=["sample_group", "cut_site", "sample", "aliquot"],
            how="left",
        )
    )

    z_cols = [
        "unique_events",
        "events_per_1k_reads",
        "mean_frag_count",
        "singleton_frac",
        "mean_mapq2",
        "mean_match2",
        "mh_length_mean",
        "ins_nonzero_frac",
        "abs_bp1_dist_mean",
    ]
    out = features.copy()
    for col in z_cols:
        out[f"z_{col}"] = out.groupby(["sample_group", "cut_site"])[col].transform(robust_z)
    z_matrix = out[[f"z_{c}" for c in z_cols]].abs()
    out["max_abs_robust_z"] = z_matrix.max(axis=1)
    out["outlier_flag_3"] = out["max_abs_robust_z"] >= 3.0
    out = out.sort_values(
        ["outlier_flag_3", "max_abs_robust_z", "cut_site", "sample", "aliquot"],
        ascending=[False, False, True, True, True],
    )
    return out


#%%
def _try_import_plotly():
    try:
        import plotly.express as px  # type: ignore
        import plotly.io as pio  # type: ignore
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        return px, pio, go, make_subplots
    except Exception:
        return None, None, None, None


#%%
def write_interactive_plots(
    outdir: Path,
    master: pd.DataFrame,
    burden: pd.DataFrame,
    orientation: pd.DataFrame,
    focus_cut_site: str,
) -> None:
    px, pio, go, make_subplots = _try_import_plotly()
    if px is None or pio is None or go is None or make_subplots is None:
        print("Plotly not available; skipping interactive plot generation.")
        return

    focus_master = master[master["cut_site"] == focus_cut_site].copy()
    focus_burden = burden[burden["cut_site"] == focus_cut_site].copy()
    focus_orientation = orientation[orientation["cut_site"] == focus_cut_site].copy()
    clonality = compute_clonality_by_aliquot(master)
    focus_clonality = clonality[clonality["cut_site"] == focus_cut_site].copy()
    for df in [focus_master, focus_burden, focus_orientation, focus_clonality]:
        if "sample_group" in df.columns:
            df["sample_group"] = pd.Categorical(
                df["sample_group"], categories=GROUP_ORDER, ordered=True
            )

    focus_master["read_frag_ratio"] = (
        pd.to_numeric(focus_master["read_count"], errors="coerce")
        / pd.to_numeric(focus_master["frag_count"], errors="coerce").replace(0, pd.NA)
    )
    focus_master["mapq_delta"] = (
        pd.to_numeric(focus_master["mapq1_mean"], errors="coerce")
        - pd.to_numeric(focus_master["mapq2_mean"], errors="coerce")
    )
    focus_master["match_delta"] = (
        pd.to_numeric(focus_master["match1_mean"], errors="coerce")
        - pd.to_numeric(focus_master["match2_mean"], errors="coerce")
    )
    focus_master["abs_mapq_delta"] = focus_master["mapq_delta"].abs()
    focus_master["abs_match_delta"] = focus_master["match_delta"].abs()
    focus_master["mh_entropy"] = seq_entropy_safe(focus_master["mh_seq"])
    focus_master["ins_entropy"] = seq_entropy_safe(focus_master["ins_seq"])
    focus_master["junction_complexity"] = (
        focus_master["mh_entropy"] + focus_master["ins_entropy"]
    ) / 2.0

    figures: list[tuple[str, str, object]] = []

    # Summary metrics explicitly tied to current diagnosis statements.
    group_summary = (
        focus_master.groupby("sample_group", as_index=False, observed=True)
        .agg(
            mean_mapq1=("mapq1_mean", "mean"),
            mean_mapq2=("mapq2_mean", "mean"),
            tra_frac=("sv_type", lambda s: (s == "TRA").mean()),
        )
    )
    # Group support summaries as mean of aliquot-level means within each group.
    support_group_means = (
        focus_burden.groupby("sample_group", as_index=False, observed=True)
        .agg(
            mean_frag_count=("mean_frag_count", "mean"),
            singleton_frac=("singleton_frac", "mean"),
        )
    )
    group_summary = group_summary.merge(
        support_group_means, on="sample_group", how="left"
    )
    rec_base = focus_master[
        ["sample_group", "aliquot", "event_key"]
    ].drop_duplicates()
    rec_event = (
        rec_base.groupby(["sample_group", "event_key"], as_index=False, observed=True)
        .agg(aliquot_count=("aliquot", "nunique"))
    )
    rec_summary = (
        rec_event.groupby("sample_group", as_index=False, observed=True)
        .agg(multi_aliquot_frac=("aliquot_count", lambda s: (s >= 2).mean()))
    )
    group_summary = group_summary.merge(rec_summary, on="sample_group", how="left")

    def _fmt(group: str, col: str) -> str:
        row = group_summary[group_summary["sample_group"] == group]
        if row.empty:
            return "NA"
        return f"{row.iloc[0][col]:.3f}"

    summary_text = (
        f"no_cut vs inner_cut | mean_frag_count {_fmt('no_cut', 'mean_frag_count')} vs {_fmt('inner_cut', 'mean_frag_count')} | "
        f"singleton_frac {_fmt('no_cut', 'singleton_frac')} vs {_fmt('inner_cut', 'singleton_frac')} | "
        f"mean_mapq1 {_fmt('no_cut', 'mean_mapq1')} vs {_fmt('inner_cut', 'mean_mapq1')} | "
        f"mean_mapq2 {_fmt('no_cut', 'mean_mapq2')} vs {_fmt('inner_cut', 'mean_mapq2')} | "
        f"TRA_frac {_fmt('no_cut', 'tra_frac')} vs {_fmt('inner_cut', 'tra_frac')} | "
        f"multi_aliquot_frac {_fmt('no_cut', 'multi_aliquot_frac')} vs {_fmt('inner_cut', 'multi_aliquot_frac')}"
    )
    palette = {
        "no_cut": "#d62728",
        "inner_cut": "#1f77b4",
        "outer_cut": "#2ca02c",
        "other": "#7f7f7f",
    }
    for metric, pretty, y_title, as_percent in [
        ("mean_frag_count", "Mean Frag Count", "Mean frag_count per event", False),
        ("singleton_frac", "Singleton Fraction", "Singleton fraction of events", True),
    ]:
        log_tickvals: list[float] = []
        log_ticktext: list[str] = []
        if metric == "mean_frag_count":
            log_tickvals, log_ticktext = log_ticks_for_mean_frag(focus_burden[metric])
        panel = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"{pretty}: group means", f"{pretty}: aliquot distribution"),
            horizontal_spacing=0.12,
        )
        gs = group_summary.dropna(subset=[metric]).copy()
        gs = gs.sort_values("sample_group")
        values = gs[metric] * 100.0 if as_percent else gs[metric]
        panel.add_trace(
            go.Bar(
                x=gs["sample_group"],
                y=values,
                marker_color=[palette.get(g, "#7f7f7f") for g in gs["sample_group"]],
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                name="group_mean",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for group in [g for g in GROUP_ORDER if g in set(focus_burden["sample_group"].astype(str))]:
            sub = focus_burden[focus_burden["sample_group"] == group]
            if sub.empty:
                continue
            y = sub[metric] * 100.0 if as_percent else sub[metric]
            panel.add_trace(
                go.Box(
                    x=sub["sample_group"],
                    y=y,
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.35,
                    marker=dict(size=9, color=palette.get(group, "#7f7f7f")),
                    line=dict(color=palette.get(group, "#7f7f7f")),
                    fillcolor="rgba(0,0,0,0)",
                    name=group,
                    legendgroup=group,
                    showlegend=False,
                    hovertext=sub["aliquot"],
                    hovertemplate="group=%{x}<br>value=%{y:.3f}<br>aliquot=%{hovertext}<extra></extra>",
                ),
                row=1,
                col=2,
            )
        panel.update_layout(
            title=f"Diagnosis summary: {pretty} ({focus_cut_site})",
            yaxis_title=y_title + (" (%)" if as_percent else ""),
            yaxis2_title=y_title + (" (%)" if as_percent else ""),
        )
        if metric == "mean_frag_count":
            panel.update_yaxes(
                type="log",
                tickmode="array",
                tickvals=log_tickvals,
                ticktext=log_ticktext,
                row=1,
                col=1,
            )
            panel.update_yaxes(
                type="log",
                tickmode="array",
                tickvals=log_tickvals,
                ticktext=log_ticktext,
                row=1,
                col=2,
            )
        if as_percent:
            panel.update_yaxes(ticksuffix="%", row=1, col=1)
            panel.update_yaxes(ticksuffix="%", row=1, col=2)
        figures.append((f"Diagnosis Summary: Support ({pretty})", summary_text, panel))

        sample_df = focus_burden.assign(
            plot_value=lambda d: d[metric] * 100.0 if as_percent else d[metric]
        )
        sample_order = sorted(sample_df["sample"].dropna().unique())
        sample_fig = go.Figure()
        for group in [g for g in GROUP_ORDER if g in set(sample_df["sample_group"].astype(str))]:
            sub = sample_df[sample_df["sample_group"] == group]
            if sub.empty:
                continue
            sample_fig.add_trace(
                go.Box(
                    x=sub["sample"],
                    y=sub["plot_value"],
                    name=group,
                    legendgroup=group,
                    marker=dict(color=palette.get(group, "#7f7f7f"), size=8),
                    line=dict(color=palette.get(group, "#7f7f7f")),
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.35,
                    hovertext=sub["aliquot"],
                    hovertemplate=(
                        "sample=%{x}<br>value=%{y:.3f}<br>group="
                        + group
                        + "<br>aliquot=%{hovertext}<extra></extra>"
                    ),
                )
            )
        sample_fig.update_layout(
            title=f"{pretty} by sample (box + aliquot points) ({focus_cut_site})",
            xaxis_title="sample",
            yaxis_title=y_title + (" (%)" if as_percent else ""),
            boxmode="overlay",
        )
        sample_fig.update_xaxes(categoryorder="array", categoryarray=sample_order)
        if metric == "mean_frag_count":
            sample_fig.update_yaxes(
                type="log",
                tickmode="array",
                tickvals=log_tickvals,
                ticktext=log_ticktext,
            )
        sample_fig.update_yaxes(title_text=y_title + (" (%)" if as_percent else ""))
        if as_percent:
            sample_fig.update_yaxes(ticksuffix="%")
        figures.append((f"Diagnosis Summary: {pretty} by Sample", summary_text, sample_fig))

    # Singleton sensitivity to aliquot depth threshold.
    singleton_thresholds = [1, 10, 20]
    singleton_rows: list[dict[str, object]] = []
    for min_events in singleton_thresholds:
        subset = focus_burden[focus_burden["unique_events"] >= min_events].copy()
        if subset.empty:
            continue
        grp = (
            subset.groupby("sample_group", as_index=False, observed=True)
            .agg(
                mean_singleton_frac=("singleton_frac", "mean"),
                median_singleton_frac=("singleton_frac", "median"),
                n_aliquots=("aliquot", "nunique"),
            )
        )
        grp["min_events"] = min_events
        singleton_rows.append(grp)
    if singleton_rows:
        singleton_sensitivity = pd.concat(singleton_rows, ignore_index=True)
        sens_fig = go.Figure()
        for group in GROUP_ORDER:
            sub = singleton_sensitivity[singleton_sensitivity["sample_group"] == group]
            if sub.empty:
                continue
            sens_fig.add_trace(
                go.Scatter(
                    x=sub["min_events"],
                    y=sub["mean_singleton_frac"] * 100.0,
                    mode="lines+markers+text",
                    line={"color": palette.get(group, "#7f7f7f"), "width": 3},
                    marker={"size": 10},
                    text=[f"n={int(n)}" for n in sub["n_aliquots"]],
                    textposition="top center",
                    name=group,
                    legendgroup=group,
                    hovertemplate=(
                        "group=%{fullData.name}<br>min_events=%{x}"
                        "<br>mean_singleton=%{y:.2f}%<br>%{text}<extra></extra>"
                    ),
                )
            )
        sens_fig.update_layout(
            title=f"Singleton fraction sensitivity to minimum events per aliquot ({focus_cut_site})",
            xaxis_title="Minimum unique events per aliquot filter",
            yaxis_title="Mean singleton fraction across retained aliquots (%)",
        )
        sens_fig.update_xaxes(
            tickmode="array",
            tickvals=singleton_thresholds,
            ticktext=[str(x) for x in singleton_thresholds],
        )
        sens_fig.update_yaxes(range=[0, 100], ticksuffix="%")
        sens_text = "no_cut vs inner_cut (mean singleton %, by filter): "
        bits = []
        for min_events in singleton_thresholds:
            block = singleton_sensitivity[singleton_sensitivity["min_events"] == min_events]
            no_cut = block.loc[block["sample_group"] == "no_cut", "mean_singleton_frac"]
            inner = block.loc[block["sample_group"] == "inner_cut", "mean_singleton_frac"]
            if len(no_cut) and len(inner):
                bits.append(f">={min_events}: {no_cut.iloc[0]*100:.1f}% vs {inner.iloc[0]*100:.1f}%")
        sens_text += " | ".join(bits) if bits else "insufficient paired group data"
        figures.append(("Singleton Sensitivity by Aliquot Depth", sens_text, sens_fig))

    # Additional frag-count distribution diagnostics (beyond means).
    focus_frag = focus_master.copy()
    focus_frag["frag_count"] = pd.to_numeric(focus_frag["frag_count"], errors="coerce")
    focus_frag = focus_frag[focus_frag["frag_count"] > 0].copy()
    if not focus_frag.empty:
        fig = px.ecdf(
            focus_frag,
            x="frag_count",
            color="sample_group",
            color_discrete_map=palette,
            category_orders={"sample_group": GROUP_ORDER},
            hover_data=["sample", "aliquot", "sv_type"],
            title=f"Event-level frag_count ECDF ({focus_cut_site})",
        )
        fig.update_xaxes(type="log")
        stats = comparison_stats_text(focus_frag, "frag_count", focus_cut_site)
        figures.append(("Frag Count Distribution (Event-level ECDF)", stats, fig))

        fig = px.ecdf(
            focus_frag,
            x="frag_count",
            color="sample",
            color_discrete_map={
                row["sample"]: palette.get(str(row["sample_group"]), "#7f7f7f")
                for _, row in focus_frag[["sample", "sample_group"]].drop_duplicates().iterrows()
            },
            hover_data=["sample", "aliquot", "sv_type"],
            title=f"Frag_count ECDF by sample (single panel) ({focus_cut_site})",
        )
        fig.update_xaxes(type="log")
        stats = comparison_stats_text(focus_frag, "frag_count", focus_cut_site)
        figures.append(("Frag Count ECDF by Sample", stats, fig))

        fig = px.ecdf(
            focus_frag,
            x="frag_count",
            color="aliquot",
            color_discrete_map={
                row["aliquot"]: palette.get(str(row["sample_group"]), "#7f7f7f")
                for _, row in focus_frag[["aliquot", "sample_group"]].drop_duplicates().iterrows()
            },
            hover_data=["sample_group", "aliquot", "sv_type"],
            title=f"Frag_count ECDF by aliquot (single panel) ({focus_cut_site})",
        )
        fig.update_xaxes(type="log")
        stats = comparison_stats_text(focus_frag, "frag_count", focus_cut_site)
        figures.append(("Frag Count ECDF by Aliquot", stats, fig))

    mapq_panel = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("mean_mapq1", "mean_mapq2"),
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )
    gs_mapq = group_summary.copy()
    gs_mapq["sample_group"] = pd.Categorical(
        gs_mapq["sample_group"], categories=GROUP_ORDER, ordered=True
    )
    gs_mapq = gs_mapq.sort_values("sample_group")
    for col_idx, metric in enumerate(["mean_mapq1", "mean_mapq2"], start=1):
        m = gs_mapq.dropna(subset=[metric])
        mapq_panel.add_trace(
            go.Bar(
                x=m["sample_group"],
                y=m[metric],
                marker_color=[palette.get(g, "#7f7f7f") for g in m["sample_group"]],
                text=[f"{v:.3f}" for v in m[metric]],
                textposition="outside",
                showlegend=False,
                name=metric,
            ),
            row=1,
            col=col_idx,
        )
        mapq_panel.update_xaxes(
            categoryorder="array",
            categoryarray=GROUP_ORDER,
            row=1,
            col=col_idx,
        )
    yvals = pd.concat(
        [gs_mapq["mean_mapq1"], gs_mapq["mean_mapq2"]], ignore_index=True
    ).dropna()
    if not yvals.empty:
        ymin = max(0.0, float(yvals.min()) - 1.5)
        ymax = float(yvals.max()) + 1.5
        mapq_panel.update_yaxes(range=[ymin, ymax], row=1, col=1)
        mapq_panel.update_yaxes(range=[ymin, ymax], row=1, col=2)
    mapq_panel.update_layout(
        title=f"Diagnosis summary: mapping quality metrics ({focus_cut_site})",
        xaxis_title="sample_group",
        xaxis2_title="sample_group",
        yaxis_title="MAPQ",
    )
    figures.append(("Diagnosis Summary: MAPQ", summary_text, mapq_panel))

    comp_panel = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("tra_frac", "multi_aliquot_frac"),
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )
    gs_comp = group_summary.copy()
    gs_comp = (
        gs_comp.set_index("sample_group")
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    gs_comp["sample_group"] = pd.Categorical(
        gs_comp["sample_group"], categories=GROUP_ORDER, ordered=True
    )
    for col_idx, metric in enumerate(["tra_frac", "multi_aliquot_frac"], start=1):
        m = gs_comp.copy()
        m[metric] = pd.to_numeric(m[metric], errors="coerce").fillna(0.0)
        comp_panel.add_trace(
            go.Bar(
                x=m["sample_group"],
                y=m[metric],
                marker_color=[palette.get(g, "#7f7f7f") for g in m["sample_group"]],
                text=[f"{v:.3f}" for v in m[metric]],
                textposition="outside",
                showlegend=False,
                name=metric,
            ),
            row=1,
            col=col_idx,
        )
        comp_panel.update_xaxes(
            categoryorder="array",
            categoryarray=GROUP_ORDER,
            row=1,
            col=col_idx,
        )
    comp_panel.update_yaxes(range=[0.0, 1.0], row=1, col=1)
    comp_panel.update_yaxes(range=[0.0, 1.0], row=1, col=2)
    comp_panel.update_layout(
        title=f"Diagnosis summary: TRA fraction and recurrence ({focus_cut_site})",
        xaxis_title="sample_group",
        xaxis2_title="sample_group",
        yaxis_title="Fraction",
    )
    figures.append(("Diagnosis Summary: Composition/Recurrence", summary_text, comp_panel))

    stats = comparison_stats_text(focus_clonality, "hhi_frag", focus_cut_site)
    fig = px.box(
        focus_clonality,
        x="sample_group",
        y="hhi_frag",
        color="sample_group",
        color_discrete_map=palette,
        points="all",
        category_orders={"sample_group": GROUP_ORDER},
        hover_data=["sample", "aliquot", "top1_frag_share", "top5_frag_share", "effective_event_number"],
        title=f"Clonality by aliquot: HHI of fragment support ({focus_cut_site})",
    )
    hhi_text = (
        stats
        + " | HHI interpretation: closer to 1 means support concentrated in a few events; "
        + "closer to 0 means diffuse support. effective_event_number is approximately 1/HHI."
    )
    figures.append(("Clonality (HHI: higher means more concentrated)", hhi_text, fig))

    stats = comparison_stats_text(focus_orientation, "forward1_frac", focus_cut_site)
    strand_long = focus_orientation.melt(
        id_vars=["sample_group", "sample", "aliquot", "event_key", "read_rows"],
        value_vars=["forward1_frac", "forward2_frac"],
        var_name="metric",
        value_name="forward_fraction",
    )
    fig = px.violin(
        strand_long,
        x="sample_group",
        y="forward_fraction",
        color="sample_group",
        color_discrete_map=palette,
        category_orders={"sample_group": GROUP_ORDER},
        facet_col="metric",
        box=True,
        points=False,
        hover_data=["sample", "aliquot", "read_rows"],
        title=f"Per-event forward strand fraction from read-level alignments ({focus_cut_site})",
    )
    figures.append(("Forward Strand Fraction", stats, fig))

    stats = comparison_stats_text(focus_master, "junction_complexity", focus_cut_site)
    junc_long = focus_master.melt(
        id_vars=["sample_group", "sample", "aliquot", "sv_type"],
        value_vars=["mh_length", "ins_length", "mh_entropy", "ins_entropy", "junction_complexity"],
        var_name="metric",
        value_name="value",
    )
    fig = px.violin(
        junc_long,
        x="sample_group",
        y="value",
        color="sample_group",
        color_discrete_map=palette,
        category_orders={"sample_group": GROUP_ORDER},
        facet_col="metric",
        box=True,
        points=False,
        title=f"Junction complexity metrics ({focus_cut_site})",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "")))
    figures.append(("Junction Complexity", stats, fig))

    stats = comparison_stats_text(focus_master, "abs_mapq_delta", focus_cut_site)
    sym_long = focus_master.melt(
        id_vars=["sample_group", "sample", "aliquot", "sv_type"],
        value_vars=["mapq_delta", "match_delta", "abs_mapq_delta", "abs_match_delta"],
        var_name="metric",
        value_name="value",
    )
    fig = px.violin(
        sym_long,
        x="sample_group",
        y="value",
        color="sample_group",
        color_discrete_map=palette,
        category_orders={"sample_group": GROUP_ORDER},
        facet_col="metric",
        box=True,
        points=False,
        title=f"Alignment symmetry metrics ({focus_cut_site})",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", "")))
    figures.append(("Symmetry Metrics", stats, fig))

    stats = comparison_stats_text(focus_burden, "events_per_1k_reads", focus_cut_site)
    fig = px.scatter(
        focus_burden,
        x="annotated_rows",
        y="unique_events",
        color="sample_group",
        color_discrete_map=palette,
        symbol="sample",
        hover_data=["aliquot", "events_per_1k_reads", "mean_frag_count", "singleton_frac"],
        title=f"Unique events vs annotated rows ({focus_cut_site})",
    )
    figures.append(("Burden Context", stats, fig))

    stats = comparison_stats_text(focus_master, "read_frag_ratio", focus_cut_site)
    fig = px.violin(
        focus_master,
        x="sample_group",
        y="read_frag_ratio",
        color="sample_group",
        color_discrete_map=palette,
        category_orders={"sample_group": GROUP_ORDER},
        box=True,
        points=False,
        hover_data=["sample", "aliquot", "sv_type", "read_count", "frag_count"],
        title=f"Per-event read_count / frag_count ({focus_cut_site})",
    )
    figures.append(("Read/Frag Ratio", stats, fig))

    report_path = outdir / f"sv_diagnosis_report_{focus_cut_site}.html"
    toc_items = [(f"sec-{i+1}", title) for i, (title, _, _) in enumerate(figures)]
    blocks: list[str] = [
        "<html><head><meta charset='utf-8'>",
        "<title>SV Diagnosis Report</title>",
        "<style>"
        "html{scroll-behavior:smooth;}"
        "body{font-family:Arial,sans-serif;margin:0;background:#0b1220;color:#e5e7eb;}"
        ".layout{max-width:1500px;margin:0 auto;padding:20px 24px;}"
        ".side{position:fixed;left:24px;top:16px;z-index:1000;width:240px;background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px;max-height:calc(100vh - 32px);overflow:auto;}"
        ".side h3{margin:0 0 10px 0;font-size:16px;color:#f3f4f6;}"
        ".side ul{list-style:none;padding:0;margin:0;}"
        ".side li{margin:0 0 8px 0;}"
        ".side a{color:#93c5fd;text-decoration:none;font-size:13px;line-height:1.35;display:block;}"
        ".side a:hover{color:#bfdbfe;text-decoration:underline;}"
        ".main{max-width:1040px;margin-left:300px;}"
        "h1{margin:0 0 8px 0;color:#f9fafb;} h2{margin:0 0 8px 0;color:#f3f4f6;}"
        ".subtitle{margin:0 0 16px 0;color:#9ca3af;}"
        ".section{margin:24px 0 44px;padding:14px;background:#111827;border:1px solid #1f2937;border-radius:12px;}"
        ".stats{background:#0f172a;padding:8px 10px;border-radius:6px;font-family:monospace;display:inline-block;color:#cbd5e1;border:1px solid #334155;}"
        ".intro{margin:8px 0 12px;color:#cbd5e1;line-height:1.45;}"
        "@media (max-width:1100px){.side{position:static;width:auto;max-height:none;margin-bottom:14px;} .main{margin-left:0;}}"
        "</style>",
        "</head><body>",
        "<div class='layout'>",
        "<aside class='side'>",
        "<h3>Sections</h3>",
        "<ul>",
        *[
            f"<li><a href='#{html.escape(sec_id)}'>{html.escape(title)}</a></li>"
            for sec_id, title in toc_items
        ],
        "</ul>",
        "</aside>",
        "<main class='main'>",
        f"<h1>SV Diagnosis Report: {html.escape(focus_cut_site)}</h1>",
        "<p class='subtitle'>Comparisons emphasize no_cut vs inner_cut where available.</p>",
    ]
    include_plotlyjs = "cdn"
    intro_text = {
        "Diagnosis Summary: Support (Mean Frag Count)": "We are comparing mean fragment count of each aliquot, summarized by sample group and shown with aliquot-level spread.",
        "Diagnosis Summary: Mean Frag Count by Sample": "Per-sample distribution of aliquot-level mean fragment support. Box shows sample spread; points are individual aliquots.",
        "Diagnosis Summary: Support (Singleton Fraction)": "Fraction of events supported by exactly one fragment. Higher values indicate more singleton-dominated profiles.",
        "Diagnosis Summary: Singleton Fraction by Sample": "Per-sample distribution of aliquot-level singleton burden. Box shows sample spread; points are individual aliquots.",
        "Singleton Sensitivity by Aliquot Depth": "Sensitivity analysis of singleton fraction after filtering out low-event aliquots. This helps separate true group shifts from low-depth inflation.",
        "Frag Count Distribution (Event-level ECDF)": "Cumulative event-level frag_count distribution by group. Separation indicates differences in support profile shape, not just mean.",
        "Frag Count ECDF by Sample": "Event-level frag_count ECDF curves overlaid in one panel (one curve per sample, colored by sample group) for direct cross-sample comparison.",
        "Frag Count ECDF by Aliquot": "Event-level frag_count ECDF curves overlaid in one panel (one curve per aliquot, colored by sample group) for direct cross-aliquot comparison.",
        "Diagnosis Summary: MAPQ": "Mean mapping quality on both break-end alignments. Similar profiles suggest no obvious mapping-quality driven bias.",
        "Diagnosis Summary: Composition/Recurrence": "TRA prevalence and fraction of event keys recurring across aliquots within a group.",
        "Clonality (HHI: higher means more concentrated)": "HHI summarizes concentration of fragment support across events per aliquot. Larger HHI means a few events dominate.",
        "Forward Strand Fraction": "Per-event forward orientation fraction from read-level alignments, summarized across groups.",
        "Junction Complexity": "Junction sequence properties from lengths and sequence entropy to assess complexity of repair signatures.",
        "Symmetry Metrics": "Differences between end1 and end2 alignment quality/length metrics; near zero implies more symmetric alignments.",
        "Burden Context": "Unique event burden versus read depth to assess whether group differences track sequencing depth.",
        "Read/Frag Ratio": "Per-event ratio of read support to fragment support; elevated values can reflect duplication or amplification effects.",
    }
    for idx, (title, stats_text, fig) in enumerate(figures, start=1):
        section_id = f"sec-{idx}"
        fig_title = fig.layout.title.text or title
        intro = intro_text.get(title, "Diagnostic view for this metric at the selected cut site.")
        fig.update_layout(
            template="plotly_dark",
            font={"size": PLOT_BASE_FONT_SIZE},
            title={"text": fig_title, "font": {"size": PLOT_TITLE_FONT_SIZE}},
            legend={
                "font": {"size": PLOT_LEGEND_FONT_SIZE},
                "title": {"font": {"size": PLOT_LEGEND_TITLE_FONT_SIZE}},
            },
            hoverlabel={"font_size": PLOT_HOVER_FONT_SIZE},
            paper_bgcolor="#111827",
            plot_bgcolor="#0f172a",
            height={
                "Diagnosis Summary: Support (Mean Frag Count)": 620,
                "Diagnosis Summary: Mean Frag Count by Sample": 560,
                "Diagnosis Summary: Support (Singleton Fraction)": 620,
                "Diagnosis Summary: Singleton Fraction by Sample": 560,
                "Singleton Sensitivity by Aliquot Depth": 560,
                "Frag Count Distribution (Event-level ECDF)": 560,
                "Frag Count ECDF by Sample": 760,
                "Frag Count ECDF by Aliquot": 900,
                "Diagnosis Summary: MAPQ": 560,
                "Diagnosis Summary: Composition/Recurrence": 560,
                "Clonality (HHI: higher means more concentrated)": 560,
                "Forward Strand Fraction": 620,
                "Junction Complexity": 760,
                "Symmetry Metrics": 760,
                "Burden Context": 560,
                "Read/Frag Ratio": 620,
            }.get(title, 600),
        )
        fig.update_xaxes(
            title_font={"size": PLOT_AXIS_TITLE_FONT_SIZE},
            tickfont={"size": PLOT_TICK_FONT_SIZE},
        )
        fig.update_yaxes(
            title_font={"size": PLOT_AXIS_TITLE_FONT_SIZE},
            tickfont={"size": PLOT_TICK_FONT_SIZE},
        )
        fig_html = pio.to_html(
            fig,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            auto_play=False,
        )
        include_plotlyjs = False
        blocks.append(f"<div class='section' id='{html.escape(section_id)}'>")
        blocks.append(f"<h2>{html.escape(title)}</h2>")
        blocks.append(f"<p class='intro'>{html.escape(intro)}</p>")
        blocks.append(f"<div class='stats'>{html.escape(stats_text)}</div>")
        blocks.append(fig_html)
        blocks.append("</div>")
    blocks.append("</main>")
    blocks.append("</div>")
    blocks.append("</body></html>")
    html_text = "\n".join(blocks)
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote consolidated plot report: {report_path}")


#%%
print("Loading aggregated event-level tables...")
master = load_aggregated_events(EVENTS_DIR)
master = add_cut_distance(master)

print("Summarizing annotated read-level tables by aliquot...")
ann_summary = summarize_annotated_by_aliquot(ANNOTATED_DIR)
orientation = load_read_level_event_orientation(ANNOTATED_DIR)

print("Building step 1 master joined table...")
master_joined = master.merge(
    ann_summary,
    on=["sample", "aliquot", "cut_site", "sample_group", "cut_site_label"],
    how="left",
    suffixes=("", "_annotated"),
)

print("Running steps 2-8 summaries...")
burden, burden_group = step2_burden(master, ann_summary)
support = step3_support(master)
align_event, align_read = step4_alignment(master, ann_summary)
sv_type, clip_combo, partner_chr = step5_structure(master)
junction = step6_junction(master)
recurrence, recurrence_summary, cross_summary = step7_recurrence(master)
outliers = step8_outliers(master, burden)

if not NO_PLOTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Writing interactive plots...")
    write_interactive_plots(
        outdir=OUTPUT_DIR,
        master=master,
        burden=burden,
        orientation=orientation,
        focus_cut_site=FOCUS_CUT_SITE,
    )
else:
    print("Skipping plots (NO_PLOTS=True).")

print("Done.")

# %%
