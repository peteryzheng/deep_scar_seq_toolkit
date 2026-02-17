#!/usr/bin/env python3
"""Aggregate annotated SV TSVs into unique event counts."""

import warnings

import pandas as pd

DEFAULT_MIN_MAPQ = 5
DEFAULT_MIN_MATCH = 30
DEFAULT_MAX_EXTRA_BASES = 0
EXPECTED_EVENT_COLUMNS = [
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
    "largest_match_length1",
    "largest_match_length2",
    "largest_clip_length1",
    "largest_clip_length2",
    "is_supplementary1",
    "extra_bases1",
    "extra_bases2",
]
OPTIONAL_UMI_COLUMNS = [
    "umi",
    "umi_source_tag",
    "umi_missing",
    "umi_mi",
]
OPTIONAL_DUPLICATE_COLUMNS = [
    "is_duplicate",
    "is_duplicate_raw",
    "duplicate_source",
]
READ_EVENT_COLUMNS = EXPECTED_EVENT_COLUMNS + OPTIONAL_UMI_COLUMNS + OPTIONAL_DUPLICATE_COLUMNS


def load_events(events):
    missing = [col for col in EXPECTED_EVENT_COLUMNS if col not in events.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing expected columns: {missing_str}")
    selected_columns = EXPECTED_EVENT_COLUMNS + [
        col
        for col in OPTIONAL_UMI_COLUMNS + OPTIONAL_DUPLICATE_COLUMNS
        if col in events.columns
    ]
    return events[selected_columns].copy()


def _coerce_bool(series):
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def clean_events(events):
    events["sv_type"] = events["sv_type"].fillna("NA")
    events["mh_seq"] = events["mh_seq"].fillna("")
    events["ins_seq"] = events["ins_seq"].fillna("")
    events["mh_length"] = pd.to_numeric(events["mh_length"], errors="coerce").fillna(0).astype(int)
    events["ins_length"] = pd.to_numeric(events["ins_length"], errors="coerce").fillna(0).astype(int)
    for col in [
        "mapping_quality1",
        "mapping_quality2",
        "largest_match_length1",
        "largest_match_length2",
        "extra_bases1",
        "extra_bases2",
    ]:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    if "umi" not in events.columns:
        events["umi"] = ""
    else:
        events["umi"] = events["umi"].fillna("").astype(str)

    if "umi_source_tag" not in events.columns:
        events["umi_source_tag"] = ""
    else:
        events["umi_source_tag"] = events["umi_source_tag"].fillna("").astype(str)

    if "umi_missing" not in events.columns:
        events["umi_missing"] = events["umi"] == ""
    else:
        events["umi_missing"] = _coerce_bool(events["umi_missing"]) | (events["umi"] == "")

    if "umi_mi" not in events.columns:
        events["umi_mi"] = ""
    else:
        events["umi_mi"] = events["umi_mi"].fillna("").astype(str)

    if "is_duplicate" not in events.columns:
        events["is_duplicate"] = False
    else:
        events["is_duplicate"] = _coerce_bool(events["is_duplicate"])

    if "is_duplicate_raw" not in events.columns:
        events["is_duplicate_raw"] = events["is_duplicate"]
    else:
        events["is_duplicate_raw"] = _coerce_bool(events["is_duplicate_raw"])

    if "duplicate_source" not in events.columns:
        events["duplicate_source"] = ""
    else:
        events["duplicate_source"] = events["duplicate_source"].fillna("").astype(str)

    return events


def filter_events(events, min_mapq, min_match_length, max_extra_bases):
    return events[
        (events["mapping_quality1"] >= min_mapq)
        & (events["mapping_quality2"] >= min_mapq)
        & (events["largest_match_length1"] >= min_match_length)
        & (events["largest_match_length2"] >= min_match_length)
        & (events["extra_bases1"] <= max_extra_bases)
        & (events["extra_bases2"] <= max_extra_bases)
        & (~events["is_duplicate"])
    ]


def aggregate_events(events):
    def count_non_empty_unique(values):
        values = values.fillna("")
        return int(values[values != ""].nunique())

    unique_events = (
        events.fillna("")
        .groupby(
            [
                "chr1",
                "breakpoint1",
                "chr2",
                "breakpoint2",
                "clip_side1",
                "clip_side2",
                "sv_type",
                "mh_seq",
                "ins_seq",
            ],
            dropna=False,
        )
        .agg(
            read_count=("read_name", "count"),
            frag_count=("read_name", lambda x: x.nunique()),
            umi_count=("umi", count_non_empty_unique),
            umi_missing_read_count=("umi_missing", "sum"),
            mapq1_mean=("mapping_quality1", "mean"),
            match1_mean=("largest_match_length1", "mean"),
            mapq2_mean=("mapping_quality2", "mean"),
            match2_mean=("largest_match_length2", "mean"),
            mh_length=("mh_length", "first"),
            ins_length=("ins_length", "first"),
        )
        .reset_index()
    )
    unique_events["umi_missing_read_count"] = (
        pd.to_numeric(unique_events["umi_missing_read_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    unique_events["umi_coverage"] = (
        1 - (unique_events["umi_missing_read_count"] / unique_events["read_count"])
    ).clip(lower=0, upper=1)
    return unique_events


def aggregate_reads(
    events,
    min_mapq=DEFAULT_MIN_MAPQ,
    min_match_length=DEFAULT_MIN_MATCH,
    max_extra_bases=DEFAULT_MAX_EXTRA_BASES,
):
    events = load_events(events)
    missing_duplicate_columns = [
        col for col in OPTIONAL_DUPLICATE_COLUMNS if col not in events.columns
    ]
    if missing_duplicate_columns:
        warnings.warn(
            "Legacy annotated TSV: duplicate-aware filtering metadata missing; relying on upstream deduplication.",
            RuntimeWarning,
            stacklevel=2,
        )
    events = clean_events(events)
    events = filter_events(events, min_mapq, min_match_length, max_extra_bases)
    return aggregate_events(events)
