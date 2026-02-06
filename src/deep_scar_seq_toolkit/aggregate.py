#!/usr/bin/env python3
"""Aggregate annotated SV TSVs into unique event counts."""

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


def load_events(events):
    missing = [col for col in EXPECTED_EVENT_COLUMNS if col not in events.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing expected columns: {missing_str}")
    return events[EXPECTED_EVENT_COLUMNS].copy()


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
    return events


def filter_events(events, min_mapq, min_match_length, max_extra_bases):
    return events[
        (events["mapping_quality1"] >= min_mapq)
        & (events["mapping_quality2"] >= min_mapq)
        & (events["largest_match_length1"] >= min_match_length)
        & (events["largest_match_length2"] >= min_match_length)
        & (events["extra_bases1"] <= max_extra_bases)
        & (events["extra_bases2"] <= max_extra_bases)
    ]


def aggregate_events(events):
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
            mapq1_mean=("mapping_quality1", "mean"),
            match1_mean=("largest_match_length1", "mean"),
            mapq2_mean=("mapping_quality2", "mean"),
            match2_mean=("largest_match_length2", "mean"),
            mh_length=("mh_length", "first"),
            ins_length=("ins_length", "first"),
        )
        .reset_index()
    )
    return unique_events


def aggregate_reads(
    events,
    min_mapq=DEFAULT_MIN_MAPQ,
    min_match_length=DEFAULT_MIN_MATCH,
    max_extra_bases=DEFAULT_MAX_EXTRA_BASES,
):
    events = load_events(events)
    events = clean_events(events)
    events = filter_events(events, min_mapq, min_match_length, max_extra_bases)
    return aggregate_events(events)
