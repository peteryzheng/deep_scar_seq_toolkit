#!/usr/bin/env python3
"""CLI wrapper for deep_scar_seq_toolkit.aggregate."""

import argparse

import pandas as pd

from deep_scar_seq_toolkit.aggregate import (
    DEFAULT_MAX_EXTRA_BASES,
    DEFAULT_MIN_MAPQ,
    DEFAULT_MIN_MATCH,
    READ_EVENT_COLUMNS,
    aggregate_reads,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate an annotated SV TSV into unique event counts."
    )
    parser.add_argument(
        "--annotated",
        required=True,
        help="Path to annotated TSV produced by annotate_svs.py.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output aggregated TSV path.",
    )
    parser.add_argument(
        "--min_mapq",
        type=int,
        default=DEFAULT_MIN_MAPQ,
        help=f"Minimum MAPQ on both sides (default: {DEFAULT_MIN_MAPQ}).",
    )
    parser.add_argument(
        "--min_match_length",
        type=int,
        default=DEFAULT_MIN_MATCH,
        help=f"Minimum largest match length on both sides (default: {DEFAULT_MIN_MATCH}).",
    )
    parser.add_argument(
        "--max_extra_bases",
        type=int,
        default=DEFAULT_MAX_EXTRA_BASES,
        help=(
            "Maximum allowed extra bases on both sides "
            f"(default: {DEFAULT_MAX_EXTRA_BASES})."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    events = pd.read_csv(
        args.annotated,
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
            "is_duplicate": "string",
            "is_duplicate_raw": "string",
            "duplicate_source": "string",
        },
    )
    aggregated = aggregate_reads(
        events, args.min_mapq, args.min_match_length, args.max_extra_bases
    )
    out_path = args.out if args.out.endswith(".tsv") else f"{args.out}.tsv"
    aggregated.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(aggregated)} aggregated events to {out_path}")


if __name__ == "__main__":
    main()
