#!/usr/bin/env python3
"""CLI wrapper for deep_scar_seq_toolkit.annotate."""

import argparse

from deep_scar_seq_toolkit.annotate import annotate_bam
from deep_scar_seq_toolkit.config import (
    DEFAULT_INCLUDE_MATE,
    DEFAULT_PYSAM_QUIET,
    DEFAULT_UMI_TAG_PRIORITY,
    DEFAULT_WINDOW,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan split reads around a breakpoint and annotate SV features."
    )
    parser.add_argument("--bam", required=True, help="Path to input BAM file.")
    parser.add_argument("--chrom", required=True, help="Chromosome to scan (e.g., chr2).")
    parser.add_argument(
        "--breakpoint",
        type=int,
        required=True,
        help="Breakpoint position (0-based, half-open).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Search window around breakpoint (bp).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output file path (TSV). If no .tsv suffix is provided, it will be appended.",
    )
    parser.add_argument(
        "--pysam_quiet",
        action="store_true",
        default=DEFAULT_PYSAM_QUIET,
        help="Silence pysam/htslib warnings (e.g., BAM index timestamps).",
    )
    parser.add_argument(
        "--include_mate",
        action="store_true",
        default=DEFAULT_INCLUDE_MATE,
        help="Include mate read alignment fields (slower).",
    )
    parser.add_argument(
        "--umi-tags",
        default=",".join(DEFAULT_UMI_TAG_PRIORITY),
        help=(
            "Comma-separated UMI tag priority (e.g., RX,UR). "
            f"Default: {','.join(DEFAULT_UMI_TAG_PRIORITY)}"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    umi_tag_priority = tuple(
        tag.strip().upper() for tag in args.umi_tags.split(",") if tag.strip()
    )
    annotated = annotate_bam(
        args.bam,
        args.chrom,
        args.breakpoint,
        window=args.window,
        include_mate=args.include_mate,
        pysam_quiet=args.pysam_quiet,
        umi_tag_priority=umi_tag_priority,
    )
    if annotated.empty:
        print("No split reads found near the specified breakpoint.")
        return
    out_path = args.out if args.out.endswith(".tsv") else f"{args.out}.tsv"
    annotated.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(annotated)} rows to {out_path}")
    print(annotated["sv_type"].value_counts())


if __name__ == "__main__":
    main()
