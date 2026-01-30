#!/usr/bin/env python3
import csv
from typing import Dict, Iterable, List, Tuple

import pysam

#%%
def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]


def build_query_match(query_seq: str, match_len: int, clip_side: str) -> str:
    if match_len <= 0:
        return ""
    if clip_side == "right":
        return query_seq[:match_len]
    return query_seq[-match_len:]

def ref_downstream(fetcher: pysam.FastaFile, chrom: str, breakpoint: int, length: int) -> str:
    if length <= 0:
        return ""
    start = max(0, breakpoint)
    end = max(start, breakpoint + length)
    return fetcher.fetch(chrom, start, end).upper()


def ref_upstream(fetcher: pysam.FastaFile, chrom: str, breakpoint: int, length: int) -> str:
    if length <= 0:
        return ""
    end = max(0, breakpoint)
    start = max(0, breakpoint - length)
    return fetcher.fetch(chrom, start, end).upper()


def fetch_ref_side(
    fetcher: pysam.FastaFile,
    chrom: str,
    breakpoint: int,
    match_len: int,
    clip_side: str,
) -> str:
    if clip_side == "right":
        return ref_upstream(fetcher, chrom, breakpoint, match_len)
    return ref_downstream(fetcher, chrom, breakpoint, match_len)


def place_segment(line: List[str], start: int, segment: str) -> None:
    for i, base in enumerate(segment):
        idx = start + i
        if 0 <= idx < len(line):
            line[idx] = base


def build_visual_line(total_len: int, segment: str, side: str) -> str:
    line = [" "] * total_len
    if side == "left":
        start = max(0, total_len - len(segment))
    else:
        start = 0
    place_segment(line, start, segment)
    return "".join(line)


def iter_rows(tsv_path: str, max_rows: int) -> Iterable[Dict[str, str]]:
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            yield row


def ref_matches_by_rule(
    fetcher: pysam.FastaFile,
    chr1: str,
    chr2: str,
    clip_side1: str,
    clip_side2: str,
    breakpoint1: int,
    breakpoint2: int,
    match_len1: int,
    match_len2: int,
) -> Tuple[str, str]:
    ref_match1 = fetch_ref_side(fetcher, chr1, breakpoint1, match_len1, clip_side1)
    ref_match2 = fetch_ref_side(fetcher, chr2, breakpoint2, match_len2, clip_side2)
    if clip_side1 == clip_side2:
        ref_match2 = reverse_complement(ref_match2)
    return ref_match1, ref_match2


def visualize_rows(
    tsv_path: str = "read_level_sv_events.tsv",
    fasta_path: str = "/Users/youyun/igv/genomes/seq/hg19.fasta",
    max_rows: int = 38,
) -> None:
    fetcher = pysam.FastaFile(fasta_path)
    for row in iter_rows(tsv_path, max_rows):
        # if they are not in the major chromosomes, skip for visualization
        if row["chr1"] not in fetcher.references or row["chr2"] not in fetcher.references:
            continue
        match_len1 = int(row["largest_match_length1"])
        match_len2 = int(row["largest_match_length2"])
        mh_len = int(row["mh_length"])
        ins_seq = row["ins_seq"]
        clip_side1 = row["clip_side1"]
        clip_side2 = row["clip_side2"]
        is_forward1 = row["is_forward1"] == "True"
        is_forward2 = row["is_forward2"] == "True"
        supp1 = row["is_supplementary1"] == "True"
        
        # query sequence is obtained from the primary alignment to avoid clipping issues
        query_seq = row["query_sequence"]
        # if primary is aln2, orient query seq relative to the target location
        if supp1 and is_forward1 != is_forward2:
            query_seq = reverse_complement(query_seq)
        # the clip side for target location side can be taken as it because we obtain forward strand sequence there
        query_match1_raw = build_query_match(query_seq, match_len1, clip_side1)
        # the other clip side needs to be flipped compared to clip side 1
        query_match2_raw = build_query_match(
            query_seq, match_len2, 
            "left" if clip_side1 == "right" else "right"
        )

        # breakpoint 1's clip side determines the visualization order below
        ref_match1, ref_match2 = ref_matches_by_rule(
            fetcher,
            row["chr1"],
            row["chr2"],
            clip_side1,
            clip_side2,
            int(row["breakpoint1"]),
            int(row["breakpoint2"]),
            match_len1,
            match_len2,
        )

        viz_clip1 = clip_side1
        # since we visualize the two ref sequences from the positive strand of the target location (is_forward1)
        # we will rc the second ref sequence if both clip sides are the same
        viz_clip2 = clip_side2 if clip_side1 != clip_side2 else ("left" if clip_side1 == "right" else "right")

        total_len = len(query_seq)
        ref1_line = build_visual_line(total_len, ref_match1, viz_clip1)
        ref2_line = build_visual_line(total_len, ref_match2, viz_clip2)
        targ_line = build_visual_line(total_len, query_match1_raw, viz_clip1)
        _line = build_visual_line(total_len, query_match2_raw, viz_clip2)

        print(f"read: {row['read_name']}, sv_type: {row['sv_type']}")
        print(f"aln type 1: {'supplementary' if supp1 else 'primary'}, aln type 2: {'primary' if supp1 else 'supplementary'}")
        print(f"breakpoints: {row['chr1']}:{row['breakpoint1']}{clip_side1} "
              f"<-> {row['chr2']}:{row['breakpoint2']}{clip_side2}")
        print(f"mh len: {mh_len}bp, ins len: {len(ins_seq)}bp")
        print(f"mh seq: {row['mh_seq']}, ins seq: {ins_seq}")
        print(f"clip_pair: {clip_side1}/{clip_side2}")
        print(
            "aln1 CIGAR: "
            f"{row['cigar1']} (is_forward1={row['is_forward1']}), "
            "aln2 CIGAR: "
            f"{row['cigar2']} (is_forward2={row['is_forward2']})"
        )
        print(f"ref1: {ref1_line}")
        print(f"ref2: {ref2_line}")
        print(f"targ: {targ_line}")
        print(f"mate: {mate_line}")
        print("")
    fetcher.close()

#%%
def main() -> None:
    visualize_rows()


if __name__ == "__main__":
    main()
