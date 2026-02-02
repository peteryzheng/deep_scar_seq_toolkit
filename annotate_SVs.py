#!/usr/bin/env python3
# %%
# Coordinate convention: 0-based, half-open for all internal positions.
import argparse
import pysam
import re
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# local vs eristwo
if os.path.expanduser('~') in ["/Users/youyun", "/Users/youyunzheng"]: 
    # in a local mac, the home directory is usuaully at '/Users/[username]'
    workdir = os.path.expanduser('~')+"/Documents/HMS/PhD/beroukhimlab/dfci_mount/"
else:
    # in eristwo, the home directory is usuaully at '/home/unix/[username]'
    workdir = "/data/beroukhim1/"

# %%
def parse_args():
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
        default=5,
        help="Search window around breakpoint (bp).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output file path (TSV). If no .tsv suffix is provided, it will be appended.",
    )
    parser.add_argument(
        "--pysam_quiet",
        action="store_true",
        help="Silence pysam/htslib warnings (e.g., BAM index timestamps).",
    )
    parser.add_argument(
        "--include_mate",
        action="store_true",
        help="Include mate read alignment fields (slower).",
    )
    return parser.parse_args()

# %%
def is_split(aln):
    # this is checking if an alignment is split by looking at the SA tag
    # this alignment can be primary or supplementary
    return ((aln.has_tag('SA')) and (not aln.is_secondary))

def parse_cigar_tuples(cigar_str):
    op_map = {
        "M": 0,
        "I": 1,
        "D": 2,
        "N": 3,
        "S": 4,
        "H": 5,
        "P": 6,
        "=": 7,
        "X": 8,
    }
    nums = []
    tuples = []
    for ch in cigar_str:
        if ch.isdigit():
            nums.append(ch)
        else:
            if not nums:
                continue
            length = int("".join(nums))
            nums = []
            tuples.append((op_map[ch], length))
    return tuples


def clip_breakpoint(cigar_tuples, ref_start):
    """Find the largest soft/hard clip (C) and largest match (M) in the CIGAR, and determine if C is left or right of M."""
    cigar = cigar_tuples
    # find the tuples for clipped sequences and matched sequences
    # CIGAR codes: 0=M, 4=S, 5=H
    clip_indices = [(i, l) for i, (op, l) in enumerate(cigar) if op in [4, 5]]
    match_indices = [(i, l) for i, (op, l) in enumerate(cigar) if op == 0]
    if not clip_indices or not match_indices:
        return None
    # Find largest clip and match sequences
    largest_clip = max(clip_indices, key=lambda x: x[1])
    largest_match = max(match_indices, key=lambda x: x[1])
    # Determine if clip is left or right of match
    clip_pos = largest_clip[0]
    match_pos = largest_match[0]
    # if left, then the breakpoint is at the left side of the alignment which is reference_start
    if clip_pos < match_pos:
        clip_side = 'left'
        breakpoint = ref_start
    # if right, then the breakpoint is at the right side of the alignment which is reference_end
    else:
        clip_side = 'right'
        ref_consuming_ops = {0, 2, 3, 7, 8}  # M, D, N, =, X
        ref_len = sum(l for op, l in cigar if op in ref_consuming_ops)
        breakpoint = ref_start + ref_len
    total_bases = sum([l for _, l in cigar])
    # extra_bases flags CIGARs that deviate from a simple "one match + one clip" model.
    return {
        "largest_clip_length": largest_clip[1],
        "largest_match_length": largest_match[1],
        "clip_side": clip_side,
        "breakpoint": breakpoint,
        "extra_bases": total_bases - largest_clip[1] - largest_match[1],
    }

def extract_aln_core(aln):
    return {
        "read_name": aln.query_name,
        "chr": aln.reference_name,
        "start": aln.reference_start,
        "cigar": aln.cigarstring,
        "read_number": aln.is_read1,
        "is_forward": aln.is_forward,
        "insert_size": aln.template_length,
        "is_supplementary": aln.is_supplementary,
        "mapping_quality": aln.mapping_quality,
        "query_sequence": aln.query_sequence,
    }

def parse_sa_tag(sa_tag):
    # SA:Z:chr,pos,strand,CIGAR,mapQ,NM;
    entry = sa_tag.split(";")[0]
    fields = entry.split(",")
    if len(fields) < 6:
        return None
    return {
        "chr": fields[0],
        "start": int(fields[1]) - 1,  # SA is 1-based
        "is_forward": fields[2] == "+",
        "cigar": fields[3],
        "mapping_quality": int(fields[4]),
    }


def find_other_alignment(sa_tag, query_name, is_read1, is_supplementary, bam):
    """Find the other alignment based on the SA tag information."""
    # if the alignment is primary, look for supplementary
    # if the alignment is supplementary, look for primary
    # SA tags are 1-based; convert to 0-based for fetch and comparisons.
    sa_fields = sa_tag.split(',')
    sa_chrom = sa_fields[0]
    sa_pos = int(sa_fields[1]) - 1  # Convert to 0-based
    sa_strand = sa_fields[2]
    
    for aln in bam.fetch(region=f"{sa_chrom}:{sa_pos}-{sa_pos+1}"):
        if (aln.query_name == query_name and
            aln.is_read1 == is_read1 and
            (aln.is_supplementary != is_supplementary)):
            return aln
    return None

def run_pipeline(bam_path, chrom, breakpoint, window, include_mate):
    bam = pysam.AlignmentFile(bam_path, "rb")
    region = f"{chrom}:{breakpoint}-{breakpoint}"

    # Count split reads and collect alignments
    alns = []
    for aln in tqdm(bam.fetch(region=region), desc="Scanning split reads"):
        # only check split reads
        if is_split(aln):
            # check breakpoint info of the split read
            bp1_info = clip_breakpoint(aln.cigartuples, aln.reference_start)
            # make sure breakpoint is within window of the pre-defined location
            if abs(bp1_info["breakpoint"] - breakpoint) <= window:
                new_aln_dict = {
                    **extract_aln_core(aln),
                    **bp1_info,
                    # keep track if the read is read1 or read2
                    "read_number": aln.is_read1,
                    "aln_obj": aln,
                }

                # add 1 to the end to indicate breakpoint 1
                new_aln_dict_1 = {k + "1": v for k, v in new_aln_dict.items()}
                alns.append(new_aln_dict_1)
    # catch if alns is empty
    if not alns:
        return alns

    sa_time = 0.0
    mate_time = 0.0
    for aln in tqdm(alns, desc="Annotating other alignment of the breakpoint"):
        aln_obj1 = aln["aln_obj1"]
        if aln_obj1.has_tag("SA"):
            # use the SA tag to find the other alignment of the split read
            # if the current alignment is primary, the other should be supplementary, and vice versa
            sa_tag = aln_obj1.get_tag("SA")
            sa_info = parse_sa_tag(sa_tag)
            if sa_info is None:
                continue
            cigar2_tuples = parse_cigar_tuples(sa_info["cigar"])
            bp2_info = clip_breakpoint(cigar2_tuples, sa_info["start"])
            new_aln_dict_2 = {
                "chr": sa_info["chr"],
                "start": sa_info["start"],
                "is_forward": sa_info["is_forward"],
                "cigar": sa_info["cigar"],
                "mapping_quality": sa_info["mapping_quality"],
                **bp2_info,
            }
            # add 2 to the end to indicate breakpoint 2
            new_aln_dict_2 = {k + "2": v for k, v in new_aln_dict_2.items()}
            aln.update(new_aln_dict_2)
            # only search for other alignment when the current is supplementary
            if aln_obj1.is_supplementary:
                t0 = pd.Timestamp.now()
                other_aln = find_other_alignment(
                    sa_tag,
                    aln["read_name1"],
                    aln["read_number1"],
                    aln["is_supplementary1"],
                    bam,
                )
                # replace the query sequence with the one from the other alignment
                # because the supplementary alignment may be hard clipped
                if other_aln is not None:
                    aln["query_sequence1"] = other_aln.query_sequence
                sa_time += (pd.Timestamp.now() - t0).total_seconds()
        if include_mate:
            # Find mate
            try:
                t1 = pd.Timestamp.now()
                mate = bam.mate(aln["aln_obj1"])
                mate_aln_info = extract_aln_core(mate)
                mate_aln_dict = {k + "_mate": v for k, v in mate_aln_info.items()}
                mate_aln_dict["aln_obj_mate"] = mate
                aln.update(mate_aln_dict)
            except ValueError as e:
                print(f"Mate not found: {e}")
            mate_time += (pd.Timestamp.now() - t1).total_seconds()

    print(f"annotate breakdown | SA: {sa_time:.2f}s | mate: {mate_time:.2f}s")
    return alns

# %%
def build_filtered_df(alns, include_mate):
    # Build a dataframe directly from alns (no filtered_alns list)
    columns = [
        "read_name1",'read_number1',
        "chr1", "breakpoint1", "is_forward1", "insert_size1",
        "is_supplementary1", "mapping_quality1", "cigar1",
        "clip_side1", "largest_clip_length1",
        "largest_match_length1", "extra_bases1",
        "query_sequence1",
        "chr2", "start2", "breakpoint2", "is_forward2",
        "mapping_quality2", "cigar2",
        "clip_side2", "largest_clip_length2",
        "largest_match_length2", "extra_bases2",
    ]
    if include_mate:
        columns += [
            "chr_mate", "start_mate", "is_forward_mate", "insert_size_mate",
            "is_supplementary_mate", "mapping_quality_mate", "cigar_mate",
        ]
    filtered_alns_df = pd.DataFrame(alns)[columns]
    return filtered_alns_df.rename(
        columns={
            "read_name1": "read_name",
            'read_number1': 'read_number',
            "insert_size1": "insert_size",
            "query_sequence1": "query_sequence",
        }
    )

# %%
# Classify SV type based on breakpoint and orientation info
def classify_sv_type(row):
    if row['chr1'] != row['chr2']:
        return 'TRA'
    if row['is_forward1'] != row['is_forward2']:
        return 'INV'

    left_is_bp1 = row['breakpoint1'] < row['breakpoint2']
    if left_is_bp1:
        clip_pair = (row['clip_side1'], row['clip_side2'])
    else:
        clip_pair = (row['clip_side2'], row['clip_side1'])

    if clip_pair == ('left', 'right'):
        return 'DUP'
    if clip_pair == ('right', 'left'):
        return 'DEL'
    return 'UNK'

def add_sv_type(filtered_alns_df):
    sv_types = []
    for _, row in filtered_alns_df.iterrows():
        sv_types.append(classify_sv_type(row))
    filtered_alns_df["sv_type"] = sv_types
    return filtered_alns_df


# %%
def reverse_complement(seq):
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]

# determine MH and INS 
def junctional_features(row):
    # MH/INS are computed from the primary alignment query sequence.
    # extra_bases indicates cases where CIGAR complexity may reduce MH/INS confidence.
    # we always do it in a way that is consistent with the orientation of the target location aka breakpoint 1.
    # above, we made sure that the query_sequence column always contains the non-supplementary alignment's full sequence
    primary_seq = row['query_sequence']
    # first get the total read length
    read_length = len(primary_seq)
    if row['is_supplementary1']:
        primary_aln = 2
    else:
        primary_aln = 1

    match_sum = row['largest_match_length1'] + row['largest_match_length2']
    mh_length = max(0, match_sum - read_length)
    ins_length = max(0, read_length - match_sum)

    # the only case we need to reverse complement the primary sequence
    # is the scenario in which primary alignment is on the forward strand and in aln2 
    # but the supplementary at the target location is in the reverse strand
    primary_seq_strand = (
        primary_seq if not ((row['is_supplementary1']) and (not row['is_forward1']) and (row['is_forward2']))
        else reverse_complement(primary_seq)
    )
    target_clip_side = row['clip_side1']
    target_match_length = row['largest_match_length1']

    def slice_near_break(seq, match_len, clip_side, length, kind):
        if length <= 0:
            return ''
        if clip_side == 'right':
            start = match_len - length if kind == 'mh' else match_len
            end = match_len if kind == 'mh' else match_len + length
            return seq[start:end]
        start = -match_len if kind == 'mh' else -match_len - length
        end = -match_len + length if kind == 'mh' else -match_len
        return seq[start:end]

    mh_seq = slice_near_break(primary_seq_strand, target_match_length, target_clip_side, mh_length, 'mh')
    ins_seq = slice_near_break(primary_seq_strand, target_match_length, target_clip_side, ins_length, 'ins')

    return mh_length, ins_length, mh_seq, ins_seq, row['extra_bases' + str(primary_aln)]

def add_junctional_features(filtered_alns_df):
    results = [junctional_features(row) for _, row in filtered_alns_df.iterrows()]
    if results:
        mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = map(
            list, zip(*results)
        )
    else:
        mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = (
            [],
            [],
            [],
            [],
            [],
        )

    filtered_alns_df["mh_length"] = mh_lengths
    filtered_alns_df["ins_length"] = ins_lengths
    filtered_alns_df["mh_seq"] = mh_seqs
    filtered_alns_df["ins_seq"] = ins_seqs
    filtered_alns_df["extra_bases"] = extra_bases_list
    return filtered_alns_df


def main():
    args = parse_args()
    if args.pysam_quiet:
        pysam.set_verbosity(0)
    alns = run_pipeline(
        args.bam, args.chrom, args.breakpoint, args.window, args.include_mate
    )
    if not alns:
        print("No split reads found near the specified breakpoint.")
        sys.exit(0)
    filtered_alns_df = build_filtered_df(alns, args.include_mate)
    filtered_alns_df = add_sv_type(filtered_alns_df)
    filtered_alns_df = add_junctional_features(filtered_alns_df)
    if not args.out:
        print("Error: --out is required and must be a TSV path.", file=sys.stderr)
        sys.exit(2)
    out_path = args.out if args.out.endswith(".tsv") else f"{args.out}.tsv"
    filtered_alns_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(filtered_alns_df)} rows to {out_path}")
    print(filtered_alns_df['sv_type'].value_counts())

if __name__ == "__main__":
    main()
    
