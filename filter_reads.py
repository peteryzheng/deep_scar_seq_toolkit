# %%
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
bam = pysam.AlignmentFile(workdir+"youyun/nti/data/DSS/run_260106/NA06_A.hg19.dedup.bam", "rb")

# %%
def is_split(aln):
    # this is checking if an alignment is split by looking at the SA tag
    # this alignment can be primary or supplementary
    return ((aln.has_tag('SA')) and (not aln.is_secondary))

def clip_breakpoint(aln):
    """Find the largest soft/hard clip (C) and largest match (M) in the CIGAR, and determine if C is left or right of M."""
    cigar = aln.cigartuples
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
        breakpoint = aln.reference_start
    # if right, then the breakpoint is at the right side of the alignment which is reference_end
    else:
        clip_side = 'right'
        breakpoint = aln.reference_end
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
        "has_SA": aln.has_tag("SA"),
        "sa_tag": aln.get_tag("SA") if aln.has_tag("SA") else None,
        "query_sequence": aln.query_sequence,
    }

def find_other_alignment(sa_tag, query_name, is_read1, is_supplementary, bam):
    """Find the other alignment based on the SA tag information."""
    # if the alignment is primary, look for supplementary
    # if the alignment is supplementary, look for primary
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

# %%
# Count split reads and collect alignments
alns = []
for aln in tqdm(bam.fetch(region='chr2:1153069-1153069'), desc='Scanning split reads'):
    # only check split reads
    if is_split(aln):
        # check breakpoint info of the split read
        bp1_info = clip_breakpoint(aln)
        # make sure breakpoint is within 5bp of the pre-defined location
        if abs(bp1_info['breakpoint'] - 1153069) <= 5:
            new_aln_dict = {
                **extract_aln_core(aln),
                **bp1_info,
                "aln_obj": aln
            }
            
            # add 1 to the end to indicate breakpoint 1
            new_aln_dict_1 = {k + "1": v for k, v in new_aln_dict.items()}
            alns.append(new_aln_dict_1)
            # this dict contains the keys: 
            # read_name1, chr1, start1, cigar1, read_number1, is_forward1, insert_size1,
            # is_supplementary1, mapping_quality1, largest_clip_length1, largest_match_length1,
            # clip_side1, breakpoint1, aln_obj1

for aln in tqdm(alns, desc='Annotating other alignment of the breakpoint and mates'):
    if aln['has_SA1']:
        # use the SA tag to find the other alignment of the split read
        # if the current alignment is primary, the other should be supplementary, and vice versa
        other_aln = find_other_alignment(aln['sa_tag1'], aln['read_name1'], aln['read_number1'], aln['is_supplementary1'], bam)
        # extract breakpoint info of the second breakpoint from the other alignment
        bp2_info = clip_breakpoint(other_aln)
        # add some print statements here for both alignments
        # print("============================")
        # print(f"Processing read: {aln['read_name1']}, Read number: {'read1' if aln['read_number1'] else 'read2'}")
        # print(f"Breakpoint 1 info: Chr: {aln['chr1']}, Pos: {aln['breakpoint1']}, Clip side: {aln['clip_side1']}, Clip length: {aln['largest_clip_length1']}, Direction: {'+' if aln['is_forward1'] else '-'}")
        # print(f"CIGAR: {aln['cigar1']}, aln type: {'Supplementary' if aln['is_supplementary1'] else 'Primary'}, query seq len: {len(aln['query_sequence1'])}")
        # print(f"Breakpoint 2 info: Chr: {other_aln.reference_name}, Pos: {bp2_info['breakpoint']}, Clip side: {bp2_info['clip_side']}, Clip length: {bp2_info['largest_clip_length']}, Direction: {'+' if other_aln.is_forward else '-'}")
        # print(f"CIGAR: {other_aln.cigarstring}, aln type: {'Supplementary' if other_aln.is_supplementary else 'Primary'}, query seq len: {len(other_aln.query_sequence)}")
        new_aln_dict_2 = {
            **extract_aln_core(other_aln),
            "aln_obj": other_aln,
            **bp2_info
        }
        # add 2 to the end to indicate breakpoint 2
        new_aln_dict_2 = {k + "2": v for k, v in new_aln_dict_2.items()}
        aln.update(new_aln_dict_2)
    # Find mate
    try:
        mate = bam.mate(aln['aln_obj1'])
        mate_aln_info = extract_aln_core(mate)
        mate_aln_dict = {k + "_mate": v for k, v in mate_aln_info.items()}
        mate_aln_dict["aln_obj_mate"] = mate
        aln.update(mate_aln_dict)
    except ValueError as e:
        print(f"Mate not found: {e}")

# %%
# Build a dataframe directly from alns (no filtered_alns list)
filtered_alns_df = pd.DataFrame(alns)[
    [
        'read_name1',
        'chr1', 'breakpoint1', 'is_forward1', 'insert_size1', 'is_supplementary1', 'mapping_quality1', 'cigar1', 
        'clip_side1', 'largest_clip_length1', 'largest_match_length1', 'extra_bases1',
        'chr2', 'breakpoint2', 'is_forward2', 'insert_size2', 'is_supplementary2', 'mapping_quality2', 'cigar2', 
        'clip_side2', 'largest_clip_length2', 'largest_match_length2', 'extra_bases2',
        'chr_mate', 'start_mate', 'is_forward_mate', 'insert_size_mate', 'is_supplementary_mate', 'mapping_quality_mate', 'cigar_mate',
        'aln_obj1', 'aln_obj2', 'aln_obj_mate'
    ]
]
filtered_alns_df = filtered_alns_df.rename(columns={'read_name1': 'read_name'})
filtered_alns_df.head()

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

sv_types = []
for _, row in filtered_alns_df.iterrows():
    sv_types.append(classify_sv_type(row))

filtered_alns_df['sv_type'] = sv_types
print(filtered_alns_df['sv_type'].value_counts())
filtered_alns_df[[
    'read_name','chr1','breakpoint1','is_forward1','clip_side1', 'extra_bases1',
    'chr2','breakpoint2','is_forward2','clip_side2', 'extra_bases2',
    'sv_type'
]][filtered_alns_df['sv_type'] == 'INV']


# %%
def reverse_complement(seq):
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]

# determine MH and INS 
def junctional_features(row):
    # MH/INS are computed from the primary alignment query sequence.
    # extra_bases indicates cases where CIGAR complexity may reduce MH/INS confidence.
    # first get the total read length
    # but only get the proper alignment's read length because supplementary alignment are often hard clipped
    # in which case the hard clipped query sequence is not retained in aln.query_sequence
    if row['is_supplementary1']:
        read_length = len(row['aln_obj2'].query_sequence)
        primary_aln = 2
    else:
        read_length = len(row['aln_obj1'].query_sequence)
        primary_aln = 1

    match_sum = row['largest_match_length1'] + row['largest_match_length2']
    mh_length = max(0, match_sum - read_length)
    ins_length = max(0, read_length - match_sum)

    primary_seq = row['aln_obj' + str(primary_aln)].query_sequence
    primary_seq_strand = (
        primary_seq if row['is_forward' + str(primary_aln)] else reverse_complement(primary_seq)
    )
    primary_clip_side = row['clip_side' + str(primary_aln)]
    primary_match_length = row['largest_match_length' + str(primary_aln)]

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

    mh_seq = slice_near_break(primary_seq_strand, primary_match_length, primary_clip_side, mh_length, 'mh')
    ins_seq = slice_near_break(primary_seq_strand, primary_match_length, primary_clip_side, ins_length, 'ins')

    if primary_aln == 2 and row['is_forward1'] != row['is_forward2']:
        mh_seq = reverse_complement(mh_seq)
        ins_seq = reverse_complement(ins_seq)

    return mh_length, ins_length, mh_seq, ins_seq, row['extra_bases' + str(primary_aln)]

results = [junctional_features(row) for _, row in filtered_alns_df.iterrows()]
if results:
    mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = map(list, zip(*results))
else:
    mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = [], [], [], [], []

filtered_alns_df['mh_length'] = mh_lengths
filtered_alns_df['ins_length'] = ins_lengths
filtered_alns_df['mh_seq'] = mh_seqs
filtered_alns_df['ins_seq'] = ins_seqs
filtered_alns_df['extra_bases'] = extra_bases_list
filtered_alns_df[['read_name','mh_length','ins_length','mh_seq','ins_seq','extra_bases']]
    
# %%
