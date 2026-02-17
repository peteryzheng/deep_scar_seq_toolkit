#!/usr/bin/env python3
"""Annotate split reads around a breakpoint and derive SV features."""

from collections import Counter

import pandas as pd
import pysam
from tqdm import tqdm

from .config import (
    DEFAULT_INCLUDE_MATE,
    DEFAULT_PYSAM_QUIET,
    DEFAULT_UMI_TAG_PRIORITY,
    DEFAULT_WINDOW,
)


def is_split(aln):
    return ((aln.has_tag("SA")) and (not aln.is_secondary))


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
    """Find the largest clip and match in the CIGAR and infer breakpoint."""
    clip_indices = [(i, l) for i, (op, l) in enumerate(cigar_tuples) if op in [4, 5]]
    match_indices = [(i, l) for i, (op, l) in enumerate(cigar_tuples) if op == 0]
    if not clip_indices or not match_indices:
        return None
    largest_clip = max(clip_indices, key=lambda x: x[1])
    largest_match = max(match_indices, key=lambda x: x[1])
    clip_pos = largest_clip[0]
    match_pos = largest_match[0]
    if clip_pos < match_pos:
        clip_side = "left"
        breakpoint = ref_start
    else:
        clip_side = "right"
        ref_consuming_ops = {0, 2, 3, 7, 8}
        ref_len = sum(l for op, l in cigar_tuples if op in ref_consuming_ops)
        breakpoint = ref_start + ref_len
    total_bases = sum([l for _, l in cigar_tuples])
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


def extract_umi_metadata(aln, umi_tag_priority):
    canonical_umi = ""
    source_tag = ""
    for tag in umi_tag_priority:
        if not aln.has_tag(tag):
            continue
        value = aln.get_tag(tag)
        if value is None:
            continue
        value_str = str(value)
        if not value_str:
            continue
        canonical_umi = value_str
        source_tag = tag
        break
    umi_mi = str(aln.get_tag("MI")) if aln.has_tag("MI") else ""
    return {
        "umi": canonical_umi,
        "umi_source_tag": source_tag,
        "umi_missing": canonical_umi == "",
        "umi_mi": umi_mi,
    }


def parse_sa_entries(sa_tag):
    sa_entries = []
    for entry in sa_tag.split(";"):
        if not entry:
            continue
        fields = entry.split(",")
        if len(fields) < 6:
            continue
        try:
            sa_entries.append(
                {
                    "chr": fields[0],
                    "start": int(fields[1]) - 1,
                    "is_forward": fields[2] == "+",
                    "cigar": fields[3],
                    "mapping_quality": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return sa_entries


def parse_sa_tag(sa_tag):
    sa_entries = parse_sa_entries(sa_tag)
    if not sa_entries:
        return None
    # max() is stable, so ties keep the first SA entry.
    return max(sa_entries, key=lambda entry: entry["mapping_quality"])


def find_other_alignment(
    sa_tag,
    query_name,
    is_read1,
    is_supplementary,
    bam,
    require_primary=False,
):
    best_aln = None
    best_mapq = -1
    seen = set()
    for sa_info in parse_sa_entries(sa_tag):
        region = f"{sa_info['chr']}:{sa_info['start']}-{sa_info['start'] + 1}"
        for aln in bam.fetch(region=region):
            if aln.query_name != query_name or aln.is_read1 != is_read1:
                continue
            if aln.is_secondary:
                continue
            if aln.is_supplementary == is_supplementary:
                continue
            if require_primary and aln.is_supplementary:
                continue
            aln_key = (
                aln.reference_id,
                aln.reference_start,
                aln.cigarstring,
                aln.is_supplementary,
                aln.mapping_quality,
                aln.flag,
            )
            if aln_key in seen:
                continue
            seen.add(aln_key)
            # Tie-break only by MAPQ; for equal MAPQ keep the first-seen alignment.
            if best_aln is None or aln.mapping_quality > best_mapq:
                best_aln = aln
                best_mapq = aln.mapping_quality
    return best_aln


def resolve_primary_alignment(aln, bam):
    if not aln.has_tag("SA"):
        return None
    return find_other_alignment(
        aln.get_tag("SA"),
        aln.query_name,
        aln.is_read1,
        aln.is_supplementary,
        bam,
        require_primary=True,
    )


def resolve_duplicate_status(aln, bam):
    raw_duplicate = bool(aln.is_duplicate)
    if not aln.is_supplementary:
        return {
            "is_duplicate": raw_duplicate,
            "is_duplicate_raw": raw_duplicate,
            "duplicate_source": "self_primary",
            "primary_alignment": aln,
        }

    primary_aln = resolve_primary_alignment(aln, bam)
    if primary_aln is None:
        return {
            "is_duplicate": raw_duplicate,
            "is_duplicate_raw": raw_duplicate,
            "duplicate_source": "self_fallback",
            "primary_alignment": None,
        }

    return {
        "is_duplicate": bool(primary_aln.is_duplicate),
        "is_duplicate_raw": raw_duplicate,
        "duplicate_source": "resolved_primary",
        "primary_alignment": primary_aln,
    }


def print_duplicate_summary(duplicate_stats):
    candidate_total = int(duplicate_stats.get("candidate_total", 0))
    excluded_duplicates = int(duplicate_stats.get("excluded_duplicates", 0))
    excluded_resolved_primary = int(duplicate_stats.get("excluded_resolved_primary", 0))
    supp_fallback_count = int(duplicate_stats.get("supp_fallback_count", 0))
    retained = int(duplicate_stats.get("retained", candidate_total - excluded_duplicates))

    excluded_pct = (excluded_duplicates / candidate_total) if candidate_total else 0.0
    fallback_pct = (supp_fallback_count / candidate_total) if candidate_total else 0.0
    print(
        "Duplicate filter | "
        f"candidates={candidate_total} "
        f"excluded={excluded_duplicates} ({excluded_pct:.1%}) "
        f"excluded_resolved_primary={excluded_resolved_primary} "
        f"supp_fallback={supp_fallback_count} ({fallback_pct:.1%}) "
        f"retained={retained}"
    )


def run_pipeline(bam_path, chrom, breakpoint, window, include_mate, umi_tag_priority):
    bam = pysam.AlignmentFile(bam_path, "rb")
    region = f"{chrom}:{breakpoint}-{breakpoint}"

    candidate_alns = []
    for aln in tqdm(bam.fetch(region=region), desc="Scanning split reads"):
        if is_split(aln):
            bp1_info = clip_breakpoint(aln.cigartuples, aln.reference_start)
            if bp1_info is None:
                continue
            if abs(bp1_info["breakpoint"] - breakpoint) <= window:
                new_aln_dict = {
                    **extract_aln_core(aln),
                    **extract_umi_metadata(aln, umi_tag_priority),
                    **bp1_info,
                    "read_number": aln.is_read1,
                    "aln_obj": aln,
                }
                new_aln_dict_1 = {k + "1": v for k, v in new_aln_dict.items()}
                candidate_alns.append(new_aln_dict_1)

    duplicate_stats = Counter()
    alns = []
    sa_time = 0.0
    mate_time = 0.0
    for aln in tqdm(
        candidate_alns,
        desc="Resolving duplicate status and annotating other alignment",
    ):
        duplicate_stats["candidate_total"] += 1
        aln_obj1 = aln["aln_obj1"]
        duplicate_info = resolve_duplicate_status(aln_obj1, bam)
        aln.update(
            {
                "is_duplicate1": duplicate_info["is_duplicate"],
                "is_duplicate_raw1": duplicate_info["is_duplicate_raw"],
                "duplicate_source1": duplicate_info["duplicate_source"],
                "primary_aln_obj1": duplicate_info["primary_alignment"],
            }
        )
        if duplicate_info["duplicate_source"] == "self_fallback":
            duplicate_stats["supp_fallback_count"] += 1
        if duplicate_info["is_duplicate"]:
            duplicate_stats["excluded_duplicates"] += 1
            if duplicate_info["duplicate_source"] == "resolved_primary":
                duplicate_stats["excluded_resolved_primary"] += 1
            continue

        if aln_obj1.has_tag("SA"):
            sa_tag = aln_obj1.get_tag("SA")
            if aln_obj1.is_supplementary:
                primary_aln = aln.get("primary_aln_obj1")
                if primary_aln is not None:
                    bp2_info = clip_breakpoint(primary_aln.cigartuples, primary_aln.reference_start)
                    if bp2_info is not None:
                        new_aln_dict_2 = {
                            "chr": primary_aln.reference_name,
                            "start": primary_aln.reference_start,
                            "is_forward": primary_aln.is_forward,
                            "cigar": primary_aln.cigarstring,
                            "mapping_quality": primary_aln.mapping_quality,
                            **bp2_info,
                        }
                        new_aln_dict_2 = {k + "2": v for k, v in new_aln_dict_2.items()}
                        aln.update(new_aln_dict_2)
                    aln["query_sequence1"] = primary_aln.query_sequence
            else:
                sa_info = parse_sa_tag(sa_tag)
                if sa_info is not None:
                    cigar2_tuples = parse_cigar_tuples(sa_info["cigar"])
                    bp2_info = clip_breakpoint(cigar2_tuples, sa_info["start"])
                    if bp2_info is not None:
                        new_aln_dict_2 = {
                            "chr": sa_info["chr"],
                            "start": sa_info["start"],
                            "is_forward": sa_info["is_forward"],
                            "cigar": sa_info["cigar"],
                            "mapping_quality": sa_info["mapping_quality"],
                            **bp2_info,
                        }
                        new_aln_dict_2 = {k + "2": v for k, v in new_aln_dict_2.items()}
                        aln.update(new_aln_dict_2)
        if include_mate:
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
        alns.append(aln)

    duplicate_stats["retained"] = len(alns)
    print_duplicate_summary(duplicate_stats)

    if not alns:
        return alns

    print(f"annotate breakdown | SA: {sa_time:.2f}s | mate: {mate_time:.2f}s")
    return alns


def build_filtered_df(alns, include_mate):
    columns = [
        "read_name1",
        "read_number1",
        "chr1",
        "breakpoint1",
        "is_forward1",
        "insert_size1",
        "is_supplementary1",
        "mapping_quality1",
        "cigar1",
        "clip_side1",
        "largest_clip_length1",
        "largest_match_length1",
        "extra_bases1",
        "query_sequence1",
        "umi1",
        "umi_source_tag1",
        "umi_missing1",
        "umi_mi1",
        "is_duplicate1",
        "is_duplicate_raw1",
        "duplicate_source1",
        "chr2",
        "start2",
        "breakpoint2",
        "is_forward2",
        "mapping_quality2",
        "cigar2",
        "clip_side2",
        "largest_clip_length2",
        "largest_match_length2",
        "extra_bases2",
    ]
    if include_mate:
        columns += [
            "chr_mate",
            "start_mate",
            "is_forward_mate",
            "insert_size_mate",
            "is_supplementary_mate",
            "mapping_quality_mate",
            "cigar_mate",
        ]
    filtered_alns_df = pd.DataFrame(alns)[columns]
    return filtered_alns_df.rename(
        columns={
            "read_name1": "read_name",
            "read_number1": "read_number",
            "insert_size1": "insert_size",
            "query_sequence1": "query_sequence",
            "umi1": "umi",
            "umi_source_tag1": "umi_source_tag",
            "umi_missing1": "umi_missing",
            "umi_mi1": "umi_mi",
            "is_duplicate1": "is_duplicate",
            "is_duplicate_raw1": "is_duplicate_raw",
            "duplicate_source1": "duplicate_source",
        }
    )


def classify_sv_type(row):
    if row["chr1"] != row["chr2"]:
        return "TRA"
    if row["is_forward1"] != row["is_forward2"]:
        return "INV"

    left_is_bp1 = row["breakpoint1"] < row["breakpoint2"]
    if left_is_bp1:
        clip_pair = (row["clip_side1"], row["clip_side2"])
    else:
        clip_pair = (row["clip_side2"], row["clip_side1"])

    if clip_pair == ("left", "right"):
        return "DUP"
    if clip_pair == ("right", "left"):
        return "DEL"
    return "UNK"


def add_sv_type(filtered_alns_df):
    sv_types = []
    for _, row in filtered_alns_df.iterrows():
        sv_types.append(classify_sv_type(row))
    filtered_alns_df["sv_type"] = sv_types
    return filtered_alns_df


def reverse_complement(seq):
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]


def junctional_features(row):
    primary_seq = row["query_sequence"]
    read_length = len(primary_seq)
    primary_aln = 2 if row["is_supplementary1"] else 1

    match_sum = row["largest_match_length1"] + row["largest_match_length2"]
    mh_length = max(0, match_sum - read_length)
    ins_length = max(0, read_length - match_sum)

    primary_seq_strand = (
        primary_seq
        if not ((row["is_supplementary1"]) and (not row["is_forward1"]) and (row["is_forward2"]))
        else reverse_complement(primary_seq)
    )
    target_clip_side = row["clip_side1"]
    target_match_length = row["largest_match_length1"]

    def slice_near_break(seq, match_len, clip_side, length, kind):
        if length <= 0:
            return ""
        if clip_side == "right":
            start = match_len - length if kind == "mh" else match_len
            end = match_len if kind == "mh" else match_len + length
            return seq[start:end]
        start = -match_len if kind == "mh" else -match_len - length
        end = -match_len + length if kind == "mh" else -match_len
        return seq[start:end]

    mh_seq = slice_near_break(primary_seq_strand, target_match_length, target_clip_side, mh_length, "mh")
    ins_seq = slice_near_break(primary_seq_strand, target_match_length, target_clip_side, ins_length, "ins")

    return mh_length, ins_length, mh_seq, ins_seq, row["extra_bases" + str(primary_aln)]


def add_junctional_features(filtered_alns_df):
    results = [junctional_features(row) for _, row in filtered_alns_df.iterrows()]
    if results:
        mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = map(list, zip(*results))
    else:
        mh_lengths, ins_lengths, mh_seqs, ins_seqs, extra_bases_list = ([], [], [], [], [])

    filtered_alns_df["mh_length"] = mh_lengths
    filtered_alns_df["ins_length"] = ins_lengths
    filtered_alns_df["mh_seq"] = mh_seqs
    filtered_alns_df["ins_seq"] = ins_seqs
    filtered_alns_df["extra_bases"] = extra_bases_list
    return filtered_alns_df


def print_umi_summary(filtered_alns_df):
    if filtered_alns_df.empty:
        return
    tag_counts = Counter(filtered_alns_df["umi_source_tag"])
    total_reads = len(filtered_alns_df)
    missing_count = int(filtered_alns_df["umi_missing"].sum())
    rx_count = int(tag_counts.get("RX", 0))
    ur_count = int(tag_counts.get("UR", 0))
    print(
        "UMI tags | "
        f"total={total_reads} "
        f"RX={rx_count} ({rx_count / total_reads:.1%}) "
        f"UR={ur_count} ({ur_count / total_reads:.1%}) "
        f"missing={missing_count} ({missing_count / total_reads:.1%})"
    )


def annotate_bam(
    bam_path,
    chrom,
    breakpoint,
    window=DEFAULT_WINDOW,
    include_mate=DEFAULT_INCLUDE_MATE,
    pysam_quiet=DEFAULT_PYSAM_QUIET,
    umi_tag_priority=DEFAULT_UMI_TAG_PRIORITY,
):
    umi_tag_priority = tuple(
        str(tag).strip().upper() for tag in umi_tag_priority if str(tag).strip()
    )
    if not umi_tag_priority:
        raise ValueError("umi_tag_priority must contain at least one SAM tag.")
    if pysam_quiet:
        pysam.set_verbosity(0)
    alns = run_pipeline(
        bam_path,
        chrom,
        breakpoint,
        window,
        include_mate,
        umi_tag_priority,
    )
    if not alns:
        return pd.DataFrame()
    filtered_alns_df = build_filtered_df(alns, include_mate)
    filtered_alns_df = add_sv_type(filtered_alns_df)
    filtered_alns_df = add_junctional_features(filtered_alns_df)
    print_umi_summary(filtered_alns_df)
    return filtered_alns_df
