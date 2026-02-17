# Deep Scar Seq Toolkit
Deep scar seq analysis toolkit for annotating split reads near a CRISPR breakpoint
and summarizing structural variation (SV) features.

## Overview
This repository contains:
- Core annotation logic (package code under `src/`)
- CLI entrypoints under `scripts/`
- Batch- or dataset-specific analyses under `scratch/`

## Repository layout
- `src/deep_scar_seq_toolkit/`: reusable package code
- `scripts/`: CLI entrypoints (stable interfaces)
- `scratch/`: exploratory and batch-specific scripts
- `data/`: input/output data (not versioned here)

## Installation (minimal)
Create a conda environment and install dependencies:
```bash
conda create -n dss python=3.10 -y
conda activate dss
pip install pysam pandas numpy matplotlib seaborn tqdm
```

If you want to import from `src/` without modifying `PYTHONPATH`, add a
packaging config later and run `pip install -e .`. For now, running the CLI
scripts directly is sufficient.

## Usage
1. Annotate SVs for one BAM:
```bash
python scripts/annotate_svs.py \\
  --bam /path/to/sample.bam \\
  --chrom chr2 \\
  --breakpoint 1152626 \\
  --window 5 \\
  --umi-tags RX,UR \\
  --out /path/to/output.annotated.tsv
```
  
2. Aggregate SVs for one annotated TSV:
```bash
python scripts/aggregate_svs.py \\
  --annotated /path/to/output.annotated.tsv \\
  --out /path/to/output.aggregated.tsv
```
3. Plot SV distributions for a batch of annotated TSVs:
```bash
python scratch/plotting/plot_sv_distributions.py \\
  --input-dir data/dss_020226 \\
  --pattern '*chr2_1152626_w5*.annotated.tsv' \\
  --output-dir sv_plots_chr2_1152626_w5
```

## Output columns
Annotated TSVs now include UMI metadata:
- `umi`: canonical UMI chosen by tag priority (default `RX,UR`)
- `umi_source_tag`: tag used for `umi` (`RX`, `UR`, or empty)
- `umi_missing`: `True` when no canonical UMI is available
- `umi_mi`: raw `MI` tag when present (for posterity)

Duplicate handling is always on during annotation: duplicate-marked reads are
excluded before writing output rows. Annotated TSVs retain duplicate metadata
for QC:
- `is_duplicate`: effective duplicate status used for filtering
- `is_duplicate_raw`: duplicate flag on the junction-overlapping alignment
- `duplicate_source`: how duplicate status was derived (`self_primary`,
  `resolved_primary`, or `self_fallback`)

Aggregated TSVs now include UMI summary metrics:
- `umi_count`: unique non-empty UMIs per event after filtering
- `umi_missing_read_count`: number of post-filter reads missing canonical UMI
- `umi_coverage`: `1 - umi_missing_read_count / read_count`

When aggregating legacy annotated TSVs that do not contain duplicate metadata,
aggregation remains compatible and emits a warning that duplicate-aware
filtering could not be re-applied.


## Development workflow
- Start new ideas in `scratch/` (e.g., `scratch/analysis` or `scratch/validation`).
- When a function becomes reusable, move it into `src/deep_scar_seq_toolkit/`
  and call it from a thin CLI in `scripts/`.
