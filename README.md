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


## Development workflow
- Start new ideas in `scratch/` (e.g., `scratch/analysis` or `scratch/validation`).
- When a function becomes reusable, move it into `src/deep_scar_seq_toolkit/`
  and call it from a thin CLI in `scripts/`.
