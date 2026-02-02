#!/bin/bash -l
#SBATCH --job-name=annotate_svs
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-42
#SBATCH --output=/scratch/y/yz762/DSS/logs/%x_%A_%a.out
#SBATCH --error=/scratch/y/yz762/DSS/logs/%x_%A_%a.err

set -euo pipefail

module load miniforge3

BAM_DIR="/data/beroukhim1/youyun/nti/data/DSS/run_260106"
SCRIPT="/PHShome/yz762/DSS/code/deep_scar_seq_toolkit/annotate_SVs.py"
RUN_DATE="${DSS_RUN_DATE:-$(date +%m%d%y)}"
OUT_DIR="/scratch/y/yz762/DSS/dss_${RUN_DATE}"

mkdir -p "${OUT_DIR}"

BAMS=("${BAM_DIR}"/*.bam)
if [[ ${#BAMS[@]} -eq 0 ]]; then
  echo "No BAMs found in ${BAM_DIR}" >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. This script is meant for a job array." >&2
  exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#BAMS[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (0..$(( ${#BAMS[@]} - 1 )))." >&2
  exit 1
fi

BAM="${BAMS[$SLURM_ARRAY_TASK_ID]}"
BASENAME="$(basename "${BAM}" .bam)"
OUT_FILE="${OUT_DIR}/${BASENAME}.annotated.tsv"

start_ts=$(date +%s)
start_human=$(date)

bam_bytes=$(stat -c%s "${BAM}")
bam_size_human=$(du -h "${BAM}" | awk '{print $1}')

echo "Job start: ${start_human}"
echo "BAM: ${BAM}"
echo "BAM size: ${bam_size_human} (${bam_bytes} bytes)"
echo "Output: ${OUT_FILE}"

time conda run -n dss python3 "${SCRIPT}" \
  --bam "${BAM}" \
  --chrom chr2 \
  --breakpoint 1152626 \
  --window 5 \
  --pysam_quiet \
  --out "${OUT_FILE}"

end_ts=$(date +%s)
end_human=$(date)
elapsed=$(( end_ts - start_ts ))

printf "Job end: %s\n" "${end_human}"
printf "Elapsed: %s seconds\n" "${elapsed}"
