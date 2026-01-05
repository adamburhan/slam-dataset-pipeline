#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-60
#SBATCH --output=logs/eth3d_%x_%A_%a.out

set -e

module load python/3.12
module load apptainer

REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
source "${REPO_DIR}/.venv/bin/activate"

# PATHS
SEQUENCE_LIST="${REPO_DIR}/lists/eth3d.txt"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/eth3d/training"
CONFIG_FILE="${REPO_DIR}/configs/experiments/eth3d_orbslam2.yaml"
OUTPUT_ROOT="/scratch/adamb14/ood_slam/results/eth3d/orbslam2"

sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

echo "=== Job $SLURM_ARRAY_TASK_ID: $sequence_id ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Stage data to local SSD
echo "Staging data..."

cp "${DATASET_ROOT}/${sequence_id}.zip" "$SLURM_TMPDIR/"

# The zip already contains the folder "${sequence_id}", so this creates "$SLURM_TMPDIR/${sequence_id}"
unzip -q "$SLURM_TMPDIR/${sequence_id}.zip" -d "$SLURM_TMPDIR"

echo "Staged to $SLURM_TMPDIR:"
ls -la "$SLURM_TMPDIR"

mkdir -p "$SLURM_TMPDIR/results"

# Note: Pipeline expects dataset_root to contain the sequence folder, which it does now.
srun python scripts/run_sequence_cli.py \
    --config "$CONFIG_FILE" \
    --sequence_id "$sequence_id" \
    --dataset_root "$SLURM_TMPDIR" \
    --output_dir "$SLURM_TMPDIR/results"

# Copy results back to persistent storage
# The pipeline creates "$SLURM_TMPDIR/results/${sequence_id}"
SEQ_OUTPUT_PATH="${OUTPUT_ROOT}/${sequence_id}"
mkdir -p "$SEQ_OUTPUT_PATH"

if [ -d "$SLURM_TMPDIR/results/${sequence_id}" ]; then
    cp -r "$SLURM_TMPDIR/results/${sequence_id}/"* "$SEQ_OUTPUT_PATH/"
else
    echo "Warning: Output directory for ${sequence_id} not found."
fi

echo "End: $(date)"