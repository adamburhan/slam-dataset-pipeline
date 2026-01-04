#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load python/3.12
module load apptainer

REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"

source "${REPO_DIR}/.venv/bin/activate"

SEQUENCE_LIST="${REPO_DIR}/lists/eth3d.txt"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/eth3d"
CONFIG_FILE="${REPO_DIR}/configs/experiments/eth3d_orbslam2.yaml"

sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

echo "=== Job $SLURM_ARRAY_TASK_ID: $sequence_id ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Stage data to local SSD
echo "Staging data..."

# Copy and extract sequence
cp "${DATASET_ROOT}/${sequence_id}/${sequence_id}.zip" "$SLURM_TMPDIR/"
unzip "$SLURM_TMPDIR/${sequence_id}.zip" -d "$SLURM_TMPDIR"/${sequence_id}


echo "Staged to $SLURM_TMPDIR:"
ls -la "$SLURM_TMPDIR"

mkdir -p "$SLURM_TMPDIR/results"

srun python scripts/run_sequence_cli.py \
    --config "$CONFIG_FILE" \
    --sequence_id "$sequence_id" \
    --dataset_root "$SLURM_TMPDIR" \
    --output_dir "$SLURM_TMPDIR/results"

# Copy results back to persistent storage
OUTPUT_ROOT="/scratch/adamb14/ood_slam/results/eth3d/orbslam2"
mkdir -p "${OUTPUT_ROOT}"
cp -r "$SLURM_TMPDIR/results/"* "${OUTPUT_ROOT}/"

echo "End: $(date)"