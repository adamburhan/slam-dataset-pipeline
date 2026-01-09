#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out

set -e

module load python/3.12

# --- 1. Define Variables Correctly ---
REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
# FIXME: Update this path to where your original .tar.gz files live!
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/kitti/raw/sequences" 
OUTPUT_DEST="/scratch/adamb14/ood_slam/datasets/kitti/orb_images"

source "${REPO_DIR}/.venv/bin/activate"

SEQUENCE_LIST="${REPO_DIR}/lists/kitti.txt"

# Get sequence ID (e.g., "04")
sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

echo "=== Job $SLURM_ARRAY_TASK_ID: $sequence_id ==="
echo "Node: $(hostname)"

# --- 2. Stage Data ---
echo "Staging data..."
mkdir -p "$SLURM_TMPDIR/sequences"
mkdir -p "$SLURM_TMPDIR/output_results"

# Copy source tarball to SSD
cp "${DATASET_ROOT}/${sequence_id}.tar.gz" "$SLURM_TMPDIR/sequences/"

# Extract. 
# NOTE: This assumes the tarball extracts a folder named "04" directly.
# If it extracts "sequences/04", the python script will still find it recursively, 
# but "output_results" might nest differently.
tar -xvzf "$SLURM_TMPDIR/sequences/${sequence_id}.tar.gz" -C "$SLURM_TMPDIR/sequences"

# --- 3. Run Preprocessing ---
echo "Running Python script..."
python "${REPO_DIR}/scripts/preprocess_orb.py" \
        --dataset_root "$SLURM_TMPDIR/sequences" \
        --output_root "$SLURM_TMPDIR/output_results"

# --- 4. Archive & Save Results ---
echo "Archiving results..."

# FIX: Create the tar file INSIDE tmpdir
# FIX: Use -C to cd into output_results, then just zip the folder name (e.g. "04")
# This ensures your zip doesn't contain a path like "home/adam/..."
tar -czvf "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" \
    -C "$SLURM_TMPDIR/output_results" \
    "$sequence_id"

# Ensure destination exists
mkdir -p "$OUTPUT_DEST"

# Move it out
echo "Saving to Scratch..."
rsync -av "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" "$OUTPUT_DEST/"

echo "Done."