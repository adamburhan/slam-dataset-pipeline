#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-60
#SBATCH --output=logs/eth3d_%A_%a.out

set -e
module load python/3.12

# --- CONFIG ---
REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/eth3d/training" 
OUTPUT_DEST="/scratch/adamb14/ood_slam/datasets/eth3d/orb_images"

# Point to the Master PCA in the repo (Safe Location)
MASTER_PCA="/scratch/adamb14/ood_slam/datasets/weights/master_orb_pca.joblib"

source "${REPO_DIR}/.venv/bin/activate"

SEQUENCE_LIST="${REPO_DIR}/lists/eth3d.txt"
sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

echo "=== Processing: $sequence_id ==="

# --- Stage Data ---
echo "Staging data..."
mkdir -p "$SLURM_TMPDIR/sequences"
mkdir -p "$SLURM_TMPDIR/output_results"

target_seq_dir="$SLURM_TMPDIR/sequences/$sequence_id"
mkdir -p "$target_seq_dir"

# Copy & Unzip
# ETH3D zips usually contain the folder structure inside (e.g., courtyard/rgb/...)
cp "${DATASET_ROOT}/${sequence_id}.zip" "$SLURM_TMPDIR/sequences/"
unzip -q "$SLURM_TMPDIR/sequences/${sequence_id}.zip" -d "$target_seq_dir"

# --- Run Preprocessing ---
echo "Running Python script..."
# Python will find '.../rgb', process it, and output to 'output_results/{seq_id}'
# (Because we updated the Python script to handle 'rgb' correctly)
python "${REPO_DIR}/scripts/preprocess_orb.py" \
        --dataset_root "$target_seq_dir" \
        --output_root "$SLURM_TMPDIR/output_results" \
        --pca_path "$MASTER_PCA"

# --- Archive Results ---
echo "Archiving results..."

# FIX: Robust Rename
# Even if Python names it 'rgb' or 'courtyard', we force it to be $sequence_id
generated_folder=$(ls "$SLURM_TMPDIR/output_results" | head -n 1)

if [ "$generated_folder" != "$sequence_id" ]; then
    echo "Renaming '$generated_folder' to '$sequence_id'..."
    mv "$SLURM_TMPDIR/output_results/$generated_folder" "$SLURM_TMPDIR/output_results/$sequence_id"
fi

# Archive
tar -czvf "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" \
    -C "$SLURM_TMPDIR/output_results" \
    "$sequence_id"

# --- Save ---
mkdir -p "$OUTPUT_DEST"
echo "Saving to $OUTPUT_DEST..."
rsync -av "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" "$OUTPUT_DEST/"

echo "Done."