#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/euroc_%A_%a.out

set -e
module load python/3.12

# --- CONFIG ---
REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/euroc" 
OUTPUT_DEST="/scratch/adamb14/ood_slam/datasets/euroc/orb_images"

# CHECK THIS PATH: usually in repo/weights, not scratch/weights
MASTER_PCA="/scratch/adamb14/ood_slam/datasets/weights/master_orb_pca.joblib"

source "${REPO_DIR}/.venv/bin/activate"

SEQUENCE_LIST="${REPO_DIR}/lists/euroc.txt"
sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

echo "=== Processing: $sequence_id ==="

# --- Stage Data ---
echo "Staging data..."
mkdir -p "$SLURM_TMPDIR/sequences"
mkdir -p "$SLURM_TMPDIR/output_results"

# Create a specific folder for this sequence to keep unzipping clean
target_seq_dir="$SLURM_TMPDIR/sequences/$sequence_id"
mkdir -p "$target_seq_dir"

# Copy & Unzip
cp "${DATASET_ROOT}/${sequence_id}/${sequence_id}.zip" "$SLURM_TMPDIR/sequences/"
unzip -q "$SLURM_TMPDIR/sequences/${sequence_id}.zip" -d "$target_seq_dir"

# --- Run Preprocessing ---
echo "Running Python script..."
# Python finds 'mav0/cam0/data', processes it, and outputs to 'output_results/cam0'
python "${REPO_DIR}/scripts/preprocess_orb.py" \
        --dataset_root "$target_seq_dir" \
        --output_root "$SLURM_TMPDIR/output_results" \
        --pca_path "$MASTER_PCA"

# --- Archive Results ---
echo "Archiving results..."

# FIX: Rename the generic output folder (likely 'cam0') to the actual sequence ID
# 1. Find whatever folder Python created
generated_folder=$(ls "$SLURM_TMPDIR/output_results" | head -n 1)

# 2. Rename it to MH_01_easy
if [ "$generated_folder" != "$sequence_id" ]; then
    echo "Renaming '$generated_folder' to '$sequence_id'..."
    mv "$SLURM_TMPDIR/output_results/$generated_folder" "$SLURM_TMPDIR/output_results/$sequence_id"
fi

# 3. Archive
tar -czvf "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" \
    -C "$SLURM_TMPDIR/output_results" \
    "$sequence_id"

# --- Save ---
mkdir -p "$OUTPUT_DEST"
echo "Saving to $OUTPUT_DEST..."
rsync -av "$SLURM_TMPDIR/${sequence_id}_orb.tar.gz" "$OUTPUT_DEST/"

echo "Done."