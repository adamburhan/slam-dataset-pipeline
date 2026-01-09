#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-336%20
#SBATCH --output=logs/tartan_%A_%a.out

set -e
module load python/3.12

REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
OUTPUT_DEST="/scratch/adamb14/ood_slam/datasets/tartanair/orb_images"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/tartanair_zipped" 

MASTER_PCA="/scratch/adamb14/ood_slam/datasets/weights/master_orb_pca.joblib"

source "${REPO_DIR}/.venv/bin/activate"

SEQUENCE_LIST="${REPO_DIR}/lists/tartanair.txt"

path_from_list=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

filename=$(basename "$path_from_list")                 # P001
seq_id="${filename}"                                   # P001
parent_dir=$(dirname "$path_from_list")                # .../Easy
difficulty=$(basename "$parent_dir")                   # Easy
env_dir=$(dirname "$parent_dir")                       # .../abandonedfactory_night
environment=$(basename "$env_dir")                     # abandonedfactory_night

flat_name="${environment}_${difficulty}_${seq_id}"

echo "=== Processing: $flat_name ==="

target_seq_dir="$SLURM_TMPDIR/sequences/$seq_id"
mkdir -p "$target_seq_dir"
mkdir -p "$SLURM_TMPDIR/output_results"

cp "${DATASET_ROOT}/${path_from_list}.zip" "$SLURM_TMPDIR/sequences/${seq_id}.zip"

# Unzip directly into the P001 folder
unzip -q "$SLURM_TMPDIR/sequences/${seq_id}.zip" -d "$target_seq_dir"

echo "Running Python script..."
python "${REPO_DIR}/scripts/preprocess_orb.py" \
        --dataset_root "$target_seq_dir" \
        --output_root "$SLURM_TMPDIR/output_results" \
        --pca_path "$MASTER_PCA"

echo "Archiving results..."

tar -czvf "$SLURM_TMPDIR/${flat_name}_orb.tar.gz" \
    -C "$SLURM_TMPDIR/output_results" \
    "$seq_id"

mkdir -p "$OUTPUT_DEST"
echo "Saving to $OUTPUT_DEST"
rsync -av "$SLURM_TMPDIR/${flat_name}_orb.tar.gz" "$OUTPUT_DEST/"

echo "Done."