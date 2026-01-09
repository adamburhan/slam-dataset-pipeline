#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/kitti_global_pca_%j.out

set -e
module load python/3.12

# --- CONFIG ---
REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/kitti/raw/sequences" 
OUTPUT_DEST="/scratch/adamb14/ood_slam/datasets/kitti/orb_images"
PCA_DEST="/scratch/adamb14/ood_slam/datasets/weights/master_orb_pca.joblib" # Save explicitly to repo weights

source "${REPO_DIR}/.venv/bin/activate"

# Define the sequences you want to use for the Global PCA
SEQUENCES=( 00 01 02 04 05 06 07 08 09 10 )

echo "=== Job: Training Global PCA & Processing KITTI ==="
echo "Node: $(hostname)"

# --- 1. Stage ALL Data ---
# We need all data present so the PCA can sample from everything
echo "Staging data..."
mkdir -p "$SLURM_TMPDIR/sequences"
mkdir -p "$SLURM_TMPDIR/output_results"

for seq in "${SEQUENCES[@]}"; do
    echo "Staging sequence $seq..."
    cp "${DATASET_ROOT}/${seq}.tar.gz" "$SLURM_TMPDIR/sequences/"
    tar -xvzf "$SLURM_TMPDIR/sequences/${seq}.tar.gz" -C "$SLURM_TMPDIR/sequences"
done

# --- 2. Run Python (Global PCA + Processing) ---
# Since orb_pca.joblib doesn't exist in TMPDIR, the script will:
# 1. Sample 5000 images from ALL staged sequences.
# 2. Train the Global PCA.
# 3. Save it to $SLURM_TMPDIR/orb_pca.joblib.
# 4. Process all sequences using that new PCA.

echo "Running Python script (Training & Processing)..."
python "${REPO_DIR}/scripts/preprocess_orb.py" \
        --dataset_root "$SLURM_TMPDIR/sequences" \
        --output_root "$SLURM_TMPDIR/output_results" \
        --pca_path "$SLURM_TMPDIR/orb_pca.joblib"

# --- 3. Save the Master PCA ---
echo "Saving Master PCA..."
mkdir -p "$(dirname "$PCA_DEST")"
rsync -av "$SLURM_TMPDIR/orb_pca.joblib" "$PCA_DEST"

# --- 4. Archive & Save Results (The Fix) ---
echo "Archiving results..."

mkdir -p "$OUTPUT_DEST"

for seq in "${SEQUENCES[@]}"; do
    echo "Archiving $seq..."
    
    # Check if output exists (Safety)
    if [ -d "$SLURM_TMPDIR/output_results/$seq" ]; then
        tar -czvf "$SLURM_TMPDIR/${seq}_orb.tar.gz" \
            -C "$SLURM_TMPDIR/output_results" \
            "$seq"

        rsync -av "$SLURM_TMPDIR/${seq}_orb.tar.gz" "$OUTPUT_DEST/"
    else
        echo "WARNING: Output for sequence $seq not found!"
    fi
done

echo "Done. Master PCA saved to $PCA_DEST"