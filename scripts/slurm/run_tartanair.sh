#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-337%10
#SBATCH --output=logs/tartan_%x_%A_%a.out

set -e

module load python/3.12
module load apptainer

REPO_DIR="/home/adamb14/repos/slam-dataset-pipeline"
source "${REPO_DIR}/.venv/bin/activate"

# PATHS
SEQUENCE_LIST="${REPO_DIR}/lists/tartanair.txt"
DATASET_ROOT="/scratch/adamb14/ood_slam/datasets/tartanair_zipped"
CONFIG_FILE="${REPO_DIR}/configs/experiments/tartanair_orbslam2.yaml"
OUTPUT_ROOT="/scratch/adamb14/ood_slam/results/tartanair/orbslam2"

# 1. Get the ID (e.g., amusement/Easy/P001)
sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEQUENCE_LIST")

if [ -z "$sequence_id" ]; then
    echo "Error: Could not retrieve sequence_id from line $((SLURM_ARRAY_TASK_ID + 1))"
    exit 1
fi

echo "=== Job $SLURM_ARRAY_TASK_ID: $sequence_id ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# 2. Parse ID components
# Split "amusement/Easy/P001" into components
IFS='/' read -r domain difficulty seq_name <<< "$sequence_id"

# 3. Stage data to local SSD
echo "Staging data..."

# Create nested directory structure in TMPDIR
# This mimics the persistent storage structure so the dataset class finds it
mkdir -p "$SLURM_TMPDIR/$domain/$difficulty"

# Copy the specific zip file
ZIP_PATH="${DATASET_ROOT}/${domain}/${difficulty}/${seq_name}.zip"

if [ ! -f "$ZIP_PATH" ]; then
    echo "Error: Zip file not found at $ZIP_PATH"
    exit 1
fi

echo "Copying $ZIP_PATH to local scratch..."
cp "$ZIP_PATH" "$SLURM_TMPDIR/$domain/$difficulty/"

# Unzip
# This extracts into $SLURM_TMPDIR/$domain/$difficulty/$seq_name
unzip -q "$SLURM_TMPDIR/$domain/$difficulty/${seq_name}.zip" -d "$SLURM_TMPDIR/$domain/$difficulty/"

echo "Staged to $SLURM_TMPDIR:"
ls -R "$SLURM_TMPDIR" | head -n 20 # Limit output

# 4. Prepare Local Output
# We point the pipeline to write results here first
LOCAL_OUTPUT_ROOT="$SLURM_TMPDIR/results"
mkdir -p "$LOCAL_OUTPUT_ROOT"

# 5. Run Pipeline
srun python scripts/run_sequence_cli.py \
    --config "$CONFIG_FILE" \
    --sequence_id "$sequence_id" \
    --dataset_root "$SLURM_TMPDIR" \
    --output_dir "$LOCAL_OUTPUT_ROOT"

# 6. Copy results back
# Pipeline output logic: Path(output_dir) / sequence_id
# Since sequence_id is "amusement/Easy/P001", results will be deep inside LOCAL_OUTPUT_ROOT

GENERATED_RESULT_DIR="$LOCAL_OUTPUT_ROOT/$sequence_id"
PERSISTENT_RESULT_DIR="${OUTPUT_ROOT}/${sequence_id}"

mkdir -p "$PERSISTENT_RESULT_DIR"

if [ -d "$GENERATED_RESULT_DIR" ]; then
    echo "Copying results from $GENERATED_RESULT_DIR to $PERSISTENT_RESULT_DIR"
    cp -r "$GENERATED_RESULT_DIR/"* "$PERSISTENT_RESULT_DIR/"
else
    echo "Error: Expected output directory $GENERATED_RESULT_DIR does not exist."
    echo "Listing local results:"
    ls -R "$LOCAL_OUTPUT_ROOT"
    exit 1
fi

echo "End: $(date)"