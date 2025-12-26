#!/bin/bash
#SBATCH --account=rrg-lpaull
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/slurm_run_array_%A_%a.out

module load python/3.10
module load apptainer

sequence_list_file="sequence_list.txt"
dataset_root="/home/adamb14/scratch/ood_slam/datasets/tartanair"
config_file="configs/experiments/tartanair_orbslam2.yaml"

sequence_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $sequence_list_file)

srun python3 scripts/run_sequence_cli.py \\
                    --config $config_file \\
                    --sequence_id $sequence_id \\
                    --dataset_root $dataset_root \\
                    --output_dir "results"