import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_histogram(data, title, xlabel, ylabel, output_path, log_scale=False):
    plt.figure()
    if log_scale:
        # Filter out zeros/negatives for log scale
        data = [x for x in data if x > 0]
        plt.hist(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data)), 50), color='blue', alpha=0.7)
        plt.xscale('log')
    else:
        plt.hist(data, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def compute_stats(values):
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'rmse': float(np.sqrt(np.mean(arr ** 2))),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr),
        'quantiles': {
            '25': float(np.percentile(arr, 25)),
            '75': float(np.percentile(arr, 75)),
            '90': float(np.percentile(arr, 90)),
            '95': float(np.percentile(arr, 95)),
            '99': float(np.percentile(arr, 99)),
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing results")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    aggregate_data = []
    all_rpe_trans_valid = []
    all_rpe_rot_valid = []
    all_rpe_trans_exists = []
    all_rpe_rot_exists = []

    print(f"Searching for metrics.json in {results_dir}...")
    metric_files = list(results_dir.rglob("metrics.json"))
    print(f"Found {len(metric_files)} sequences.")

    for result_file in metric_files:
        # The parent folder is the sequence folder (e.g., .../P001 or .../00)
        seq_folder = result_file.parent 

        # 1. Load Metrics
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                aggregate_data.append(data)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
            continue
        
        # 2. Check for failure status before looking for CSV
        if data.get('status') == 'failed':
            # Failed runs won't have labels.csv, skip CSV processing
            continue

        # 3. Load CSV Labels
        csv_path = seq_folder / "labels.csv"
        if not csv_path.exists():
            print(f"Warning: labels.csv missing for successful run {seq_folder}")
            continue

        try:
            df = pd.read_csv(csv_path)
            valid = df[df['valid'] == 1]
            exists = df[df['exists'] == 1]
            all_rpe_trans_valid.extend(valid['rpe_trans'].tolist())
            all_rpe_rot_valid.extend(valid['rpe_rot'].tolist())
            all_rpe_trans_exists.extend(exists['rpe_trans'].tolist())
            all_rpe_rot_exists.extend(exists['rpe_rot'].tolist())
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    # Check if we found data
    if not aggregate_data:
        print("No data found. Exiting.")
        return

    # Handle empty lists before computing stats (in case all runs failed)
    if len(all_rpe_trans_valid) == 0:
        print("No valid RPE data found (all runs failed or no valid frames).")
        stats_valid = {'trans': {}, 'rot': {}} # Or appropriate empty placeholder
        stats_exists = {'trans': {}, 'rot': {}}
    else:
        stats_valid = {
            'trans': compute_stats(all_rpe_trans_valid),
            'rot': compute_stats(all_rpe_rot_valid)
        }

        stats_exists = {
            'trans': compute_stats(all_rpe_trans_exists),
            'rot': compute_stats(all_rpe_rot_exists)
        }

    # Build summary
    summary = {
        'dataset': results_dir.parent.name,  # e.g., "kitti"
        'system': results_dir.name,           # e.g., "orbslam2"
        'num_sequences': len(aggregate_data),
        'num_successful': sum(1 for d in aggregate_data if d.get('status') == 'success'),
        'rpe_valid': stats_valid,
        'rpe_exists': stats_exists,
    }

    # Save summary.json
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {results_dir / 'summary.json'}")

    # Build comparison table
    rows = []
    for data in aggregate_data:
        # Handle failed runs in table
        if data.get('status') == 'failed':
             rows.append({
                'sequence_id': data['sequence_id'],
                'status': 'failed',
                'error': data.get('error', 'unknown'),
                'rpe_trans_rmse': None, # Fill Nones for alignment
                'rpe_trans_mean': None,
                'rpe_rot_rmse': None,
                'rpe_rot_mean': None,
                'tracking_rate': None,
                'filled_frames': None,
                'scale_factor': None,
                'total_frames': None,
            })
        else:
            rows.append({
                'sequence_id': data['sequence_id'],
                'status': data['status'],
                'rpe_trans_rmse': data['rpe_valid']['trans']['rmse'],
                'rpe_trans_mean': data['rpe_valid']['trans']['mean'],
                'rpe_rot_rmse': data['rpe_valid']['rot']['rmse'],
                'rpe_rot_mean': data['rpe_valid']['rot']['mean'],
                'tracking_rate': data['tracking']['tracking_rate'],
                'filled_frames': data['tracking']['filled_frames'],
                'scale_factor': data['scale_factor'],
                'total_frames': data['tracking']['total_frames'],
            })

    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(results_dir / "comparison_table.csv", index=False)
    print(f"Saved comparison table to {results_dir / 'comparison_table.csv'}")

    # Plot histograms (Only if we have data)
    if len(all_rpe_trans_valid) > 0:
        (results_dir / "plots").mkdir(exist_ok=True)
        for scale in ['log', 'linear']:
            plot_histogram(
                all_rpe_trans_valid,
                title="RPE Translation (Valid Frames)",
                xlabel="RPE Translation (m)",
                ylabel="Frequency",
                output_path=results_dir / "plots" / f"rpe_trans_valid_{scale}.png",
                log_scale=(scale == 'log')
            )

            plot_histogram(
                all_rpe_rot_valid,
                title="RPE Rotation (Valid Frames)",
                xlabel="RPE Rotation (deg)",
                ylabel="Frequency",
                output_path=results_dir / "plots" / f"rpe_rot_valid_{scale}.png",
                log_scale=(scale == 'log')
            )

            plot_histogram(
                all_rpe_trans_exists,
                title="RPE Translation (Valid & filled Frames)",
                xlabel="RPE Translation (m)",
                ylabel="Frequency",
                output_path=results_dir / "plots" / f"rpe_trans_exists_{scale}.png",
                log_scale=(scale == 'log')
            )

            plot_histogram(
                all_rpe_rot_exists,
                title="RPE Rotation (Valid & filled Frames)",
                xlabel="RPE Rotation (deg)",
                ylabel="Frequency",
                output_path=results_dir / "plots" / f"rpe_rot_exists_{scale}.png",
                log_scale=(scale == 'log')
            )

if __name__ == "__main__":
    main()