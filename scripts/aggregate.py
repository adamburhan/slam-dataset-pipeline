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

    for seq_folder in results_dir.iterdir():
        if not seq_folder.is_dir():
            continue

        metrics_path = seq_folder / "metrics.json"
        csv_path = seq_folder / "labels.csv"
        
        # Skip folders that don't have expected files
        if not metrics_path.exists() or not csv_path.exists():
            continue

        result_file = seq_folder / "metrics.json"
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                aggregate_data.append(data)
        except FileNotFoundError:
            print(f"Warning: No metrics.json found in {seq_folder}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {result_file}: {e}")

        csv_path = seq_folder / "labels.csv"
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
        rows.append({
            'sequence_id': data['sequence_id'],
            'rpe_trans_rmse': data['rpe_valid']['trans']['rmse'],
            'rpe_trans_mean': data['rpe_valid']['trans']['mean'],
            'rpe_rot_rmse': data['rpe_valid']['rot']['rmse'],
            'rpe_rot_mean': data['rpe_valid']['rot']['mean'],
            'tracking_rate': data['tracking']['tracking_rate'],
            'filled_frames': data['tracking']['filled_frames'],
            'scale_factor': data['scale_factor'],
            'total_frames': data['tracking']['total_frames'],
            'status': data['status'],
        })

    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(results_dir / "comparison_table.csv", index=False)
    print(f"Saved comparison table to {results_dir / 'comparison_table.csv'}")

    # Plot histograms
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