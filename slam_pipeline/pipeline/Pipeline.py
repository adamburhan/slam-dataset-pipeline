from pathlib import Path
import numpy as np
import json
import subprocess
from datetime import datetime
import time
from dataclasses import asdict

from slam_pipeline.datasets.factory import get_dataset
from slam_pipeline.slam_systems.factory import get_system
from slam_pipeline.trajectories.matching import prepare_matched_pair
from slam_pipeline.trajectories.alignment import align, align_valid_only
from slam_pipeline.metrics.rpe import compute_rpe
from slam_pipeline.trajectories.trajectory import TrajFormat, fill_and_correct_trajectory

class JSONEncoder(json.JSONEncoder):
    """Custom encoder to handle Path and other non-serializable types."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def get_git_hash():
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_rpe_stats(values):
    """Compute RPE statistics with sufficient statistics for aggregation."""
    if len(values) == 0:
        return {
            "rmse": None,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "sum": 0.0,
            "sum_sq": 0.0,
            "count": 0,
        }
    
    return {
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
        "sum_sq": float(np.sum(values ** 2)),
        "count": int(len(values)),
    }


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def run_sequence(self, sequence_id):
        start_time = time.time()
        
        # 1. Setup
        dataset = get_dataset(self.cfg.dataset)
        sequence = dataset.get_sequence(sequence_id)
        N = sequence.num_frames()
        
        slam_system = get_system(self.cfg.system)
        output_dir = Path(self.cfg.pipeline.output.output_dir) / sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Run SLAM
        slam_output = slam_system.run(sequence, output_dir)
        if slam_output is None:
            print(f"SLAM failed for sequence {sequence_id}")
            self.save_metrics_json_failed(sequence_id, sequence.dataset_name, output_dir, start_time)
            return None

        # 3. Match & Fill
        loading_cfg = self.cfg.pipeline.loading
        est_format = TrajFormat.TRACKING_CSV_V1 if loading_cfg.est_format == "tracking_csv_v1" else TrajFormat.TUM
        matched = prepare_matched_pair(
            dataset=dataset,
            seq_id=sequence_id,
            est_path=slam_output.trajectory_path,
            est_format=est_format,
            assoc_cfg={
                "max_diff": loading_cfg.association.max_diff,
                "interpolate_gt": loading_cfg.association.interpolate_gt,
                "require_unique": loading_cfg.association.require_unique,
                "assign_gt_frame_ids_to_est": loading_cfg.association.assign_gt_frame_ids_to_est,
                "strict": loading_cfg.association.strict,
            },
            fill_policy=loading_cfg.fill_policy
        )
        
        tracked_frames = matched.num_valid()
        valid_ratio = tracked_frames / N
        
        # Check for filled frames
        num_filled = 0
        if matched.est.tracking_states is not None:
            num_filled = int((matched.est.tracking_states == 5).sum())
            
        print(f"\nSeq {sequence_id}: {N} frames, {tracked_frames}/{N} valid ({valid_ratio:.2%})")
        if num_filled > 0:
            print(f"  Filled frames: {num_filled} ({(num_filled/N):.2%})")
            
        matched.anchor_to_first_valid()
        
        # 4. Align (Parameterized)
        align_cfg = self.cfg.pipeline.alignment
        use_sim3 = align_cfg.method == "sim3"
        
        aligned_est, _, _, scale = align_valid_only(matched, with_scale=use_sim3, only_scale=align_cfg.only_scale)
        matched.est = aligned_est
        
        # 5. Compute Metrics
        rpe_trans, rpe_rot = compute_rpe(matched)
        
        # 6. Convert to dense
        dense_rpe_trans = matched.to_dense_rpe(rpe_trans, num_frames=N)
        dense_rpe_rot = matched.to_dense_rpe(rpe_rot, num_frames=N)
        
        valid_mask = matched.get_rpe_valid_mask()
        exists_mask = ~np.isnan(dense_rpe_trans)
        
        # Extract arrays for valid-only and exists (valid + filled)
        rpe_trans_valid = dense_rpe_trans[valid_mask]
        rpe_rot_valid = dense_rpe_rot[valid_mask]
        rpe_trans_exists = dense_rpe_trans[exists_mask]
        rpe_rot_exists = dense_rpe_rot[exists_mask]
        
        print(f"  Scale: {scale:.2f}")
        print(f"  RPE trans - mean: {np.nanmean(dense_rpe_trans):.4f}m, max: {np.nanmax(dense_rpe_trans):.4f}m")
        print(f"  Valid RPE: {valid_mask.sum()}/{N-1}")
        print(f"  Exists RPE: {exists_mask.sum()}/{N-1}")

        # 7. Construct Result Dictionary
        total_seconds = time.time() - start_time
        
        results = {
            "sequence_id": sequence_id,
            "dataset": sequence.dataset_name,
            "system": self.cfg.system.name,
            "config": asdict(self.cfg),
            "git_hash": get_git_hash(),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "scale_factor": float(scale),
            "trajectory_path": str(slam_output.trajectory_path),
            
            # Dense arrays for labels.csv
            "dense_rpe_trans": dense_rpe_trans,
            "dense_rpe_rot": dense_rpe_rot,
            "exists_mask": exists_mask,
            "valid_mask": valid_mask,
            
            # RPE stats - valid only (no filled frames)
            "rpe_valid": {
                "trans": compute_rpe_stats(rpe_trans_valid),
                "rot": compute_rpe_stats(rpe_rot_valid),
            },
            
            # RPE stats - exists (valid + filled)
            "rpe_exists": {
                "trans": compute_rpe_stats(rpe_trans_exists),
                "rot": compute_rpe_stats(rpe_rot_exists),
            },
            
            # Tracking stats
            "tracking": {
                "total_frames": N,
                "tracked_frames": tracked_frames,
                "filled_frames": num_filled,
                "tracking_rate": float(tracked_frames / N),
            },
            
            # Timing
            "timing": {
                "total_seconds": round(total_seconds, 2),
            },
        }
        
        # 8. Save Results
        if self.cfg.pipeline.output.save_labels:
            self.save_results_csv(results, output_dir)

        self.save_metrics_json(results, output_dir)
        
        # 9. Plot Results
        if self.cfg.pipeline.output.save_plots:
            self.plot_results(results, matched, output_dir)
            
        return results

    def plot_results(self, results, matched, output_dir):
        from slam_pipeline.utils.plotting import plot_rpe, plot_trajectory
        
        plot_rpe(
            results["dense_rpe_trans"], 
            results["dense_rpe_rot"], 
            output_dir
        )
        
        plot_trajectory(
            matched.est.poses, 
            matched.gt.poses, 
            output_dir / "trajectory_plot.png"
        )
        print(f"Saved plots to {output_dir}")

    def save_results_csv(self, results, output_dir: Path):
        import csv

        csv_path = output_dir / "labels.csv"
        N = len(results["dense_rpe_trans"])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "rpe_trans",
                "rpe_rot",
                "exists",
                "valid"
            ])

            for i in range(N):
                writer.writerow([
                    i,
                    results["dense_rpe_trans"][i],
                    results["dense_rpe_rot"][i],
                    int(results["exists_mask"][i]),
                    int(results["valid_mask"][i]),
                ])

        print(f"Saved labels to {csv_path}")

    def save_metrics_json(self, results, output_dir: Path):
        """Save machine-readable metrics for aggregation."""
        metrics = {
            "sequence_id": results["sequence_id"],
            "dataset": results["dataset"],
            "system": results["system"],
            "config": results["config"],
            "timestamp": results["timestamp"],
            "git_hash": results["git_hash"],
            "status": results["status"],
            "scale_factor": results["scale_factor"],
            "rpe_valid": results["rpe_valid"],
            "rpe_exists": results["rpe_exists"],
            "tracking": results["tracking"],
            "timing": results["timing"],
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=JSONEncoder)
        
        print(f"Saved metrics to {output_dir / 'metrics.json'}")

    def save_metrics_json_failed(self, sequence_id, dataset_name, output_dir: Path, start_time: float):
        """Save metrics.json for failed runs."""
        metrics = {
            "sequence_id": sequence_id,
            "dataset": dataset_name,
            "system": self.cfg.system.name,
            "config": asdict(self.cfg),
            "timestamp": datetime.now().isoformat(),
            "git_hash": get_git_hash(),
            "status": "failed",
            "error": "SLAM execution returned None",
            "timing": {
                "total_seconds": round(time.time() - start_time, 2),
            },
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=JSONEncoder)
        
        print(f"Saved failed metrics to {output_dir / 'metrics.json'}")