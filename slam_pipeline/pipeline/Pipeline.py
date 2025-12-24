from pathlib import Path
import numpy as np
from slam_pipeline.datasets.factory import get_dataset
from slam_pipeline.slam_systems.factory import get_system
from slam_pipeline.trajectories.matching import prepare_matched_pair
from slam_pipeline.trajectories.alignment import align, align_valid_only
from slam_pipeline.metrics.rpe import compute_rpe
from slam_pipeline.trajectories.trajectory import TrajFormat, fill_and_correct_trajectory


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def run_sequence(self, sequence_id):
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
        
        
        valid_ratio = matched.num_valid() / N
        
        # Check for filled frames
        num_filled = 0
        if matched.est.tracking_states is not None:
            # Assuming TRACKING_FILLED = 5
            num_filled = (matched.est.tracking_states == 5).sum()
            
        print(f"\nSeq {sequence_id}: {N} frames, {matched.num_valid()}/{N} valid ({valid_ratio:.2%})")
        if num_filled > 0:
            print(f"  Filled frames: {num_filled} ({(num_filled/N):.2%})")
            
        matched.est = matched.est.anchor_to_first_valid()
        matched.gt = matched.gt.anchor_to_first_valid()
        
        # 4. Align (Parameterized)
        align_cfg = self.cfg.pipeline.alignment
        use_sim3 = align_cfg.method == "sim3"
        
        #aligned_est, _, _, scale = align(matched.est, matched.gt, with_scale=use_sim3)
        aligned_est, _, _, scale = align_valid_only(matched, with_scale=use_sim3)
        matched.est = aligned_est
        #matched.est = fill_and_correct_trajectory(matched.est)
        # 5. Compute Metrics
        # TODO: Iterate over self.cfg.pipeline.metrics list
        rpe_trans, rpe_rot = compute_rpe(matched)
        #rpe_trans, rpe_rot = compute_rpe_normalized(matched)
        
        # 6. Convert to dense
        dense_rpe_trans = matched.to_dense_rpe(rpe_trans, num_frames=N)
        dense_rpe_rot = matched.to_dense_rpe(rpe_rot, num_frames=N)
        
        print(f"  Scale: {scale:.2f}")
        print(f"  RPE trans - mean: {np.nanmean(dense_rpe_trans):.4f}m, max: {np.nanmax(dense_rpe_trans):.4f}m")
        print(f"  Valid RPE: {(~np.isnan(dense_rpe_trans)).sum()}/{N-1}")
        
        # 7. Construct Result Dictionary
        results = {
            "sequence_id": sequence_id,
            "valid_ratio": valid_ratio,
            "scale_factor": scale,
            "rpe_trans_mean": np.nanmean(dense_rpe_trans),
            "rpe_trans_max": np.nanmax(dense_rpe_trans),
            #"trajectory_path": str(slam_output.trajectory_path),
            "dense_rpe_trans": dense_rpe_trans,
            "dense_rpe_rot": dense_rpe_rot,
            "validity_mask": ~np.isnan(dense_rpe_trans)
        }
        
        # 8. Save Results
        if self.cfg.pipeline.output.save_npz:
            self.save_results(results, output_dir)
        
        # 9. Plot Results
        if self.cfg.pipeline.output.save_plots:
            self.plot_results(results, matched, output_dir)
            
        return results

    def plot_results(self, results, matched, output_dir):
        from slam_pipeline.utils.plotting import plot_rpe, plot_trajectory
        
        # Plot RPE
        plot_rpe(
            results["dense_rpe_trans"], 
            results["dense_rpe_rot"], 
            output_dir
        )
        
        # Plot Trajectory
        plot_trajectory(
            matched.est.poses, 
            matched.gt.poses, 
            output_dir / "trajectory_plot.png"
        )
        print(f"Saved plots to {output_dir}")

    def save_results(self, results, output_dir: Path):
        """Saves the dense results to an .npz file for ML training."""
        npz_path = output_dir / "labels.npz"
        np.savez_compressed(
            npz_path,
            rpe_trans=results["dense_rpe_trans"],
            rpe_rot=results["dense_rpe_rot"],
            validity_mask=results["validity_mask"],
            scale_factor=results["scale_factor"]
        )
        print(f"Saved labels to {npz_path}")