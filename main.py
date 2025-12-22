from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.datasets.KittiDataset import KittiDataset
from slam_pipeline.trajectories.matching import prepare_matched_pair, MatchedPair
from slam_pipeline.trajectories.trajectory import TrajFormat
from slam_pipeline.trajectories.alignment import align
from slam_pipeline.metrics.rpe import compute_rpe

from pathlib import Path
import numpy as np

def plot_trajectories(traj1, traj2):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 12))  # Square figure for equal aspect
    
    # Plot trajectories
    ax.plot(traj1.poses[:, 0, 3], traj1.poses[:, 1, 3], 
            'b-', linewidth=2, label="Estimated", alpha=0.7)
    ax.plot(traj2.poses[:, 0, 3], traj2.poses[:, 1, 3], 
            'r--', linewidth=2, label="Ground Truth", alpha=0.7)
    
    # Mark start and end points
    ax.plot(traj1.poses[0, 0, 3], traj1.poses[0, 1, 3], 
            'go', markersize=10, label='Start', zorder=5)
    ax.plot(traj1.poses[-1, 0, 3], traj1.poses[-1, 1, 3], 
            'rs', markersize=10, label='End', zorder=5)
    
    # CRITICAL: Equal aspect ratio so 1m horizontal = 1m vertical on screen
    ax.axis('equal')
    
    # Labels and formatting
    ax.set_xlabel('X (meters) - Lateral', fontsize=12)
    ax.set_ylabel('Z (meters) - Forward', fontsize=12)
    ax.set_title('Top-Down Trajectory Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
       
    plt.tight_layout()
    plt.show()

def main():
    root_dir = Path("/media/adam/T97/ood_slam_data/datasets/KITTI/RGB")
    dataset = KittiDataset(root_dir)
    print("Available sequences:", dataset.list_sequences())
    
    seqs = ["04", "01", "00", "02", "05", "06", "07", "08", "09", "10"]
    
    for seq_id in seqs:
        seq = dataset.get_sequence(seq_id)
        N = seq.num_frames()
        
        # 1. Load and match (with fill & correct)
        matched = prepare_matched_pair(
            dataset=dataset,
            seq_id=seq_id,
            est_path=root_dir / "sequences" / seq_id / "frame_stats.txt",
            est_format=TrajFormat.TRACKING_CSV_V1,
            assoc_cfg={
                "max_diff": 0.0002, 
                "require_unique": True,
                "assign_gt_frame_ids_to_est": True, 
                "strict": True,
                "interpolate_gt": False
            },
            fill_policy="constant_velocity",  # NEW
        )
        
        valid_ratio = matched.num_valid() / N
        print(f"\nSeq {seq_id}: {N} frames, {matched.num_valid()}/{N} valid ({valid_ratio:.2%})")
        
        # 2. Align (scale correction)
        aligned_est, _, _, scale = align(matched.est, matched.gt, with_scale=True)
        matched.est = aligned_est
        
        # 3. Compute RPE
        rpe_trans, rpe_rot = compute_rpe(matched)
        
        # 4. Convert to dense
        dense_rpe_trans = matched.to_dense_rpe(rpe_trans, num_frames=N)
        dense_rpe_rot = matched.to_dense_rpe(rpe_rot, num_frames=N)
        
        print(f"  Scale: {scale:.2f}")
        print(f"  RPE trans - mean: {np.nanmean(dense_rpe_trans):.4f}m, max: {np.nanmax(dense_rpe_trans):.4f}m")
        print(f"  Valid RPE: {(~np.isnan(dense_rpe_trans)).sum()}/{N-1}")
        
        # # 5. Save labels
        # df = pd.DataFrame({
        #     "frame": np.arange(N - 1),
        #     "rpe_trans": dense_rpe_trans,
        #     "rpe_rot": dense_rpe_rot,
        # })
        # df.to_csv(root_dir / "sequences" / seq_id / "rpe_labels.csv", index=False)
        
        # 6. Visualize (optional)
        plot_trajectories(matched.est, matched.gt)
            
# Example of how to run the main function
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequences", nargs="+", default=["04", "01"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset = load_dataset_from_config(config.dataset)
    slam_system = load_slam_system_from_config(config.slam_system)

    metric = RPEMetric()  # later from config

    for seq_id in args.sequences:
        sequence = dataset.get_sequence(seq_id)

        run_dir = Path(config.outputs_root) / slam_system.name / dataset.name / seq_id
        run_dir.mkdir(parents=True, exist_ok=True)

        labels_out = Path(config.labels_root) / slam_system.name / dataset.name / f"{seq_id}.csv"
        labels_out.parent.mkdir(parents=True, exist_ok=True)

        if labels_out.exists() and not args.force:
            print(f"Skipping {seq_id} (labels already exist): {labels_out}")
            continue

        artifacts = slam_system.run(sequence, run_dir)

        gt_traj = dataset.load_ground_truth(sequence)         # returns Trajectory
        est_traj = load_trajectory(artifacts.trajectory_path) # returns Trajectory

        est_traj, gt_traj = associate_trajectories(est_traj, gt_traj, config.association)
        est_traj, gt_traj = align_trajectories(est_traj, gt_traj, config.alignment)  # can be 'none'

        labels = metric.compute(gt_traj, est_traj)
        write_labels_csv(labels_out, labels)
"""

if __name__ == "__main__":
    main()
