from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.datasets.KittiDataset import KittiDataset
from slam_pipeline.trajectories.matching import prepare_matched_pair, MatchedPair
from slam_pipeline.trajectories.trajectory import TrajFormat

from pathlib import Path
import numpy as np

def main():
    # load dataset
    root_dir = Path("/Volumes/T9/ood_slam_data/datasets/KITTI/RGB")
    dataset = KittiDataset(root_dir)
    print("Available sequences:", dataset.list_sequences())
    
    seqs = ["04", "01"]

    seq_04 = dataset.get_sequence("04")
    ground_truth = dataset.load_ground_truth(seq_04)


    fmt = TrajFormat.TRACKING_CSV_V1
    trajectory_file = Path("/Volumes/T9/ood_slam_data/datasets/KITTI/odometry_gray/sequences/output_test/track_thread_poses.txt")
    
    matched_trajectories = prepare_matched_pair(
        dataset=dataset,
        seq_id="04",
        est_path=trajectory_file,
        est_format=fmt,
        assoc_cfg={
            "max_diff": 0.02,
            "require_unique": True,
            "assign_gt_frame_ids_to_est": False,
            "strict": True,
        },
    )
    
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
