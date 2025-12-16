from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.datasets.KittiDataset import KittiDataset
from slam_pipeline.utils.trajectories import load_estimated_trajectory
from slam_pipeline.association.timestamp_nearest import associate_nearest_timestamp
from pathlib import Path
import numpy as np

def main():
    root_dir = Path("/Volumes/T9/ood_slam_data/datasets/KITTI/RGB")
    dataset = KittiDataset(root_dir)
    print("Available sequences:", dataset.list_sequences())
    
    seqs = ["04", "01"]

    seq_04 = dataset.get_sequence("04")
    ground_truth = dataset.load_ground_truth(seq_04)
    print(ground_truth.poses[0])
    print(ground_truth.poses[-1])
    print(ground_truth.poses.shape)

    fmt = "tracking_csv_v1"
    trajectory_file = Path("/Volumes/T9/ood_slam_data/datasets/KITTI/odometry_gray/sequences/output_test/track_thread_poses.txt")
    est_traj = load_estimated_trajectory(trajectory_file, fmt)
    print(est_traj.poses[0])
    print(est_traj.poses[-1])
    print(est_traj.poses.shape)
    print(est_traj.tracking_states)
    print(est_traj.stamps)
    print(ground_truth.stamps)

    est_indices, gt_indices, est_matched, gt_matched = associate_nearest_timestamp(
        est_traj,
        ground_truth,
        max_diff=0.05,
        require_unique=True,
        assign_gt_frame_ids_to_est=True,
        strict=True,
    )

    print("Matched estimated trajectory:")
    print(est_matched.poses[0])
    print(est_matched.poses[-1])
    print(est_matched.poses.shape)
    print("Matched ground truth trajectory:")
    print(gt_matched.poses[0])
    print(gt_matched.poses[-1])
    print(gt_matched.poses.shape)

    print("First timestamp:", gt_matched.stamps[0])
    print("Last timestamp:", gt_matched.stamps[-1])
    print("Estimated trajectory first timestamp:", est_matched.stamps[0])
    print("Estimated trajectory last timestamp:", est_matched.stamps[-1])
    print("First frame ID:", est_matched.frame_ids[0])
    print("Last frame ID:", est_matched.frame_ids[-1])
    print("First frame ID:", gt_matched.frame_ids[0])
    print("Last frame ID:", gt_matched.frame_ids[-1])

    errs = np.abs(est_matched.stamps - gt_matched.stamps)
    print("Max time error:", errs.max())
    print("Mean time error:", errs.mean())

    print("Unique GT matches:", len(np.unique(gt_indices)), "out of", len(gt_indices))
    assert len(np.unique(gt_indices)) == len(gt_indices)



    # print(ground_truth[-1])
    
    # for seq_id in seqs:
    #     sequence = dataset.get_sequence(seq_id)
    #     print(f"Running ORB-SLAM2 on sequence {seq_id}...")
    #     output_dir = Path(f"/home/adam/Documents/MILA/projects/slam-dataset-pipeline/output/{seq_id}")
    #     output_dir.mkdir(parents=True, exist_ok=True)
        
    #     orb_slam = ORBSLAMSystem()
    #     trajectory_file = orb_slam.run(sequence, output_dir)
    #     if trajectory_file:
    #         print(f"Trajectory for sequence {seq_id} saved at: {trajectory_file}")
    #     else:
    #         print(f"Failed to generate trajectory for sequence {seq_id}.")
    
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
