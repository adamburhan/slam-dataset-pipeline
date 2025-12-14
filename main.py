from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.datasets.KittiDataset import KittiDataset
from pathlib import Path

def main():
    dataset = KittiDataset(Path("/media/adam/T97/ood_slam_data/datasets/KITTI/odometry_gray"))
    print("Available sequences:", dataset.list_sequences())
    
    seqs = ["04", "01"]
    
    for seq_id in seqs:
        sequence = dataset.get_sequence(seq_id)
        print(f"Running ORB-SLAM2 on sequence {seq_id}...")
        output_dir = Path(f"/home/adam/Documents/MILA/projects/slam-dataset-pipeline/output/{seq_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        orb_slam = ORBSLAMSystem()
        trajectory_file = orb_slam.run(sequence, output_dir)
        if trajectory_file:
            print(f"Trajectory for sequence {seq_id} saved at: {trajectory_file}")
        else:
            print(f"Failed to generate trajectory for sequence {seq_id}.")
    
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
