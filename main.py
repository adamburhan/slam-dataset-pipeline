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
    dataset = KittiDataset(Path("/media/adam/T97/ood_slam_data/datasets/KITTI/odometry_gray"))

    seqs = ["04", "01"]

    orb_slam = ORBSLAMSystem(
        backend="docker",  # later: "singularity"
        image="orbslam2",
        data_mount="/media/adam/T97/ood_slam_data/datasets",
    )

    rpe_metric = RPEMetric()  # later: configured

    for seq_id in seqs:
        sequence = dataset.get_sequence(seq_id)

        output_dir = Path(f"/home/adam/.../outputs/orb_slam2/kitti_odometry/{seq_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = orb_slam.run(sequence, output_dir)
        print(f"Trajectory saved at: {artifacts.trajectory_path}")

        gt_traj = dataset.load_ground_truth(sequence)          # or sequence.gt_path
        est_traj = load_trajectory(artifacts.trajectory_path)  # io module

        # optional later:
        # est_traj = preprocessors.apply(sequence, est_traj)

        labels = rpe_metric.compute(gt_traj, est_traj)

        labels_out = Path(f"/home/adam/.../ml_dataset/labels/orb_slam2/kitti_odometry/{seq_id}.csv")
        write_labels_csv(labels_out, labels)
"""

if __name__ == "__main__":
    main()
