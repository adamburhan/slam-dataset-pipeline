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
    

if __name__ == "__main__":
    main()
