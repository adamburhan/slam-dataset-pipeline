from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem
from slam_pipeline.datasets.Sequence import Sequence

def main():
    print("Hello from slam-dataset-pipeline!")
    
    sequence = Sequence(
        dataset_name="KITTI", 
        id="04", 
        images_dir="/media/adam/T97/ood_slam_data/datasets/KITTI/odometry_gray/sequences",
        timestamps_file="",
        ground_truth_file="",
        config_file="",
    )
    
    orb_slam = ORBSLAMSystem()
    orb_slam.run(sequence, "/root/data/output_test/")


if __name__ == "__main__":
    main()
