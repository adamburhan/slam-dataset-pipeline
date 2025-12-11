from slam_pipeline.slam_systems.ORBSLAMSystem import ORBSLAMSystem

def main():
    print("Hello from slam-dataset-pipeline!")
    
    orb_slam = ORBSLAMSystem()
    orb_slam.run(1, 2)


if __name__ == "__main__":
    main()
