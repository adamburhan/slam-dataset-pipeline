from .SLAMSystem import SLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.slam_systems.common import SLAMOutput
from slam_pipeline.runtime.ContainerRuntime import ContainerRuntime

from pathlib import Path
import os

class ORBSLAMSystem(SLAMSystem):
    def __init__(self, config, runtime: ContainerRuntime):
        super().__init__()
        self.config = config
        self.runtime = runtime
        
    def run(self, sequence: Sequence, output_dir: Path):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define volumes
        volumes = {
            sequence.sequence_dir: f"/data/{sequence.dataset_name}/{sequence.id}",
            output_dir: "/output",
            Path("/tmp/.X11-unix"): "/tmp/.X11-unix" # For display
        }
        
        # Define environment
        env = {
            "DISPLAY": os.environ.get('DISPLAY', ''),
            "QT_X11_NO_MITSHM": "1"
        }
        
        # Define command
        command = [
            "/dpds/ORB_SLAM2/run_slam.sh",
            sequence.dataset_name,
            sequence.id,
            "/output"
        ]
        
        image = self.config.docker_image
        
        ret_code = self.runtime.run(
            image=image,
            command=command,
            volumes=volumes,
            env=env
        )

        # check if output files are created successfully
        trajectory_file = output_dir / "track_thread_poses.txt"
        if ret_code == 0 and trajectory_file.exists():
            print(f"Trajectory file created at: {trajectory_file}")
            return SLAMOutput(trajectory_path=trajectory_file)
        else:
            print("Trajectory file was not created or container failed.")
            return None

"""
docker run -it --rm --net=host -v /media/adam/T96/ood_slam_data/datasets/KITTI/odometry_gray/sequences:/root/data -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix orbslam2 /bin/bash
"""