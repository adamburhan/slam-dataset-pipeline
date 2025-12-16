from .SLAMSystem import SLAMSystem
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.slam_systems.common import SLAMOutput

from pathlib import Path
import subprocess
import os

class ORBSLAMSystem(SLAMSystem):
    def __init__(self):
        super().__init__()
        
    def run(self, sequence: Sequence, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--net=host",
                    "-v",
                    f"{sequence.images_dir}:/data/{sequence.dataset_name}/{sequence.id}",
                    "-v",
                    f"{output_dir}:/output",
                    "-e",
                    f"DISPLAY={os.environ.get('DISPLAY', '')}",
                    "-e",
                    "QT_X11_NO_MITSHM=1",
                    "-v",
                    "/tmp/.X11-unix:/tmp/.X11-unix",
                    "orbslam2",
                    "/dpds/ORB_SLAM2/run_slam.sh",
                    sequence.dataset_name,
                    sequence.id,
                    "/output"  # Use container path, not host path
                ], 
                capture_output=True, 
                text=True, 
                check=True
            )

            # check if output files are created successfully
            trajectory_file = output_dir / "track_thread_poses.txt"
            if trajectory_file.exists():
                print(f"Trajectory file created at: {trajectory_file}")
                return SLAMOutput(trajectory_path=trajectory_file)
            else:
                print("Trajectory file was not created.")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                print("Return code:", result.returncode)
                return None
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print("Error output:", e.stderr)

"""
docker run -it --rm --net=host -v /media/adam/T96/ood_slam_data/datasets/KITTI/odometry_gray/sequences:/root/data -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix orbslam2 /bin/bash
"""