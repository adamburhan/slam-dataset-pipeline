from .SLAMSystem import SLAMSystem
from slam_pipeline.datasets.Sequence import Sequence

import subprocess
import os

class ORBSLAMSystem(SLAMSystem):
    def __init__(self):
        super().__init__()
        
    def run(self, sequence: Sequence, output_dir):
        try:
            # Use 'ls -l' on Linux/macOS
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--net=host",
                    "-v",
                    "/media/adam/T96/ood_slam_data/datasets/KITTI/odometry_gray/sequences:/root/data",
                    "-e",
                    f"DISPLAY={os.environ['DISPLAY']}",
                    "-e",
                    "QT_X11_NO_MITSHM=1",
                    "-v",
                    "/tmp/.X11-unix:/tmp/.X11-unix",
                    "orbslam2",
                    "/bin/bash",
                    "-lc",
                    (
                        "cd /dpds/ORB_SLAM2 && "
                        "./Examples/Monocular/mono_kitti "
                        "Vocabulary/ORBvoc.txt "
                        "Examples/Monocular/KITTI00-02.yaml "
                        "/root/data/00"
                    ),
                ], 
                capture_output=True, text=True, check=True
            )
            # Use 'cmd /c dir' on Windows if necessary
            # result = subprocess.run(["cmd", "/c", "dir"], capture_output=True, text=True, check=True)

            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return code:", result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print("Error output:", e.stderr)

"""
docker run -it --rm --net=host -v /media/adam/T96/ood_slam_data/datasets/KITTI/odometry_gray/sequences:/root/data -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix orbslam2 /bin/bash
"""