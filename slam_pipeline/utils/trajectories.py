from dataclasses import dataclass
import numpy as np
from pathlib import Path
from slam_pipeline.utils.transformations import pos_quat2SE

@dataclass
class Trajectory:
    stamps: np.ndarray  # shape (N,), timestamps in seconds
    poses: np.ndarray   # shape (N, 4, 4), homogeneous transformation matrices

    def to_evo(self):
        pass
    @staticmethod
    def from_evo(evo_traj):
        pass

def associate_trajectories(est_traj: Trajectory, gt_traj: Trajectory, method: str) -> tuple[int, Trajectory, Trajectory]:
    if method == "one-to-one":
        n_frames = len(gt_traj.stamps)
        n_frames_est = len(est_traj.stamps)

        start_frame = n_frames - n_frames_est if n_frames_est < n_frames else 0
        
        new_gt_poses = gt_traj.poses[start_frame:]
        new_gt_stamps = np.arange(new_gt_poses.shape[0])

        assert new_gt_poses.shape[0] == est_traj.poses.shape[0], "Trajectories have different lengths after association"
        assert np.allclose(new_gt_stamps, est_traj.stamps), "Timestamps do not match after association"

        new_gt_traj = Trajectory(stamps=new_gt_stamps, poses=new_gt_poses)
        return start_frame, est_traj, new_gt_traj
    
def load_estimated_trajectory(file_path: Path, format: str) -> Trajectory:
    """
    Load estimated trajectory from file.
    frame_id timestamp tx ty tz qx qy qz qw
    """
    data = np.loadtxt(file_path)
    stamps = data[:, 1]  # assuming second column is timestamp
    poses = []
    for row in data:
        tx, ty, tz = row[2:5]
        qx, qy, qz, qw = row[5:9]
        SE = pos_quat2SE(np.array([tx, ty, tz, qx, qy, qz, qw]))
        poses.append(SE.reshape(3, 4))
    poses = np.array(poses)
    poses_homogeneous = np.zeros((poses.shape[0], 4, 4))
    poses_homogeneous[:, :3, :] = poses
    poses_homogeneous[:, 3, 3] = 1.0
    return Trajectory(stamps=stamps, poses=poses_homogeneous)