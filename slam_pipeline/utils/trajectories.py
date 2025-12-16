from dataclasses import dataclass
import numpy as np
from pathlib import Path
from slam_pipeline.utils.transformations import pos_quats2SE_matrices

@dataclass
class Trajectory:
    stamps: np.ndarray  # shape (N,), timestamps in seconds
    poses: np.ndarray   # shape (N, 4, 4), homogeneous transformation matrices
    frame_ids: np.ndarray | None = None  # optional, shape (N,), frame IDs
    tracking_states: np.ndarray | None = None  # optional, shape (N,), tracking states

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
    """

    if format == "tracking_csv_v1":
        return _load_tracking_csv_v1(file_path)
    elif format == "tum":
        return _load_tum(file_path)
    else:
        raise ValueError(f"Unknown trajectory format: {format}")
    
def _load_tracking_csv_v1(file_path: Path) -> Trajectory:
    """
    Expected format per row:
    frame_id timestamp tracking_state tx ty tz qx qy qz qw
    """
    data = np.atleast_2d(np.loadtxt(file_path))

    frame_ids = data[:, 0].astype(np.int32)
    stamps = data[:, 1].astype(np.float64)
    tracking_states = data[:, 2].astype(np.int32)

    poses = pos_quats2SE_matrices(data[:, 3:10])  # (N,7) -> (N,4,4)

    return Trajectory(
        stamps=stamps,
        poses=poses,
        frame_ids=frame_ids,
        tracking_states=tracking_states
    )

def _load_tum(file_path: Path) -> Trajectory:
    pass