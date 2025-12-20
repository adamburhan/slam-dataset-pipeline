from dataclasses import dataclass
import numpy as np
from pathlib import Path
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from slam_pipeline.trajectories.tracking_states import is_track_valid
from enum import Enum

class TrajFormat(Enum):
    TUM = "tum"
    TRACKING_CSV_V1 = "tracking_csv_v1"

@dataclass
class Trajectory:
    stamps: np.ndarray  # shape (N,), timestamps in seconds
    poses: np.ndarray   # shape (N, 4, 4), homogeneous transformation matrices
    frame_ids: np.ndarray | None = None  # optional, shape (N,), frame IDs
    tracking_states: np.ndarray | None = None  # optional, shape (N,), tracking states

    def __len__(self):
        return self.stamps.shape[0]

    def __post_init__(self):
        self.stamps = np.asarray(self.stamps, dtype=np.float64).reshape(-1) # (N,)
        self.poses = np.asarray(self.poses, dtype=np.float64) # (N,4,4)

        if self.poses.ndim != 3 or self.poses.shape[1:] != (4,4):
            raise ValueError("poses must be of shape (N,4,4)")
        if self.stamps.shape[0] != self.poses.shape[0]:
            raise ValueError("stamps and poses must have the same length")
        if not np.isfinite(self.stamps).all() or not np.isfinite(self.poses).all():
            raise ValueError("Non-finite stamps/poses found")
        if not np.all(np.diff(self.stamps) >= 0):
            raise ValueError("stamps must be non-decreasing")
        
        N = self.stamps.shape[0]
        if self.frame_ids is not None:
            self.frame_ids = np.asarray(self.frame_ids, dtype=np.int32).reshape(-1)
            if self.frame_ids.shape[0] != N:
                raise ValueError("frame_ids must have the same length as stamps/poses")
        if self.tracking_states is not None:
            self.tracking_states = np.asarray(self.tracking_states, dtype=np.int32).reshape(-1)
            if self.tracking_states.shape[0] != N:
                raise ValueError("tracking_states must have the same length as stamps/poses")

    def to_evo(self):
        pass
    @staticmethod
    def from_evo(evo_traj):
        pass

def load_estimated_trajectory(file_path: Path, format: TrajFormat) -> Trajectory:
    """
    Load estimated trajectory from file.
    """
    if format == TrajFormat.TRACKING_CSV_V1:
        return _load_tracking_csv_v1(file_path)
    elif format == TrajFormat.TUM:
        return _load_tum(file_path)
    else:
        raise ValueError(f"Unknown trajectory format: {format}")
    
def _load_tracking_csv_v1(file_path: Path) -> Trajectory:
    """
    Expected format per row:
    frame_id timestamp tracking_state tx ty tz qx qy qz qw
    """
    data = np.atleast_2d(np.loadtxt(file_path, usecols=range(10)))

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

def fill_and_correct_trajectory(traj: Trajectory) -> Trajectory:
    """
    1. Fill gaps with constant velocity model
    2. Correct reinitialized segments to match predicted pose 
    """
    if traj.tracking_states is None:
        return traj
    
    valid = is_track_valid(traj.tracking_states)
    
    if valid.all():
        return traj
    
    poses = traj.poses.copy()
    states = traj.tracking_states.copy()
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) < 2:
        raise ValueError("Not enough valid poses to fill trajectory.")
    
    # Process frame by frame
    i = 0
    while i < len(poses):
        if valid[i]:
            i += 1
            continue
        
        # Found start of gap
        gap_start = i 
        while i < len(poses) and not valid[i]:
            i += 1
        gap_end = i # First valid frame after gap (or len if none)
        
        # Get the last 2 valid poses before gap for velocity
        prev_valid = valid_indices[valid_indices < gap_start]
        if len(prev_valid) < 2:
            # can't estimate velocity, hold last
            for j in range(gap_start, min(gap_end, len(poses))):
                poses[j] = poses[prev_valid[-1]]
                states[j] = 5 # filled state
            continue
        
        i1, i2 = prev_valid[-2], prev_valid[-1]
        delta = np.linalg.inv(poses[i1]) @ poses[i2]
        
        # Fill gap with constant velocity
        T_pred = poses[i2].copy()
        for j in range(i2 + 1, gap_end):
            T_pred = T_pred @ delta
            poses[j] = T_pred.copy()
            states[j] = 5 
            
        # Correct subsequent segment if exists
        if gap_end < len(poses):
            # T_pred is now prediction for frame gap_end
            # poses[gap_end] is the reinitialized pose (different frame)
            T_reinit = poses[gap_end].copy()
            T_corr = T_pred @ np.linalg.inv(T_reinit)
            
            # Apply correction to all subsequent poses
            for j in range(gap_end, len(poses)):
                poses[j] = T_corr @ poses[j]
    
    return Trajectory(
        stamps=traj.stamps,
        poses=poses,
        frame_ids=traj.frame_ids,
        tracking_states=states
    )
        