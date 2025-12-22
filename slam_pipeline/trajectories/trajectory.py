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
    
    poses = traj.poses.copy()
    states = traj.tracking_states.copy()
    
    # We need to update validity dynamically as we fill
    valid = is_track_valid(states)
    if valid.all():
        return traj
    
    N = len(poses)
    i = 0
    
    while i < N:
        if valid[i]:
            i += 1
            continue
            
        # Found start of gap at index 'i'
        gap_start = i
        while i < N and not valid[i]:
            i += 1
        gap_end = i # First valid frame after gap (or N)
        
        # --- STRATEGY SELECTION ---
        
        # Case A: Gap is at the very beginning
        if gap_start == 0:
            if gap_end == N:
                # Whole trajectory is invalid
                return traj 
            
            # Backfill with the first valid pose (constant position backwards)
            # We cannot estimate velocity without history.
            first_valid_pose = poses[gap_end]
            for j in range(gap_start, gap_end):
                poses[j] = first_valid_pose.copy()
                states[j] = 5 # Filled
                valid[j] = True # Mark as valid for future lookups
            
            # No correction needed for the future because we aligned to it
            continue

        # Case B: We have history
        # Use the immediately preceding frames (whether original or filled)
        # to estimate velocity.
        i1, i2 = gap_start - 2, gap_start - 1
        
        if i1 < 0:
            # Only 1 frame of history (gap starts at index 1)
            # Assume zero velocity (constant position)
            delta = np.eye(4)
        else:
            # Calculate relative motion: T_{k-1} -> T_k
            # delta = T_{k-1}^{-1} @ T_k
            delta = np.linalg.inv(poses[i1]) @ poses[i2]

        # --- FILLING ---
        T_pred = poses[gap_start - 1].copy()
        for j in range(gap_start, gap_end):
            T_pred = T_pred @ delta
            poses[j] = T_pred.copy()
            states[j] = 5
            valid[j] = True

        # --- CORRECTION ---
        # If there is a valid segment after this gap, it might be re-initialized.
        # We align it to our prediction.
        if gap_end < N:
            T_reinit = poses[gap_end].copy()
            
            # Our prediction for gap_end (one step after the filled gap)
            T_pred_at_gap_end = T_pred @ delta
            
            # Correction matrix: T_corr @ T_reinit = T_pred
            T_corr = T_pred_at_gap_end @ np.linalg.inv(T_reinit)
            
            # Apply correction to the contiguous valid segment
            # We stop at the next gap (or end)
            k = gap_end
            while k < N and is_track_valid(np.array([states[k]]))[0]: # Check original state validity
                # Note: We check original state to stop at next gap. 
                # But 'valid' array is being updated. 
                # Actually, we can just check 'valid[k]' because we haven't filled future gaps yet.
                if not valid[k]: 
                    break
                
                poses[k] = T_corr @ poses[k]
                k += 1
                
    return Trajectory(
        stamps=traj.stamps,
        poses=poses,
        frame_ids=traj.frame_ids,
        tracking_states=states
    )
        