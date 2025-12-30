from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from slam_pipeline.trajectories.tracking_states import is_track_valid
from enum import Enum
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d
import slam_pipeline.utils.lie_algebra as lie

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
        return self.poses.shape[0]

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
            
    def left_multiply(self, T: np.ndarray) -> Trajectory:
        return transform(self, T, right_mul=False, propagate=False)
    
    def right_multiply(self, T: np.ndarray, propagate: bool = False) -> Trajectory:
        return transform(self, T, right_mul=True, propagate=propagate)
    
    def scale_translation(self, scale_factor: float) -> Trajectory:
        return scale(self, scale_factor)
    
    def anchor(self, idx: int) -> Trajectory:
        T0_inv = np.linalg.inv(self.poses[idx])
        return transform(self, T0_inv, right_mul=False)
    
    def first_valid_index(self) -> int:
        if self.tracking_states is None:
            return 0
        valid = is_track_valid(self.tracking_states)
        if not valid.any():
            raise ValueError("No valid poses to anchor to")
        return int(np.argmax(valid))
        
    def anchor_to_first_valid(self) -> Trajectory:
        return self.anchor(self.first_valid_index())

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

def _scale_delta(delta: np.ndarray, steps: int) -> np.ndarray:
    """Scale relative transform for linear extrapolation."""
    t_delta = delta[:3, 3]
    rotvec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    
    delta_scaled = np.eye(4)
    delta_scaled[:3, 3] = t_delta * steps
    delta_scaled[:3, :3] = Rotation.from_rotvec(rotvec * steps).as_matrix()
    return delta_scaled

def fill_and_correct_trajectory(traj: Trajectory) -> Trajectory:
    """
    Fill tracking gaps with constant velocity and correct reinitialized segments.
    
    When SLAM loses tracking and reinitializes, poses are in a new coordinate frame.
    This function:
    1. Skips initial gap (SLAM initialization - no fill)
    2. Fills subsequent gaps using scaled constant velocity extrapolation
    3. Corrects post-gap segments to align with predicted pose
    
    Args:
        traj: Trajectory with tracking_states indicating valid/lost frames
        
    Returns:
        Trajectory with gaps filled (tracking_states=5 for filled frames)
        Initial invalid frames remain unfilled.
    """
    if traj.tracking_states is None:
        return traj
    
    poses = traj.poses.copy()
    states = traj.tracking_states.copy()
    
    # We need to update validity dynamically as we fill
    valid = is_track_valid(states)
    original_valid = valid.copy()
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
            
            # No fill, these are initialization frames
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
        for j in range(gap_start, gap_end):
            steps = j - (gap_start - 1)
            poses[j] = poses[gap_start - 1] @ _scale_delta(delta, steps)
            states[j] = 5
            valid[j] = True

        # --- CORRECTION ---
        # If there is a valid segment after this gap, it might be re-initialized.
        # We align it to our prediction.
        if gap_end < N:
            T_reinit = poses[gap_end].copy()
            # Our prediction for gap_end (one step after the filled gap)
            steps = gap_end - (gap_start - 1)
            T_pred_at_gap_end = poses[gap_start - 1] @ _scale_delta(delta, steps)
            
            # Correction matrix: T_corr @ T_reinit = T_pred
            T_corr = T_pred_at_gap_end @ np.linalg.inv(T_reinit)
            
            # Apply correction to the contiguous valid segment after the gap
            k = gap_end
            while k < N and original_valid[k]: 
                poses[k] = T_corr @ poses[k]
                k += 1
                
    return Trajectory(
        stamps=traj.stamps,
        poses=poses,
        frame_ids=traj.frame_ids,
        tracking_states=states
    )
        
        
def scale(
    traj: Trajectory, 
    scale_factor: float
) -> Trajectory:
    """
    Scales the trajectory by the given scale factor.
    
    Args:
        traj: Trajectory to be scaled
        scale_factor: Scale factor to apply
    
    Returns:
        Scaled Trajectory
    """
    scaled_poses = traj.poses.copy()
    scaled_poses[:, :3, 3] *= scale_factor
    return Trajectory(
        stamps=traj.stamps,
        poses=scaled_poses,
        frame_ids=traj.frame_ids,
        tracking_states=traj.tracking_states
    )

def transform(
    traj: Trajectory,
    T: np.ndarray,
    right_mul: bool = False,
    propagate: bool = False
) -> Trajectory:
    """
    Apply a left or right multiplicative transformation to the whole trajectory.
    
    Args:
        traj: Trajectory to be transformed
        T: 4x4 transformation matrix (e.g. SE(3) or Sim(3))
        right_mul: Whether to apply right-multiplicative
        propagate: Whether to propagate drift with RHS transformations
    
    Returns:
        Transformed Trajectory
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4,4) or not np.isfinite(T).all():
        raise ValueError("T must be a finite (4,4) matrix")

    transformed_poses = traj.poses.copy()
    
    if right_mul and not propagate:
        transformed_poses = transformed_poses @ T
    elif right_mul and propagate:
        ids = np.arange(0, traj.poses.shape[0], 1, dtype=int)
        rel_poses = [
            lie.relative_se3(traj.poses[i], traj.poses[i + 1]) @ T for i in ids[:-1]
        ]
        transformed_poses[0] = transformed_poses[0] 
        for i in ids[1:]:
            transformed_poses[i] = transformed_poses[i-1] @ rel_poses[i-1]
    else:
        transformed_poses = T[None, :, :] @ transformed_poses
    
    return Trajectory(
        stamps=traj.stamps,
        poses=transformed_poses,
        frame_ids=traj.frame_ids,
        tracking_states=traj.tracking_states
    )


def interpolate_trajectory(
    traj: Trajectory,
    target_stamps: np.ndarray,
    target_frame_ids: np.ndarray = None
) -> Trajectory:
    """
    Interpolate trajectory to target timestamps.
    
    Uses linear interpolation for translation and SLERP for rotation.
    
    Args:
        traj: Source trajectory (high-frequency, e.g., GT at 200Hz)
        target_stamps: Timestamps to interpolate to (e.g., image timestamps)
        target_frame_ids: Frame IDs for output trajectory (default: 0, 1, 2, ...)
    
    Returns:
        Interpolated trajectory at target_stamps
    """
    src_stamps = traj.stamps
    src_poses = traj.poses  # (N, 4, 4)
    
    # Check bounds - target stamps must be within source range
    if target_stamps.min() < src_stamps.min() or target_stamps.max() > src_stamps.max():
        raise ValueError(
            f"Target stamps [{target_stamps.min()}, {target_stamps.max()}] "
            f"outside source range [{src_stamps.min()}, {src_stamps.max()}]"
        )
    
    # Extract translations (N, 3)
    translations = src_poses[:, :3, 3]
    
    # Extract rotations (N, 3, 3) -> Rotation objects
    rotations = Rotation.from_matrix(src_poses[:, :3, :3])
    
    # Interpolate translations with linear interp
    trans_interp = interp1d(src_stamps, translations, axis=0, kind='linear')
    interp_translations = trans_interp(target_stamps)  # (M, 3)
    
    # Interpolate rotations with SLERP
    slerp = Slerp(src_stamps, rotations)
    interp_rotations = slerp(target_stamps)  # Rotation object with M rotations
    
    # Build output poses (M, 4, 4)
    M = len(target_stamps)
    interp_poses = np.zeros((M, 4, 4), dtype=np.float64)
    interp_poses[:, :3, :3] = interp_rotations.as_matrix()
    interp_poses[:, :3, 3] = interp_translations
    interp_poses[:, 3, 3] = 1.0
    
    if target_frame_ids is None:
        target_frame_ids = np.arange(M, dtype=np.int32)
    
    return Trajectory(
        stamps=target_stamps.copy(),
        poses=interp_poses,
        frame_ids=target_frame_ids,
    )