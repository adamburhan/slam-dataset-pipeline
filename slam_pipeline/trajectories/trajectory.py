from dataclasses import dataclass
import numpy as np
from pathlib import Path
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
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