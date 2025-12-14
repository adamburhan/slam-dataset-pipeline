from dataclasses import dataclass
import numpy as np

@dataclass
class Trajectory:
    stamps: np.ndarray  # shape (N,), timestamps in seconds
    poses: np.ndarray   # shape (N, 4, 4), homogeneous transformation matrices

    def to_evo(self):
        pass
    @staticmethod
    def from_evo(evo_traj):
        pass