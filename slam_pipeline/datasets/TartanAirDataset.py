""" 
See dataset documentation for details on TartanAir dataset structure, sampling, etc.
https://tartanair.org/modalities.html#
"""
from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.trajectories.trajectory import Trajectory
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from pathlib import Path
from typing import Optional
import numpy as np

class TartanAirDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.sequences_dir = root_dir / "Easy"
        self.poses_dir = root_dir / "Easy" / "P000"
        
    def list_sequences(self):
        return [seq_dir.name for seq_dir in self.sequences_dir.iterdir() if seq_dir.is_dir()]
    
    def get_sequence(self, sequence_id: str) -> Sequence:
        seq_dir = self.sequences_dir / sequence_id
        return Sequence(
            id=sequence_id,
            dataset_name="TartanAir",
            sequence_dir=seq_dir,
            ground_truth_file=self.poses_dir / f"{sequence_id}.txt",
            timestamps_file=seq_dir / "times.txt",
        )

    def load_ground_truth(self, sequence: Sequence) -> Trajectory:
        """
        Load TartanAir ground truth poses.

        Each line contains 8 values representing a pose in TUM format (timestamp tx ty tz qx qy qz qw).
        Returns a Trajectory with timestamps and SE(3) poses.
        """
        gt_file = sequence.ground_truth_file
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

        # Load poses
        poses = []
        with open(gt_file, "r") as f:
            for i, line in enumerate(f):
                values = np.fromstring(line, sep=" ")
                if values.size != 8:
                    raise ValueError(
                        f"Line {i} in {gt_file} has {values.size} values, expected 8"
                    )
                poses.append(values[1:])  # Exclude timestamp

        #poses = np.stack(poses, axis=0).astype(np.float64)
        poses = pos_quats2SE_matrices(np.stack(poses, axis=0).astype(np.float64))  # (N,7) -> (N,4,4)

        # Load timestamps
        timestamps_file = sequence.timestamps_file
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

        stamps = np.loadtxt(timestamps_file, dtype=np.float64)
        stamps = np.atleast_1d(stamps)

        if len(stamps) != poses.shape[0]:
            raise ValueError(
                f"GT timestamps ({len(stamps)}) and poses ({poses.shape[0]}) length mismatch"
            )

        frame_ids = np.arange(len(stamps), dtype=np.int32)

        return Trajectory(
            stamps=stamps,
            poses=poses,
            frame_ids=frame_ids,
        )
