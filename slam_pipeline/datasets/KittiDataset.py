from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.utils.transformations import line2mat
from slam_pipeline.utils.trajectories import Trajectory
from pathlib import Path
from typing import Optional
import numpy as np

class KittiDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.sequences_dir = root_dir / "sequences"
        self.poses_dir = root_dir / "poses"
        
    def list_sequences(self):
        return [seq_dir.name for seq_dir in self.sequences_dir.iterdir() if seq_dir.is_dir()]
    
    def get_sequence(self, sequence_id: str) -> Sequence:
        return Sequence(
            id=sequence_id,
            dataset_name="KITTI",
            images_dir=self.sequences_dir / sequence_id,
            ground_truth_file=self.poses_dir / f"{sequence_id}.txt",
            timestamps_file=self.sequences_dir / sequence_id / "times.txt",
        )
        
    def load_ground_truth(self, sequence: Sequence) -> Trajectory:
        """
        Load KITTI ground truth poses.

        Each line contains 12 values representing a 3x4 pose matrix.
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
                if values.size != 12:
                    raise ValueError(
                        f"Line {i} in {gt_file} has {values.size} values, expected 12"
                    )
                poses.append(line2mat(values))

        poses = np.stack(poses, axis=0).astype(np.float64)

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
