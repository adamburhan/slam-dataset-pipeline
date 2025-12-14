from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.utils.transformations import line2mat
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
            ground_truth_file=self.poses_dir / f"{sequence_id}.txt"
        )
        
    def load_ground_truth(self, sequence: Sequence) -> np.ndarray:
        """
        Load KITTI ground truth poses.

        Each line contains 12 values representing a 3x4 pose matrix.
        Returns an array of shape (N, 4, 4).
        """
        gt_file = sequence.ground_truth_file
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

        poses = []
        with open(gt_file, "r") as f:
            for i, line in enumerate(f):
                values = np.fromstring(line, sep=" ")
                if values.size != 12:
                    raise ValueError(
                        f"Line {i} in {gt_file} has {values.size} values, expected 12"
                    )

                T = line2mat(values)
                poses.append(T)

        return np.array(poses)