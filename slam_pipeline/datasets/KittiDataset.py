from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from pathlib import Path

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