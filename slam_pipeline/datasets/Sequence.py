from dataclasses import dataclass
from pathlib import Path

@dataclass
class Sequence:
    id: str
    dataset_name: str
    images_dir: Path
    timestamps_file: Path
    ground_truth_file: Path
    config_file: Path