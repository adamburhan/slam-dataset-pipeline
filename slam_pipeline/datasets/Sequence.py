from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Sequence:
    id: str # "00", "01", etc.
    dataset_name: str # "KITTI"
    images_dir: Path # /data/raw/kitti/sequences/00 (contains image_0/, times.txt)
    ground_truth_file: Optional[Path] = None # /data/raw/kitti/poses/00.txt
    timestamps_file: Optional[Path] = None # /data/raw/kitti/sequences/00/times.txt
    config_file: Optional[Path] = None # /data/configs