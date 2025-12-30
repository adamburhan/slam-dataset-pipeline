from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class Sequence:
    id: str
    dataset_name: str
    sequence_dir: Path
    dataset_root_dir: Path
    ground_truth_file: Optional[Path] = None
    timestamps_file: Optional[Path] = None
    config_file: Optional[Path] = None
    
    # Cached data (populated by Dataset methods)
    _frame_stamps: Optional[np.ndarray] = None
    _num_frames: Optional[int] = None
    
    def num_frames(self) -> int:
        if self._num_frames is None:
            raise ValueError("num_frames not set. Call dataset.load_frame_stamps() first.")
        return self._num_frames
    
    def frame_stamps(self) -> np.ndarray:
        if self._frame_stamps is None:
            raise ValueError("frame_stamps not set. Call dataset.load_frame_stamps() first.")
        return self._frame_stamps
    
    def camera_image_dir(self, cam: str = "image_0") -> Path:
        return self.sequence_dir / cam