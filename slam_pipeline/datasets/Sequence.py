from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Optional

@dataclass
class Sequence:
    id: str
    dataset_name: str

    # sequence root (contains image_0/, times.txt, calib, etc.)
    sequence_dir: Path

    ground_truth_file: Optional[Path] = None
    timestamps_file: Optional[Path] = None
    config_file: Optional[Path] = None

    # Optional cache fields (computed lazily)
    _frame_stamps: Optional[np.ndarray] = None
    _num_frames: Optional[int] = None

    def frame_stamps(self) -> np.ndarray:
        """
        Timestamps for the image stream (one per frame).
        For KITTI, this is times.txt.
        """
        if self._frame_stamps is None:
            if self.timestamps_file is None:
                raise ValueError("Sequence.timestamps_file is not set.")
            stamps = np.loadtxt(self.timestamps_file, dtype=np.float64)
            self._frame_stamps = np.atleast_1d(stamps)
        return self._frame_stamps

    def num_frames(self) -> int:
        """
        Number of image frames in the sequence.
        For KITTI, we trust times.txt length (maybe optionally cross-check image count later).
        """
        if self._num_frames is None:
            self._num_frames = int(self.frame_stamps().shape[0])
        return self._num_frames

    def camera_image_dir(self, cam: str = "image_0") -> Path:
        return self.sequence_dir / cam
