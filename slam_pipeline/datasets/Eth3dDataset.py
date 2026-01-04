from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.trajectories.trajectory import Trajectory, interpolate_trajectory
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from pathlib import Path
from typing import Optional
import numpy as np

class Eth3dDataset(Dataset):
    """
    ETH3D SLAM dataset structure:
    root_dir/
    |--- cables_1/
    |    |--- rgb/
    |    |     11784.337488.png
    |    |___  ...
    |    |--- groundtruth.txt (TUM format: timestamp tx ty tz qx qy qz qw)
    |    |--- calibration.txt (fx fy cx cy)
    |    |--- rgb.txt (timestamp filename)
    |--- cables_2/
    |___ ...
    """
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        
    def list_sequences(self):
        """Return list of sequence IDs (e.g., ['cables_1', 'sofa_1', ...])"""
        sequences = []
        for seq_dir in self.root_dir.iterdir():
            # Check for key ETH3D file to confirm it's a sequence
            if seq_dir.is_dir() and (seq_dir / "rgb.txt").exists():
                sequences.append(seq_dir.name)
        return sorted(sequences)
    
    def get_sequence(self, sequence_id: str) -> Sequence:
        seq_dir = self.root_dir / sequence_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
        
        return Sequence(
            id=sequence_id,
            dataset_name="eth3d",
            sequence_dir=seq_dir,
            dataset_root_dir=self.root_dir,
            ground_truth_file=seq_dir / "groundtruth.txt",
            timestamps_file=seq_dir / "rgb.txt", # We will parse this file for timestamps
        )

    def load_frame_stamps(self, sequence: Sequence) -> np.ndarray:
        """
        Load image timestamps from rgb.txt.
        Format: timestamp filename
        """
        if sequence._frame_stamps is None:
            if not sequence.timestamps_file.exists():
                raise FileNotFoundError(f"Timestamp file (rgb.txt) not found: {sequence.timestamps_file}")
            
            # Read first column of rgb.txt
            timestamps = []
            with open(sequence.timestamps_file, 'r') as f:
                for line in f:
                    if line.startswith("#"): continue
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        timestamps.append(float(parts[0]))
            
            sequence._frame_stamps = np.array(timestamps, dtype=np.float64)
            sequence._num_frames = len(sequence._frame_stamps)
        
        return sequence._frame_stamps
    
    def _load_raw_ground_truth(self, sequence: Sequence) -> Trajectory:
        """
        Load ETH3D ground truth poses (TUM format).
        timestamp tx ty tz qx qy qz qw
        """
        gt_file = sequence.ground_truth_file
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        # Load data (TUM format is space separated)
        # 8 columns: time, tx, ty, tz, qx, qy, qz, qw
        data = np.loadtxt(gt_file, comments="#")
        
        timestamps = data[:, 0]
        poses_raw = data[:, 1:8] # tx ty tz qx qy qz qw
        
        # Convert to SE(3) matrices
        poses = pos_quats2SE_matrices(poses_raw)
        
        frame_ids = np.arange(len(timestamps), dtype=np.int32)
        
        return Trajectory(
            stamps=timestamps,
            poses=poses,
            frame_ids=frame_ids,
        )

    def load_ground_truth(self, sequence: Sequence, interpolate_to_images: bool = True) -> Trajectory:
        """Load ETH3D ground truth, optionally interpolated to image timestamps."""
        
        gt_raw = self._load_raw_ground_truth(sequence)
        
        if interpolate_to_images:
            image_stamps = self.load_frame_stamps(sequence)
            
            # Clip to GT range
            gt_min, gt_max = gt_raw.stamps.min(), gt_raw.stamps.max()
            
            valid_mask = (image_stamps >= gt_min) & (image_stamps <= gt_max)
            
            if valid_mask.sum() < len(image_stamps):
                print(f"Warning: {len(image_stamps) - valid_mask.sum()} images outside GT range.")
            
            clipped_stamps = image_stamps[valid_mask]
            frame_ids = np.where(valid_mask)[0].astype(np.int32)
            
            # Interpolate GT (high freq) to match image timestamps (low freq)
            return interpolate_trajectory(gt_raw, clipped_stamps, frame_ids)
        
        return gt_raw