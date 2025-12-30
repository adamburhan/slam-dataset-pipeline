from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.datasets.Sequence import Sequence
from slam_pipeline.trajectories.trajectory import Trajectory, interpolate_trajectory
from slam_pipeline.utils.transformations import pos_quats2SE_matrices
from pathlib import Path
from typing import Optional
import numpy as np


class EurocDataset(Dataset):
    """
    EuRoC dataset structure:
    root_dir/
    |--- MH_01_easy/
    |    |--- mav0/
    |    |    |--- cam0/
    |    |         |--- data/
    |    |         |--- data.csv
    |    |    |--- state_groundtruth_estimate0/
    |    |         |--- data.csv
    |--- MH_02_easy/
    |___ ...
    """
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        
    def list_sequences(self):
        """Return list of sequence IDs (e.g., ['MH_01_easy', 'MH_02_easy', ...])"""
        sequences = []
        for seq_dir in self.root_dir.iterdir():
            if seq_dir.is_dir() and (seq_dir / "mav0").exists():
                sequences.append(seq_dir.name)
        return sorted(sequences)
    
    def get_sequence(self, sequence_id: str) -> Sequence:
        seq_dir = self.root_dir / sequence_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
        
        return Sequence(
            id=sequence_id,
            dataset_name="euroc",
            sequence_dir=seq_dir,
            dataset_root_dir=self.root_dir,
            ground_truth_file=seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv",
            timestamps_file=None,  # Timestamps are in the GT file and image filenames
        )

    def load_frame_stamps(self, sequence: Sequence) -> np.ndarray:
        """Load image timestamps from cam0/data.csv"""
        if sequence._frame_stamps is None:
            csv_path = sequence.sequence_dir / "mav0" / "cam0" / "data.csv"
            
            timestamps = []
            with open(csv_path, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split(",")
                    if len(parts) >= 1:
                        timestamp_ns = int(parts[0])
                        timestamps.append(timestamp_ns / 1e9)
            
            sequence._frame_stamps = np.array(timestamps, dtype=np.float64)
            sequence._num_frames = len(sequence._frame_stamps)
        
        return sequence._frame_stamps
    
    def _load_raw_ground_truth(self, sequence: Sequence) -> Trajectory:
        """
        Load EuRoC ground truth poses.
        
        GT file format (data.csv):
        #timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,v_RS_R_x,v_RS_R_y,v_RS_R_z,...
        
        Returns a Trajectory with timestamps and SE(3) poses.
        """
        gt_file = sequence.ground_truth_file
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        timestamps = []
        poses_raw = []  # (tx, ty, tz, qx, qy, qz, qw)
        
        with open(gt_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.strip().split(",")
                if len(values) < 8:
                    continue
                
                timestamp_ns = int(values[0])
                tx, ty, tz = float(values[1]), float(values[2]), float(values[3])
                qw, qx, qy, qz = float(values[4]), float(values[5]), float(values[6]), float(values[7])
                
                timestamps.append(timestamp_ns / 1e9)  # Convert ns to seconds
                poses_raw.append([tx, ty, tz, qx, qy, qz, qw])  # Note: reorder quat to match pos_quats2SE_matrices
        
        stamps = np.array(timestamps, dtype=np.float64)
        poses_raw = np.array(poses_raw, dtype=np.float64)
        poses = pos_quats2SE_matrices(poses_raw)  # (N, 7) -> (N, 4, 4)
        
        frame_ids = np.arange(len(stamps), dtype=np.int32)
        
        return Trajectory(
            stamps=stamps,
            poses=poses,
            frame_ids=frame_ids,
        )

    def load_ground_truth(self, sequence: Sequence, interpolate_to_images: bool = True) -> Trajectory:
        """Load EuRoC ground truth, optionally interpolated to image timestamps."""
        
        # Load raw GT (high frequency)
        gt_raw = self._load_raw_ground_truth(sequence)
        
        if interpolate_to_images:
            image_stamps = self.load_frame_stamps(sequence)
            frame_ids = np.arange(len(image_stamps), dtype=np.int32)
            return interpolate_trajectory(gt_raw, image_stamps, frame_ids)
        
        return gt_raw