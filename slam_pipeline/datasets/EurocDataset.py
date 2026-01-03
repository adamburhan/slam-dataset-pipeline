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
    |--- timestamps/
    |    |--- MH_01_easy.txt
    |    |--- MH_02_easy.txt
    |    |___ ...
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
        """Load image timestamps from ORB-SLAM2 timestamp files"""
        if sequence._frame_stamps is None:
            ts_file = self.root_dir / "timestamps" / f"{sequence.id}.txt"
            
            if not ts_file.exists():
                raise FileNotFoundError(f"Timestamp file not found: {ts_file}")
            
            timestamps = np.loadtxt(ts_file, dtype=np.int64)
            sequence._frame_stamps = timestamps.astype(np.float64)  # Keep in nanoseconds
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
                
                timestamps.append(timestamp_ns)
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
        
        gt_raw = self._load_raw_ground_truth(sequence)
        
        if interpolate_to_images:
            image_stamps = self.load_frame_stamps(sequence)
            
            # Clip to GT range — some images may be outside GT coverage
            gt_min, gt_max = gt_raw.stamps.min(), gt_raw.stamps.max()
            valid_mask = (image_stamps >= gt_min) & (image_stamps <= gt_max)
            
            if valid_mask.sum() < len(image_stamps):
                print(f"Warning: {len(image_stamps) - valid_mask.sum()} images outside GT range, skipping them")
            
            clipped_stamps = image_stamps[valid_mask]
            frame_ids = np.where(valid_mask)[0].astype(np.int32)  # Original indices
            
            return interpolate_trajectory(gt_raw, clipped_stamps, frame_ids)
        
        return gt_raw