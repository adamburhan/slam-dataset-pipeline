from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from slam_pipeline.datasets.Dataset import Dataset
from slam_pipeline.trajectories.trajectory import Trajectory, load_estimated_trajectory, TrajFormat, fill_and_correct_trajectory
from slam_pipeline.trajectories.association import associate_nearest_timestamp
from slam_pipeline.trajectories.tracking_states import is_track_valid

@dataclass
class MatchedPair:
    """
    Matched ground truth and estimated trajectories.
    
    Structure:
    - est, gt: Trajectories containing only the M successfully matched frames
    - matched_frame_ids: Maps each of the M matched poses back to original frame index [0, N)
    - valid_frame_mask: Dense (N,) boolean array indicating tracking validity for ALL frames
    
    Design rationale:
    - Matched trajectories (M,) enable efficient computation (no NaN handling)
    - Dense mask (N,) preserves frame alignment for ML dataset generation
    - Use to_dense_rpe() to convert computed metrics back to dense format
    
    Example:
        Sequence has N=100 frames
        SLAM matched M=80 frames with GT
        Of those 80, tracking valid for 75
        
        est.poses.shape = (80, 4, 4)  # Only matched
        matched_frame_ids.shape = (80,)  # e.g. [0, 1, 2, 5, 6, ...]
        valid_frame_mask.shape = (100,)  # True for 75 of the 80 matched frames
    """
    est: Trajectory
    gt: Trajectory
    valid_frame_mask: np.ndarray      # (N,) boolean - tracking valid across ALL frames
    matched_frame_ids: np.ndarray     # (M,) int64 - frame indices for matched poses
    
    def __len__(self):
        """Number of matched frames (M)"""
        return len(self.est)
    
    def num_valid(self) -> int:
        """Number of frames with valid tracking"""
        return int(self.valid_frame_mask.sum())
    
    def to_dense_rpe(
        self,
        rpe_values: np.ndarray,  # (M-1,)
        num_frames: int,
        fill_value: float = np.nan
    ) -> np.ndarray:
        """
        Convert matched RPE to dense RPE array for motion-model-filled trajectory.
        
        After motion model preprocessing, the trajectory is continuous (N frames).
        This method maps RPE from matched pairs to the dense frame sequence,
        filling only consecutive frame pairs (delta=1).
        
        Args:
            rpe_values: (M-1,) RPE errors computed on matched trajectory
            num_frames: Total frames in preprocessed sequence (N)
            fill_value: Value for frames not in matched pairs (default: NaN)
            
        Returns:
            (N-1,) array where dense[i] = RPE between frame i and i+1
            Only consecutive matched pairs are filled; gaps remain NaN.
        """
        dense = np.full(num_frames - 1, fill_value)
        
        for i in range(len(rpe_values)):
            # In the case where frame indices are not contiguous
            frame_idx_0 = self.matched_frame_ids[i]
            frame_idx_1 = self.matched_frame_ids[i + 1]
            # RPE[i] is error between frame_idx and frame_idx+1
            if frame_idx_1 == frame_idx_0 + 1:
                dense[frame_idx_0] = rpe_values[i]
        
        return dense
    
    def get_rpe_valid_mask(self) -> np.ndarray:
        """
        RPE[i] is valid iff frame i AND frame i+1 both have valid tracking.
        Returns (N-1,) boolean array.
        """
        return self.valid_frame_mask[:-1] & self.valid_frame_mask[1:]
    
    def anchor_to_first_valid(self) -> MatchedPair:
        valid_mask = self.valid_frame_mask[self.matched_frame_ids]
        if not valid_mask.any():
            raise ValueError("No valid tracking frames to anchor to.")
        k = int(np.argmax(valid_mask))
        self.est = self.est.anchor(k)
        self.gt = self.gt.anchor(k)
        return self


def prepare_matched_pair(
    dataset: Dataset,
    seq_id: str,
    est_path: Path,
    est_format: TrajFormat,
    assoc_cfg,
    fill_policy: str = "none" # none or "constant_velocity"
) -> MatchedPair:
    sequence = dataset.get_sequence(seq_id)
    gt_traj = dataset.load_ground_truth(sequence)
    est_traj = load_estimated_trajectory(est_path, est_format)
    
    # Fill and correct BEFORE association
    if fill_policy == "constant_velocity":
        est_traj = fill_and_correct_trajectory(est_traj)

    if assoc_cfg["interpolate_gt"] == True:
        raise NotImplementedError("GT interpolation not implemented yet.")
    
    _, _, est_matched, gt_matched = associate_nearest_timestamp(
        est_traj,
        gt_traj,
        max_diff=assoc_cfg["max_diff"],
        require_unique=assoc_cfg["require_unique"],
        assign_gt_frame_ids_to_est=assoc_cfg["assign_gt_frame_ids_to_est"],
        strict=assoc_cfg["strict"],
    )

    N = sequence.num_frames()

    if est_matched.frame_ids is None:
        raise ValueError("est_matched.frame_ids is None; cannot build dense mask.")

    matched_frame_ids = np.asarray(est_matched.frame_ids, dtype=np.int64)
    if matched_frame_ids.min() < 0 or matched_frame_ids.max() >= N:
        raise ValueError(f"Matched frame_ids out of range [0, {N-1}]")

    if est_matched.tracking_states is None:
        valid_mask = np.ones(len(est_matched.stamps), dtype=bool)
    else:
        valid_mask = is_track_valid(np.asarray(est_matched.tracking_states, dtype=np.int32))

    valid_frame_mask = np.zeros((N,), dtype=bool)
    valid_frame_mask[matched_frame_ids] = valid_mask

    return MatchedPair(
        est=est_matched,
        gt=gt_matched,
        valid_frame_mask=valid_frame_mask,
        matched_frame_ids=matched_frame_ids,
    )
