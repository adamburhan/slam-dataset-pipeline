from slam_pipeline.trajectories.matching import MatchedPair
import numpy as np

def compute_rpe(matched: MatchedPair) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (M-1,) translation and rotation errors.
    """
    
    est = matched.est.poses
    gt = matched.gt.poses
    
    # Vectorized computation of RPE
    est_rel = np.linalg.inv(est[:-1]) @ est[1:]
    gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
    err = np.linalg.inv(gt_rel) @ est_rel
    
    # Translation error (meters)
    trans_err = np.linalg.norm(err[:, :3, 3], axis=1)
    
    # Rotation error (degrees)
    traces = np.trace(err[:, :3, :3], axis1=1, axis2=2)
    rot_err = np.degrees(np.arccos(np.clip((traces - 1) / 2, -1.0, 1.0)))
    
    return trans_err, rot_err