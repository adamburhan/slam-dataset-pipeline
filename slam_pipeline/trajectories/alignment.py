"""
Trajectory alignment using Umeyama's method (Sim3).

Credit: Core alignment algorithm from Michael Grupp's evo package
https://github.com/MichaelGrupp/evo
Licensed under BSD 3-Clause License.
"""

import numpy as np
from slam_pipeline.trajectories.trajectory import Trajectory
import slam_pipeline.utils.lie_algebra as lie

UmeyamaResult = tuple[np.ndarray, np.ndarray, float]

class GeometryException(Exception):
    pass

def scale(
    traj: Trajectory, 
    scale_factor: float
) -> Trajectory:
    """
    Scales the trajectory by the given scale factor.
    
    Args:
        traj: Trajectory to be scaled
        scale_factor: Scale factor to apply
    
    Returns:
        Scaled Trajectory
    """
    scaled_poses = traj.poses.copy()
    scaled_poses[:, :3, 3] *= scale_factor
    return Trajectory(
        stamps=traj.stamps,
        poses=scaled_poses,
        frame_ids=traj.frame_ids,
        tracking_states=traj.tracking_states
    )

def transform(
    traj: Trajectory,
    T: np.ndarray,
    right_mul: bool = False,
    propagate: bool = False
) -> Trajectory:
    """
    Apply a left or right multiplicative transformation to the whole trajectory.
    
    Args:
        traj: Trajectory to be transformed
        T: 4x4 transformation matrix (e.g. SE(3) or Sim(3))
        right_mul: Whether to apply right-multiplicative
        propagate: Whether to propagate drift with RHS transformations
    
    Returns:
        Transformed Trajectory
    """
    transformed_poses = traj.poses.copy()
    
    if right_mul and not propagate:
        transformed_poses = transformed_poses @ T
    elif right_mul and propagate:
        ids = np.arange(0, traj.poses.shape[0], 1, dtype=int)
        rel_poses = [
            lie.relative_se3(traj.poses[i], traj.poses[i + 1]) @ T for i in ids[:-1]
        ]
        transformed_poses[0] = transformed_poses[0] 
        for i in ids[1:]:
            transformed_poses[i] = transformed_poses[i-1] @ rel_poses[i-1]
    else:
        transformed_poses = T @ transformed_poses
    
    return Trajectory(
        stamps=traj.stamps,
        poses=transformed_poses,
        frame_ids=traj.frame_ids,
        tracking_states=traj.tracking_states
    )

def align(
    est_traj: Trajectory, 
    gt_traj: Trajectory, 
    with_scale: bool = False,
    only_scale: bool = False
) -> tuple[Trajectory, np.ndarray, np.ndarray, float]:
    """
    Aligns the estimated trajectory to the ground truth trajectory using Umeyama's method.
    
    Args:
        est_traj: Estimated trajectory
        gt_traj: Ground truth trajectory
        with_scale: Whether to estimate scale factor (True for monocular SLAM)
        only_scale: Whether to apply only scale (not rotation/translation)
    
    Returns:
        aligned_est: Aligned estimated trajectory
        r: Rotation matrix (3x3)
        t: Translation vector (3,)
        c: Scale factor
    """
    if len(est_traj) != len(gt_traj):
        raise GeometryException("Trajectories must have the same number of poses for alignment.")
    
    est_positions = est_traj.poses[:, :3, 3].T  # Shape (3, N)
    gt_positions = gt_traj.poses[:, :3, 3].T    # Shape (3, N)
    
    r, t, c = umeyama_alignment(est_positions, gt_positions, with_scale)

    if only_scale:
        scaled_est_traj = scale(est_traj, c)
        return scaled_est_traj, r, t, c
    elif with_scale:
        # Apply scale first, then SE(3) transformation
        scaled_est_traj = scale(est_traj, c)
        aligned_est = transform(scaled_est_traj, lie.se3(r, t))
        return aligned_est, r, t, c
    else:
        # Only apply SE(3) transformation (no scale)
        aligned_est = transform(est_traj, lie.se3(r, t))
        return aligned_est, r, t, c

def umeyama_alignment(
    x: np.ndarray, y: np.ndarray, with_scale: bool = False
) -> UmeyamaResult:
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    
    Args:
        x: mxn matrix of points, m = dimension, n = nr. of data points
        y: mxn matrix of points, m = dimension, n = nr. of data points
        with_scale: set to True to align also the scale (default: 1.0 scale)
    
    Returns:
        r: Rotation matrix (3x3)
        t: Translation vector (3,)
        c: Scale factor
    """
    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise GeometryException(
            "Degenerate covariance rank, " "Umeyama alignment is not possible"
        )

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c