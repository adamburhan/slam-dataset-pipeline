def associate_one_to_one(est: Trajectory, gt: Trajectory) -> tuple[np.ndarray, Trajectory, Trajectory]:
    """
    Associate by exact matching stamps (frame indices).
    Returns matched indices (in GT frame) and the cropped trajectories.
    """
    est_idx = est.stamps.astype(int)
    gt_idx  = gt.stamps.astype(int)

    # Build a fast lookup from stamp -> pose index
    gt_map = {s: i for i, s in enumerate(gt_idx)}
    common = [s for s in est_idx if s in gt_map]

    if len(common) == 0:
        raise ValueError("No overlapping stamps between estimated and GT trajectories.")

    est_sel = np.array([np.where(est_idx == s)[0][0] for s in common])
    gt_sel  = np.array([gt_map[s] for s in common])

    est_new = Trajectory(stamps=est.stamps[est_sel], poses=est.poses[est_sel])
    gt_new  = Trajectory(stamps=gt.stamps[gt_sel], poses=gt.poses[gt_sel])

    return np.array(common), est_new, gt_new
