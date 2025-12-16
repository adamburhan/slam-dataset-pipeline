from slam_pipeline.trajectories.trajectory import Trajectory
import numpy as np

def associate_nearest_timestamp(
    est: Trajectory,
    gt: Trajectory,
    max_diff: float = 1e-4,
    require_unique: bool = True,
    assign_gt_frame_ids_to_est: bool = True,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, Trajectory, Trajectory]:
    est_t = np.asarray(est.stamps, dtype=np.float64)
    gt_t  = np.asarray(gt.stamps, dtype=np.float64)

    if est_t.ndim != 1 or gt_t.ndim != 1:
        raise ValueError("Both est.stamps and gt.stamps must be 1D arrays.")
    if len(est_t) == 0 or len(gt_t) == 0:
        raise ValueError("Empty trajectory provided to association.")
    if not np.isfinite(est_t).all() or not np.isfinite(gt_t).all():
        raise ValueError("Non-finite timestamps found.")
    if not np.all(np.diff(gt_t) >= 0):
        raise ValueError("GT timestamps are not non-decreasing.")
    if not np.all(np.diff(est_t) >= 0):
        raise ValueError("Estimated timestamps are not non-decreasing.")

    used_gt = set()
    est_indices = []
    gt_indices = []

    for i, t in enumerate(est_t):
        j = int(np.searchsorted(gt_t, t, side="left"))

        candidates = []
        if j > 0:
            candidates.append(j - 1)
        if j < len(gt_t):
            candidates.append(j)

        best = min(candidates, key=lambda k: abs(gt_t[k] - t))
        dt = abs(gt_t[best] - t)

        if dt > max_diff:
            if strict:
                raise ValueError(f"No GT match within max_diff for est index {i}: dt={dt}")
            continue

        if require_unique and best in used_gt:
            if strict:
                raise ValueError(f"Duplicate GT match for gt index {best} (est index {i})")
            continue

        used_gt.add(best)
        est_indices.append(i)
        gt_indices.append(best)

    est_indices = np.asarray(est_indices, dtype=np.int64)
    gt_indices  = np.asarray(gt_indices, dtype=np.int64)

    if len(est_indices) == 0:
        raise ValueError(f"No matches found within max_diff={max_diff}. Check timestamp units/logging.")

    # Enforce monotonic match ordering (good invariant)
    if require_unique:
        if not np.all(np.diff(gt_indices) > 0):
            raise ValueError("Matched GT indices are not strictly increasing (unexpected).")
    if not np.all(np.diff(est_indices) > 0):
        raise ValueError("Matched est indices are not strictly increasing (unexpected).")

    def _subset(tr: Trajectory, idx: np.ndarray) -> Trajectory:
        return Trajectory(
            stamps=tr.stamps[idx],
            poses=tr.poses[idx],
            frame_ids=(tr.frame_ids[idx] if tr.frame_ids is not None else None),
            tracking_states=(tr.tracking_states[idx] if tr.tracking_states is not None else None),
        )

    est_new = _subset(est, est_indices)
    gt_new  = _subset(gt, gt_indices)

    if assign_gt_frame_ids_to_est and gt_new.frame_ids is not None:
        est_new.frame_ids = gt_new.frame_ids.copy()

    return est_indices, gt_indices, est_new, gt_new