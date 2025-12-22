import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_trajectory(est_poses, gt_poses, save_path: Path):
    """
    Plots the top-down trajectory (X vs Z).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # KITTI / ORB-SLAM convention: X=Right, Y=Down, Z=Forward
    # Top-down view is X vs Z
    
    if gt_poses is not None:
        ax.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], 'k--', linewidth=1.5, label="Ground Truth", alpha=0.6)
        
    ax.plot(est_poses[:, 0, 3], est_poses[:, 2, 3], 'b-', linewidth=1.5, label="Estimated", alpha=0.8)
    
    # Mark start/end
    ax.plot(est_poses[0, 0, 3], est_poses[0, 2, 3], 'go', markersize=8, label='Start')
    ax.plot(est_poses[-1, 0, 3], est_poses[-1, 2, 3], 'rs', markersize=8, label='End')
    
    ax.axis('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Z (meters)')
    ax.set_title('Trajectory Alignment (Top-Down)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_rpe(rpe_trans, rpe_rot, save_dir: Path):
    """
    Plots RPE over time.
    rpe_trans: Dense array of translation errors (meters)
    rpe_rot: Dense array of rotation errors (radians)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    x = np.arange(len(rpe_trans))
    
    # Translation
    # Plot valid points
    mask = ~np.isnan(rpe_trans)
    ax1.plot(x[mask], rpe_trans[mask], 'b-', linewidth=1, label='Trans. Error', alpha=0.8)
    
    # Highlight gaps/invalid
    if np.isnan(rpe_trans).any():
        # Create a collection of broken segments or just plot points? 
        # Line plot with NaNs automatically breaks, which is what we want.
        pass

    ax1.set_ylabel('Translation Error (m)')
    ax1.set_title('Relative Pose Error (RPE)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Rotation (convert to degrees)
    rpe_rot_deg = rpe_rot
    ax2.plot(x[mask], rpe_rot_deg[mask], 'r-', linewidth=1, label='Rot. Error', alpha=0.8)
    
    ax2.set_ylabel('Rotation Error (deg)')
    ax2.set_xlabel('Frame Index')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / "rpe_plot.png", dpi=150)
    plt.close(fig)
