import numpy as np

# ORB-SLAM2 tracking states
TRACKING_UNINITIALIZED = 0
TRACKING_OK = 2
TRACKING_LOST = 3
TRACKING_FILLED = 5  # Custom: filled by motion model

def is_track_valid(states: np.ndarray, include_filled: bool = False) -> np.ndarray:
    """
    Check if tracking states indicate valid poses.
    
    ORB-SLAM2: 0=uninitialized, 2=OK, 4=lost, 5=filled (custom)
    """
    valid = (states != TRACKING_UNINITIALIZED) & (states != TRACKING_LOST)
    if include_filled:
        valid |= (states == TRACKING_FILLED)
    return valid