# Copyright (c) 2020 Carnegie Mellon University, Wenshan Wang <wenshanw@andrew.cmu.edu>
# For License information please see the LICENSE file in the root directory.
# Cridit: Xiangwei Wang https://github.com/TimingSpace

import numpy as np
from scipy.spatial.transform import Rotation as R

def line2mat(line_data):
    """Convert KITTI pose format (12 values) to 4x4 homogeneous matrix.
    
    Args:
        line_data: Array of 12 values representing 3x4 transformation matrix
        
    Returns:
        4x4 homogeneous transformation matrix as np.ndarray
    """
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return mat

def motion2pose(data):
    """Convert relative motions to absolute poses.
    
    Args:
        data: List or array of (N, 4, 4) relative transformation matrices
        
    Returns:
        List of (N+1, 4, 4) absolute poses (includes identity as first pose)
    """
    data_size = len(data)
    all_pose = []
    all_pose.append(np.eye(4,4))
    pose = np.eye(4,4)
    for i in range(0,data_size):
        pose = pose @ data[i]  # Use @ instead of .dot()
        all_pose.append(pose.copy())  # Make copy to avoid reference issues
    return all_pose

def pose2motion(data):
    """Convert absolute poses to relative motions.
    
    Args:
        data: Array or list of (N, 4, 4) absolute pose matrices
        
    Returns:
        np.ndarray of shape (N-1, 4, 4) relative transformation matrices
    """
    data_size = len(data)
    all_motion = []
    for i in range(0,data_size-1):
        motion = np.linalg.inv(data[i]) @ data[i+1]  # Use @ for matrix mult
        all_motion.append(motion)

    return np.array(all_motion)  # N-1 x 4 x 4

def SE2se(SE_data):
    """Convert SE(3) matrix to se(3) Lie algebra representation.
    
    Args:
        SE_data: 4x4 homogeneous transformation matrix
        
    Returns:
        6D vector [tx, ty, tz, rx, ry, rz] in Lie algebra
    """
    result = np.zeros((6))
    result[0:3] = SE_data[0:3, 3]  # Extract translation (already 1D for ndarray)
    result[3:6] = SO2so(SE_data[0:3, 0:3])  # Extract rotation as axis-angle
    return result
    
def SO2so(SO_data):
    """Convert SO(3) rotation matrix to so(3) rotation vector (axis-angle).
    
    Args:
        SO_data: 3x3 rotation matrix
        
    Returns:
        3D rotation vector (axis * angle)
    """
    return R.from_matrix(SO_data).as_rotvec()

def so2SO(so_data):
    """Convert so(3) rotation vector to SO(3) rotation matrix.
    
    Args:
        so_data: 3D rotation vector (axis * angle)
        
    Returns:
        3x3 rotation matrix
    """
    return R.from_rotvec(so_data).as_matrix()

def se2SE(se_data):
    """Convert se(3) Lie algebra to SE(3) matrix.
    
    Args:
        se_data: 6D vector [tx, ty, tz, rx, ry, rz]
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    result_mat = np.eye(4)
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3] = se_data[0:3]
    return result_mat
def se_mean(se_datas):
    """Compute mean of se(3) transformations.
    
    WARNING: This is an approximation and may give incorrect results for large rotations.
    For accurate mean on SO(3), use Fréchet mean or quaternion averaging.
    
    Args:
        se_datas: (N, 6) array of se(3) vectors
        
    Returns:
        6D mean se(3) vector
    """
    # Simple element-wise mean (works for small perturbations only)
    return np.mean(se_datas, axis=0)

def ses_mean(se_datas):
    """Compute mean of batched se(3) transformations.
    
    Args:
        se_datas: Batched array of se(3) vectors (batch, height, width, 6)
        
    Returns:
        Array of mean se(3) vectors per batch
    """
    se_datas = np.array(se_datas)
    se_datas = np.transpose(se_datas.reshape(se_datas.shape[0],se_datas.shape[1],se_datas.shape[2]*se_datas.shape[3]),(0,2,1))
    se_result = np.zeros((se_datas.shape[0],se_datas.shape[2]))
    for i in range(0,se_datas.shape[0]):
        mean_se = se_mean(se_datas[i,:,:])
        se_result[i,:] = mean_se
    return se_result

def ses2poses(data):
    """Convert se(3) vectors to pose sequence.
    
    Args:
        data: (N, 6) array of se(3) vectors
        
    Returns:
        (N+1, 12) array of poses in KITTI format (3x4 flattened)
    """
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.eye(4,4)
    for i in range(0,data_size):
        data_mat = se2SE(data[i,:])
        pose = pose @ data_mat
        pose_line = pose[0:3,:].reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose

def SEs2ses(motion_data):
    """Convert SE(3) matrices to se(3) vectors.
    
    Args:
        motion_data: (N, 12) array in KITTI format (3x4 flattened)
        
    Returns:
        (N, 6) array of se(3) vectors
    """
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.eye(4)
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def so2quat(so_data):
    """Convert so(3) rotation vector to quaternion (manual implementation).
    
    Note: Consider using SO2quat() with scipy for better numerical stability.
    
    Args:
        so_data: 3D rotation vector (axis * angle)
        
    Returns:
        Quaternion [qx, qy, qz, qw] in scalar-last format
    """
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data*so_data))
    axis = so_data/theta
    quat=np.zeros(4)
    quat[0:3] = np.sin(theta/2)*axis
    quat[3] = np.cos(theta/2)
    return quat

def quat2so(quat_data):
    """Convert quaternion to so(3) rotation vector (manual implementation).
    
    Note: Consider using quat2SO() with scipy for better numerical stability.
    
    Args:
        quat_data: Quaternion [qx, qy, qz, qw] in scalar-last format
        
    Returns:
        3D rotation vector (axis * angle)
    """
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3]*quat_data[0:3]))
    axis = quat_data[0:3]/sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2*np.arctan2(sin_half_theta,cos_half_theta)
    so = theta*axis
    return so

def sos2quats(so_datas,mean_std=[[1],[1]]):
    """Batch convert so(3) rotation vectors to quaternions.
    
    Processes batched rotation data (e.g., from neural networks).
    
    Args:
        so_datas: Array of shape (batch, channel, height, width) with 3-channel rotation vectors
        mean_std: Mean and std for normalization (unused in current implementation)
        
    Returns:
        Array of quaternions with shape (batch, height*width, 4)
    """
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0],so_datas.shape[1],so_datas.shape[2]*so_datas.shape[3])
    so_datas = np.transpose(so_datas,(0,2,1))
    quat_datas = np.zeros((so_datas.shape[0],so_datas.shape[1],4))
    for i_b in range(0,so_datas.shape[0]):
        for i_p in range(0,so_datas.shape[1]):
            so_data = so_datas[i_b,i_p,:]
            quat_data = so2quat(so_data)
            quat_datas[i_b,i_p,:] = quat_data
    return quat_datas

def SO2quat(SO_data):
    """Convert SO(3) rotation matrix to quaternion.
    
    Args:
        SO_data: 3x3 rotation matrix
        
    Returns:
        Quaternion in [x, y, z, w] format (scalar-last, scipy convention)
    """
    rr = R.from_matrix(SO_data)
    return rr.as_quat()

def quat2SO(quat_data):
    """Convert quaternion to SO(3) rotation matrix.
    
    Args:
        quat_data: Quaternion in [x, y, z, w] format (scalar-last, scipy convention)
        
    Returns:
        3x3 rotation matrix
    """
    return R.from_quat(quat_data).as_matrix()


def pos_quat2SE(quat_data):
    """Convert position+quaternion to SE(3) matrix.
    
    Args:
        quat_data: 7D vector [x, y, z, qx, qy, qz, qw] (position + quaternion)
        
    Returns:
        (1, 12) array in KITTI format (3x4 flattened)
    """
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.eye(4)
    SE[0:3,0:3] = SO
    SE[0:3,3] = quat_data[0:3]
    SE = SE[0:3,:].reshape(1,12)
    return SE


def pos_quats2SEs(quat_datas):
    """Convert position+quaternion array to KITTI format matrices.
    
    Args:
        quat_datas: (N, 7) array of [x, y, z, qx, qy, qz, qw]
        
    Returns:
        (N, 12) array in KITTI format (3x4 flattened per row)
    """
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


def pos_quats2SE_matrices(quat_datas):
    """Convert position+quaternion array to 4x4 transformation matrices.
    
    Args:
        quat_datas: (N, 7) array of [x, y, z, qx, qy, qz, qw]
        
    Returns:
        List of N (4, 4) homogeneous transformation matrices
    """
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3,0:3] = SO
        SE[0:3,3]   = quat[0:3]
        SEs.append(SE)
    return SEs

def SE2pos_quat(SE_data):
    """Convert SE(3) matrix to position+quaternion representation.
    
    Args:
        SE_data: 4x4 homogeneous transformation matrix
        
    Returns:
        7D vector [x, y, z, qx, qy, qz, qw] (position + quaternion in [x,y,z,w] format)
    """
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3, 0:3])
    pos_quat[:3] = SE_data[0:3, 3]  # Extract translation (already 1D for ndarray)
    return pos_quat
