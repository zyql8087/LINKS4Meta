import numpy as np

def compute_angle(p1, p2, p3):
    """
    Compute angle at p2 formed by p1-p2-p3
    Returns angle in radians in range [0, pi]
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute cosine similarity
    norm_v1 = np.linalg.norm(v1, axis=-1)
    norm_v2 = np.linalg.norm(v2, axis=-1)
    
    # Avoid division by zero
    denominator = norm_v1 * norm_v2 + 1e-8
    
    cos_angle = np.sum(v1 * v2, axis=-1) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)

def normalize_trajectory(traj):
    """
    Normalize trajectory to [0, 1] range based on its own min/max.
    Args:
        traj: (n_steps, 2) array
    Returns:
        normalized_traj: (n_steps, 2) array
    """
    traj_min = traj.min(axis=0)
    traj_max = traj.max(axis=0)
    traj_range = traj_max - traj_min
    traj_range[traj_range == 0] = 1.0  # Avoid division by zero
    return (traj - traj_min) / traj_range

def extract_kinematics(sample):
    """
    Extract normalized kinematics curves from a sample dictionary.
    
    Args:
        sample: Dictionary containing 'analysis' with 'x_sol' and keypoint indices.
        
    Returns:
        foot_traj_norm: (n_steps, 2) - Normalized foot trajectory [0, 1]
        knee_angle_norm: (n_steps,) - Normalized knee angle [0, 1] (angle/pi)
        ankle_angle_norm: (n_steps,) - Normalized ankle angle [0, 1] (angle/pi)
    """
    analysis = sample['analysis']
    
    # x_sol shape: (num_nodes, n_steps, 2)
    x_sol = analysis['x_sol']
    
    hip_idx = analysis['hip']
    knee_idx = analysis['knee']
    ankle_idx = analysis['ankle']
    foot_idx = analysis['foot']
    
    # Transpose to (n_steps, num_nodes, 2) for easier indexing
    x_sol_t = np.transpose(x_sol, (1, 0, 2))
    
    # 1. Foot trajectory
    foot_traj = x_sol_t[:, foot_idx, :]
    foot_traj_norm = normalize_trajectory(foot_traj)
    
    # 2. Knee angle (Hip-Knee-Ankle)
    knee_angle = compute_angle(
        x_sol_t[:, hip_idx, :],
        x_sol_t[:, knee_idx, :],
        x_sol_t[:, ankle_idx, :]
    )
    knee_angle_norm = knee_angle / np.pi
    
    # 3. Ankle angle (Knee-Ankle-Foot)
    ankle_angle = compute_angle(
        x_sol_t[:, knee_idx, :],
        x_sol_t[:, ankle_idx, :],
        x_sol_t[:, foot_idx, :]
    )
    ankle_angle_norm = ankle_angle / np.pi
    
    return foot_traj_norm, knee_angle_norm, ankle_angle_norm
