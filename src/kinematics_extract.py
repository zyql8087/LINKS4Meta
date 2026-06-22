# src/kinematics_extract.py
# 运动学特征提取模块，从仿真结果中提取归一化的足部轨迹和关节角度曲线。

import numpy as np


def compute_angle(p1, p2, p3):
    """计算 p1-p2-p3 在 p2 处的夹角（弧度），形状 (...)。"""
    v1 = p1 - p2
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1, axis=-1)
    norm_v2 = np.linalg.norm(v2, axis=-1)

    denominator = norm_v1 * norm_v2 + 1e-8
    cos_angle = np.sum(v1 * v2, axis=-1) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


def normalize_trajectory(traj):
    """对轨迹每个维度做 min-max 归一化到 [0, 1]。traj 形状 (n_steps, 2)。"""
    traj_min = traj.min(axis=0)
    traj_max = traj.max(axis=0)

    traj_range = traj_max - traj_min
    traj_range[traj_range == 0] = 1.0  # 防止除零

    return (traj - traj_min) / traj_range


def extract_kinematics(sample):
    """
    从 sample['analysis'] 中提取三个归一化运动学特征：
    1. 足部轨迹 (n_steps, 2)，归一化到 [0, 1]
    2. 膝关节角度 (n_steps,)，除以 pi 归一化到 [0, 1]
    3. 踝关节角度 (n_steps,)，除以 pi 归一化到 [0, 1]

    sample['analysis'] 需包含 'x_sol'(num_nodes, n_steps, 2) 及关键点索引 hip/knee/ankle/foot。
    """
    analysis = sample['analysis']
    x_sol = analysis['x_sol']

    hip_idx = analysis['hip']
    knee_idx = analysis['knee']
    ankle_idx = analysis['ankle']
    foot_idx = analysis['foot']

    # 转置为 (n_steps, num_nodes, 2) 以便按时间步索引
    x_sol_t = np.transpose(x_sol, (1, 0, 2))

    # 足部轨迹
    foot_traj = x_sol_t[:, foot_idx, :]
    foot_traj_norm = normalize_trajectory(foot_traj)

    # 膝关节角度：hip-knee-ankle
    knee_angle = compute_angle(
        x_sol_t[:, hip_idx, :],
        x_sol_t[:, knee_idx, :],
        x_sol_t[:, ankle_idx, :]
    )
    knee_angle_norm = knee_angle / np.pi

    # 踝关节角度：knee-ankle-foot
    ankle_angle = compute_angle(
        x_sol_t[:, knee_idx, :],
        x_sol_t[:, ankle_idx, :],
        x_sol_t[:, foot_idx, :]
    )
    ankle_angle_norm = ankle_angle / np.pi

    return foot_traj_norm, knee_angle_norm, ankle_angle_norm
