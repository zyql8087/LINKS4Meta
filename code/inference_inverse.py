import argparse
import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Batch, Data

# Import internal modules
from src.inverse.curve_encoder import CurveEncoder
from src.inverse.gnn_policy import GNNPolicy, policy_load_incompatibilities
from src.inverse.rl_env import load_frozen_surrogate, MechanismEnv, apply_j_operator
from src.inverse.mcts import MCTS
from src.inverse.train_il import ensure_expert_paths

def plot_kinematics_result(y_foot, y_knee, y_ankle, 
                           pred_foot, pred_knee, pred_ankle, 
                           sample_id, save_dir):
    """
    绘制对比图: 目标曲线 vs 生成机构的前向计算曲线
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. Foot Trajectory (XY Plot)
    axes[0].plot(y_foot[:, 0], y_foot[:, 1], label='Target (Desired)', color='blue', linestyle='dashed', alpha=0.7)
    axes[0].plot(pred_foot[:, 0], pred_foot[:, 1], label='Generated (Simulated)', color='red', alpha=0.7)
    axes[0].set_title(f'Foot Trajectory (Sample {sample_id})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Foot Trajectory (vs Step)
    steps = range(len(y_knee))
    axes[1].plot(steps, y_foot[:, 0], label='Target X', color='lightblue', linestyle='dashed')
    axes[1].plot(steps, y_foot[:, 1], label='Target Y', color='blue', linestyle='dashed')
    axes[1].plot(steps, pred_foot[:, 0], label='Generated X', color='lightcoral')
    axes[1].plot(steps, pred_foot[:, 1], label='Generated Y', color='red')
    axes[1].set_title('Foot Position vs Crank Step')
    axes[1].set_xlabel('Step')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Knee Angle
    axes[2].plot(steps, y_knee, label='Target', color='blue', linestyle='dashed', alpha=0.7)
    axes[2].plot(steps, pred_knee, label='Generated', color='red', alpha=0.7)
    axes[2].set_title(f'Knee Angle')
    axes[2].set_xlabel('Step')
    axes[2].legend()
    axes[2].grid(True)
    
    # 4. Ankle Angle
    axes[3].plot(steps, y_ankle, label='Target', color='blue', linestyle='dashed', alpha=0.7)
    axes[3].plot(steps, pred_ankle, label='Generated', color='red', alpha=0.7)
    axes[3].set_title(f'Ankle Angle')
    axes[3].set_xlabel('Step')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'inference_result_sample_{sample_id}.png'), dpi=150)
    plt.close()

def build_base_4bar():
    """构建基础 4 杆机构作为起始图 State"""
    # 典型的初始 4 杆机构形态
    x = np.array([
        [0.0, 0.0, 1.0, 1.0],  # 0: Ground/Crank pivot
        [0.1, 0.1, 0.0, 0.0],  # 1: Crank moving pivot
        [0.3, 0.1, 0.0, 0.0],  # 2: Coupler-Rocker pivot
        [0.4, 0.0, 1.0, 0.0],  # 3: Ground/Rocker pivot
    ], dtype=np.float32)
    pos = x[:, :2]
    edge_index = np.array([
        [0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2, 3, 0, 0, 3],
        [1, 0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 0, 3, 3, 0]
    ], dtype=np.int64)
    # 对于 4 杆图，暂时用随机的特征即可
    return Data(
        x=torch.tensor(x),
        pos=torch.tensor(pos),
        edge_index=torch.tensor(edge_index),
        knee_idx=torch.tensor([2], dtype=torch.long),
    )

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config_inverse.yaml')
    parser.add_argument('--model_type', type=str, choices=['il', 'rl'], default='rl', help='Use IL or RL saved weights')
    parser.add_argument('--output_dir', type=str, default=str(script_dir / 'demo' / 'outputs' / 'inference'))
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples to infer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Using device: {device}")
    
    # 1. Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    # 2. Init models
    curve_cfg = cfg.get('curve_encoder', {})
    curve_encoder = CurveEncoder(
        input_dim=curve_cfg.get('input_dim', 800),
        hidden_dims=curve_cfg.get('hidden_dims', [512, 256]),
        latent_dim=curve_cfg.get('latent_dim', 128)
    ).to(device)
    policy = GNNPolicy(cfg).to(device)
    
    weight_path = cfg['paths']['rl_model_output'] if args.model_type == 'rl' else cfg['paths']['il_model_output']
    if not os.path.exists(weight_path):
        print(f"[Warning] Weights '{weight_path}' not found. Tripping back to IL weights...")
        weight_path = cfg['paths']['il_model_output']
        
    if os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location=device, weights_only=True)
        policy_state = policy.load_state_dict(ckpt['policy'], strict=False)
        missing_keys, unexpected_keys = policy_load_incompatibilities(policy_state)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Policy checkpoint is incompatible with current config. Missing={missing_keys}, unexpected={unexpected_keys}"
            )
        encoder_state = curve_encoder.load_state_dict(ckpt['curve_encoder'], strict=False)
        if encoder_state.missing_keys or encoder_state.unexpected_keys:
            print(
                f"[Inference] Warning: curve encoder checkpoint has key drift. Missing={encoder_state.missing_keys}, unexpected={encoder_state.unexpected_keys}"
            )
        print(f"[Inference] Loaded `{args.model_type}` weights from '{weight_path}'")
    else:
        print("[Error] No model weights found!")
        return

    policy.eval()
    curve_encoder.eval()
    
    # 3. Load Surrogate Forward Model
    surrogate_path = cfg['paths']['forward_model']
    surrogate_cfg = cfg['paths']['config_forward']
    surrogate, _ = load_frozen_surrogate(surrogate_path, surrogate_cfg, device)
    
    # 4. Load Dataset to get Test Targets
    # 为了直观比较，我们直接从完整的专家图数据集中抽取目标曲线
    dataset_path = cfg['paths']['il_dataset_output']
    print(f"[Inference] Loading target curves from test portion of dataset: {dataset_path}")
    raw_data = ensure_expert_paths(
        pkl_path=cfg['paths']['pkl_dataset'],
        output_path=dataset_path,
        use_cached=True,
    )
    # 用最后面的数据当作测试集
    test_samples = raw_data[-args.num_samples:]
    
    # 5. Inference Loop
    for i, sample_data in enumerate(test_samples):
        print(f"\n--- Inferring Sample {i+1}/{args.num_samples} ---")
        
        target = {
            'y_foot': sample_data['y_foot'],
            'y_knee': sample_data['y_knee'],
            'y_ankle': sample_data['y_ankle']
        }
        
        target_foot = target['y_foot'].unsqueeze(0).to(device)
        target_knee = target['y_knee'].unsqueeze(0).to(device)
        target_ankle = target['y_ankle'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_c = curve_encoder(target_foot, target_knee, target_ankle) # (1, latent_dim)
        
        base_graph = sample_data['base_data'] # extract expert's base graph
        
        # MCTS 环境
        env = MechanismEnv(
            surrogate,
            cfg.get('reward', {}),
            max_steps=1,
            device=device,
            constraint_cfg=cfg.get('constraints', {}),
        )
        env.reset(target, base_graph, z_c)
        mcts = MCTS(policy, surrogate, env, cfg, device)
        
        # 搜索最佳 J-Operator
        start_t = time.time()
        best_action, _ = mcts.search(base_graph, z_c, target)
        search_time = time.time() - start_t
        if best_action is None:
            print("[Inference] No valid action returned by MCTS; skipping sample.")
            continue
        print(f"MCTS Search finished in {search_time:.2f}s. Action: u={best_action['u']}, v={best_action['v']}, w={best_action['w']}")
        
        # 生成最终图
        gen_graph = apply_j_operator(
            base_graph, 
            best_action['u'], best_action['v'], best_action['w'], 
            best_action['n1'], best_action['n2']
        )
        
        # Surrogate 前向测试
        batch_eval = Batch.from_data_list([gen_graph]).to(device)
        with torch.no_grad():
            pred_foot, pred_knee, pred_ankle = surrogate(batch_eval)
        
        pred_foot = pred_foot.squeeze(0).cpu()
        pred_knee = pred_knee.squeeze(0).cpu()
        pred_ankle = pred_ankle.squeeze(0).cpu()
        
        # Plot
        plot_kinematics_result(
            target['y_foot'].numpy(), target['y_knee'].numpy(), target['y_ankle'].numpy(),
            pred_foot.numpy(), pred_knee.numpy(), pred_ankle.numpy(),
            i, args.output_dir
        )
        print(f"Saved visualization to {args.output_dir}/inference_result_sample_{i}.png")
        
if __name__ == '__main__':
    main()
