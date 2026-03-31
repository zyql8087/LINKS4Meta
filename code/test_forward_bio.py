import argparse
import yaml
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
from src.data_load import DataLoaderFactory

def plot_kinematics(pred_foot, y_foot, pred_knee, y_knee, pred_ankle, y_ankle, sample_id, save_dir):
    """
    绘制预测曲线与真实曲线的对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Foot Trajectory
    axes[0].plot(y_foot[:, 0], y_foot[:, 1], label='Ground Truth', color='blue', linestyle='dashed', alpha=0.7)
    axes[0].plot(pred_foot[:, 0], pred_foot[:, 1], label='Prediction', color='red', alpha=0.7)
    axes[0].set_title(f'Foot Trajectory (Sample {sample_id})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].legend()
    axes[0].grid(True)
    
    # Knee Angle
    steps = range(len(y_knee))
    axes[1].plot(steps, y_knee, label='Ground Truth', color='blue', linestyle='dashed', alpha=0.7)
    axes[1].plot(steps, pred_knee, label='Prediction', color='red', alpha=0.7)
    axes[1].set_title(f'Knee Angle')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Angle (Norm)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Ankle Angle
    axes[2].plot(steps, y_ankle, label='Ground Truth', color='blue', linestyle='dashed', alpha=0.7)
    axes[2].plot(steps, pred_ankle, label='Prediction', color='red', alpha=0.7)
    axes[2].set_title(f'Ankle Angle')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Angle (Norm)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'test_sample_{sample_id}.png'))
    plt.close()

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_model', type=str, default=str(script_dir / 'src' / 'config_model_bio.yaml'))
    parser.add_argument('--config_dataset', type=str, default=str(script_dir / 'src' / 'config_dataset.yaml'))
    parser.add_argument('--model_path', type=str, default=str(script_dir / 'model_bio_best.pt'), help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default=str(script_dir / 'demo' / 'outputs' / 'test' / 'latest'),
                        help='Directory to save test visualizations')
    parser.add_argument('--num_vis', type=int, default=5, help='Number of test samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create directory for visualizations
    vis_dir = os.path.abspath(args.output_dir)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 加载配置
    with open(args.config_model, 'r', encoding='utf-8') as f:
        config_model = yaml.safe_load(f)
    with open(args.config_dataset, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    print(f"Running on device: {args.device}")

    # 2. 加载 Dataset (Test Loader)
    print("Loading test dataset...")
    factory = DataLoaderFactory(config_data)
    test_loader = factory.create_test_loader(
        batch_size=1, # 使用 batch_size = 1 方便逐个样本可视化
        shuffle=True, # 随机选取测试集样本
        num_workers=0
    )

    # 3. 初始化并加载训练好的模型
    model = BioKinematicsGNN(config_model).to(args.device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=args.device, weights_only=True))
        print(f"Successfully loaded weights from {args.model_path}")
    else:
        print(f"Warning: Model weights {args.model_path} not found! Testing with randomly initialized weights.")
    
    model.eval()

    # 4. 在测试集上评估
    total_samples = 0
    total_foot_loss = 0.0
    total_knee_loss = 0.0
    total_ankle_loss = 0.0

    print("\nStarting evaluation on test set...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(args.device)
            
            # 模型预测
            pred_foot, pred_knee, pred_ankle = model(data)
            
            # Ground truth
            y_foot = data.y_foot.view_as(pred_foot)
            y_knee = data.y_knee.view_as(pred_knee)
            y_ankle = data.y_ankle.view_as(pred_ankle)
            
            # 计算各项 MSE Loss
            foot_loss = torch.nn.functional.mse_loss(pred_foot, y_foot).item()
            knee_loss = torch.nn.functional.mse_loss(pred_knee, y_knee).item()
            ankle_loss = torch.nn.functional.mse_loss(pred_ankle, y_ankle).item()
            
            total_foot_loss += foot_loss
            total_knee_loss += knee_loss
            total_ankle_loss += ankle_loss
            total_samples += 1
            
            # 可视化前 N 个样本
            if i < args.num_vis:
                # 将张量转移回 CPU 并转换为 numpy 数组以进行绘图
                p_foot = pred_foot[0].cpu().numpy()
                g_foot = y_foot[0].cpu().numpy()
                p_knee = pred_knee[0].cpu().numpy()
                g_knee = y_knee[0].cpu().numpy()
                p_ankle = pred_ankle[0].cpu().numpy()
                g_ankle = y_ankle[0].cpu().numpy()
                
                # 有些模型可能会返回带有 sample_id 属性的 Data，没有就默认用索引
                sample_id = getattr(data, 'sample_id', [i])[0].item() if hasattr(data, 'sample_id') else i
                plot_kinematics(p_foot, g_foot, p_knee, g_knee, p_ankle, g_ankle, sample_id, vis_dir)
                print(f"Saved visualization for Sample {sample_id} to '{os.path.join(vis_dir, f'test_sample_{sample_id}.png')}'")

    # 5. 打印测试集上的整体性能表现
    avg_foot_loss = total_foot_loss / total_samples
    avg_knee_loss = total_knee_loss / total_samples
    avg_ankle_loss = total_ankle_loss / total_samples
    
    print("\n--- Testing Results Summary ---")
    print(f"Total Test Samples Evaluated: {total_samples}")
    print(f"Average Foot Trajectory MSE:  {avg_foot_loss:.6f}")
    print(f"Average Knee Angle MSE:       {avg_knee_loss:.6f}")
    print(f"Average Ankle Angle MSE:      {avg_ankle_loss:.6f}")
    print("-------------------------------")
    print(f"Test visualizations are saved in: {vis_dir}")

if __name__ == '__main__':
    main()
