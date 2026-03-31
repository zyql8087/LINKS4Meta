# train_forward_bio.py

import argparse
import yaml
import torch
import os
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.generative_curve.GNN_model_biokinematics import BioKinematicsGNN
from src.generative_curve.GNN_train_biokinematics import train_epoch, eval_epoch
from src.data_load import DataLoaderFactory 
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_model', type=str, default='src/config_model_bio.yaml')
    parser.add_argument('--config_dataset', type=str, default='src/config_dataset.yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config_model, 'r', encoding='utf-8') as f:
        config_model = yaml.safe_load(f)
    with open(args.config_dataset, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    print(f"Running on device: {args.device}")

    # 2. 加载 Dataset 80k
    print("Loading Dataset 80k...")
    factory = DataLoaderFactory(config_data)
    train_loader_orig = factory.create_train_loader(
        batch_size=config_model['training']['batch_size'],
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    val_loader_orig = factory.create_val_loader(
        batch_size=config_model['training']['batch_size'],
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    print("\nPre-caching batched graphs to Device memory to bypass PyG collation bottleneck...")
    from tqdm import tqdm
    import random
    
    class PreCachedLoader:
        def __init__(self, dataloader, device, desc, shuffle):
            self.batches = []
            self.shuffle = shuffle
            for batch in tqdm(dataloader, desc=desc):
                # 提前将合并后的巨图转移至 GPU
                self.batches.append(batch.to(device))
                
        def __iter__(self):
            if self.shuffle:
                random.shuffle(self.batches)
            return iter(self.batches)
            
        def __len__(self):
            return len(self.batches)
            
    train_loader = PreCachedLoader(train_loader_orig, args.device, desc="Caching Train", shuffle=True)
    val_loader = PreCachedLoader(val_loader_orig, args.device, desc="Caching Val", shuffle=False)

    # 3. 初始化 BioKinematics 模型
    model = BioKinematicsGNN(config_model).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_model['training']['learning_rate'])
    
    print(f"Model Initialized. Params: {sum(p.numel() for p in model.parameters())}")

    # 记录 Loss 用于可视化
    train_losses = []
    val_losses = []

    # 4. 训练循环
    best_val_loss = float('inf')
    for epoch in range(config_model['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, config_model['training'], args.device)
        val_loss, val_foot_err = eval_epoch(model, val_loader, config_model['training'], args.device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Foot Err: {val_foot_err:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_bio_best.pt')
            print("  --> Model Saved!")

    # 5. 可视化 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    output_dir = Path(__file__).resolve().parent / 'demo' / 'outputs' / 'training'
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = output_dir / 'loss_curve.png'
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to '{loss_curve_path}'.")
    # plt.show() # Uncomment to show interactively, but usually best to just save when training remotely

if __name__ == '__main__':
    main()
