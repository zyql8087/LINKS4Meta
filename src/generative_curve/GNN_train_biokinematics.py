# src/generative_curve/train_biokinematics.py
# Training logic with multi-objective loss for BioKinematics model

import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_loss(pred_foot, pred_knee, pred_ankle, data, config):
    """
    Multi-objective Loss Function
    Args:
        pred_foot: (B, curve_steps, 2) - predicted foot trajectory (x, y)
        pred_knee: (B, curve_steps) - predicted knee angle curve
        pred_ankle: (B, curve_steps) - predicted ankle angle curve
        data: PyG batch data containing y_foot, y_knee, y_ankle
        config: training config with loss weights
    """
    # Get weights from config
    w_foot = config.get('w_foot', 1.0)
    w_knee = config.get('w_knee', 0.5)
    w_ankle = config.get('w_ankle', 0.5)
    
    # Foot trajectory loss (MSE on 2D coordinates)
    loss_foot = F.mse_loss(pred_foot, data.y_foot.view_as(pred_foot))
    
    # Knee angle loss
    loss_knee = F.mse_loss(pred_knee, data.y_knee.view_as(pred_knee))
    
    # Ankle angle loss
    loss_ankle = F.mse_loss(pred_ankle, data.y_ankle.view_as(pred_ankle))
    
    # Weighted total loss
    total_loss = w_foot * loss_foot + w_knee * loss_knee + w_ankle * loss_ankle
    
    return total_loss, loss_foot, loss_knee, loss_ankle


def train_epoch(model, loader, optimizer, config, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred_foot, pred_knee, pred_ankle = model(data)
        
        # Compute loss
        loss, _, _, _ = compute_loss(pred_foot, pred_knee, pred_ankle, data, config)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def eval_epoch(model, loader, config, device):
    """
    Evaluate for one epoch
    Returns:
        avg_loss: average total loss
        avg_foot_err: average foot trajectory error (for monitoring)
    """
    model.eval()
    total_loss = 0.0
    total_foot_err = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", leave=False):
            data = data.to(device)
            
            # Forward pass
            pred_foot, pred_knee, pred_ankle = model(data)
            
            # Compute loss
            loss, loss_foot, _, _ = compute_loss(pred_foot, pred_knee, pred_ankle, data, config)
            
            total_loss += loss.item()
            total_foot_err += loss_foot.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_foot_err = total_foot_err / max(num_batches, 1)
    
    return avg_loss, avg_foot_err
