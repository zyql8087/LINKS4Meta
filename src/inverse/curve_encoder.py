# src/inverse/curve_encoder.py
# 曲线编码器：将 3 条目标运动学曲线压缩为固定维度的条件向量 z_c

import torch
import torch.nn as nn


class CurveEncoder(nn.Module):
    """
    将 3 条运动学曲线展平拼接后，通过多层 MLP 编码为低维条件向量 z_c。
    输入: [y_foot (200,2), y_knee (200,), y_ankle (200,)]
    输出: z_c (latent_dim,)
    """

    def __init__(self, input_dim: int = 800, hidden_dims: list = None, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),   # LayerNorm: works with batch_size=1 (BatchNorm fails for RL single-sample inference)
                nn.ELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, y_foot: torch.Tensor, y_knee: torch.Tensor, y_ankle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_foot  : (B, 200, 2)  足端轨迹曲线
            y_knee  : (B, 200)     膝关节角度曲线
            y_ankle : (B, 200)     踝关节角度曲线
        Returns:
            z_c     : (B, latent_dim)  条件潜在向量
        """
        B = y_foot.size(0)
        foot_flat = y_foot.view(B, -1)          # (B, 400)
        x = torch.cat([foot_flat, y_knee, y_ankle], dim=-1)  # (B, 800)
        return self.net(x)
