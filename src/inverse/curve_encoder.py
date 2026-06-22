# 曲线编码器：将足端轨迹、膝关节角度、踝关节角度三条曲线压缩为固定维度条件向量 z_c。
# 使用 LayerNorm（兼容 RL 场景 batch_size=1）、ELU 激活、Dropout 正则化。

import torch
import torch.nn as nn


class CurveEncoder(nn.Module):
    """将 3 条运动学曲线展平拼接后通过 MLP 编码为低维条件向量 z_c。
    输入: y_foot(B,200,2) + y_knee(B,200) + y_ankle(B,200) -> 展平拼接(B,800) -> z_c(B,latent_dim)
    """

    def __init__(self, input_dim: int = 800, hidden_dims: list = None, latent_dim: int = 128, dropout: float = 0.1):
        """初始化编码器。hidden_dims 默认 [512, 256]，latent_dim 输出维度，dropout 正则化概率。"""
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(p=dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y_foot: torch.Tensor, y_knee: torch.Tensor, y_ankle: torch.Tensor) -> torch.Tensor:
        """前向传播：展平 y_foot 并与 y_knee、y_ankle 拼接后通过 MLP 编码。"""
        B = y_foot.size(0)
        foot_flat = y_foot.view(B, -1)
        x = torch.cat([foot_flat, y_knee, y_ankle], dim=-1)
        return self.net(x)
