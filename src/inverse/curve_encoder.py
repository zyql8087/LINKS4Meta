# 曲线编码器：将足端轨迹、膝关节角度、踝关节角度三条曲线编码为固定维度条件向量 z_c。
#
# 设计要点（相对旧版 flatten+MLP 的改进）：
#   1. 不再把 (B,200,2)+(B,200)+(B,200) 展平成 (B,800) 再过 MLP，而是组织成
#      4 通道时序张量 (B, C=4, T)，用 1D-CNN 提取时序局部特征，保留时间结构与相位关系。
#   2. 逐通道实例归一化（沿时间维做 mean/std 标准化 + 可学习仿射），消除足端坐标与
#      关节角度之间的量纲差异，且与 batch_size 无关（兼容 RL 场景 batch_size=1）。
#   3. 循环 padding（padding_mode='circular'）契合步态曲线的周期性。
#   4. 全局平均池化 + 全局最大池化拼接后过 MLP，输出维度 T 无关，可处理任意时间步数。
#
# 构造签名与输出维度与旧版保持一致：CurveEncoder(input_dim, hidden_dims, latent_dim, dropout)
# input_dim 仅作为容量/兼容性参数保留（实际形状在 forward 时从输入推断）。

import torch
import torch.nn as nn


class CurveEncoder(nn.Module):
    """将 3 条运动学曲线编码为低维条件向量 z_c（序列感知版本）。

    输入: y_foot(B,T,2) + y_knee(B,T) + y_ankle(B,T)
       -> 组织为 (B, 4, T) -> 逐通道实例归一化 -> 1D-CNN -> 全局池化 -> z_c(B,latent_dim)
    """

    NUM_CHANNELS = 4  # foot_x, foot_y, knee, ankle

    def __init__(self, input_dim: int = 800, hidden_dims: list = None, latent_dim: int = 128, dropout: float = 0.1):
        """初始化编码器。

        hidden_dims: 1D 卷积各层的通道数，默认 [512, 256]；
        latent_dim: 输出条件向量维度；
        dropout: 正则化概率；
        input_dim: 兼容旧接口的容量参数，不参与形状计算。
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.eps = 1.0e-5

        # 逐通道可学习仿射（实例归一化之后使用）
        self.channel_scale = nn.Parameter(torch.ones(self.NUM_CHANNELS, 1))
        self.channel_bias = nn.Parameter(torch.zeros(self.NUM_CHANNELS, 1))

        # 1D-CNN 主干：循环 padding + ELU + Dropout
        conv_layers = []
        in_ch = self.NUM_CHANNELS
        for out_ch in hidden_dims:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode="circular"),
                nn.ELU(),
                nn.Dropout(p=dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.feature_dim = in_ch

        # 全局平均池化 + 全局最大池化拼接 -> latent
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, self.latent_dim),
        )

    def _stack_channels(self, y_foot: torch.Tensor, y_knee: torch.Tensor, y_ankle: torch.Tensor) -> torch.Tensor:
        """将三条曲线组织为 (B, 4, T) 时序张量。"""
        if y_foot.dim() == 2:  # (B, 2T) 形式的旧式展平输入，无法恢复时序，退化为 (B,1,2T)
            y_foot = y_foot.unsqueeze(1)
        # y_foot: (B, T, 2) -> (B, 2, T)
        foot = y_foot.transpose(1, 2)
        knee = y_knee.unsqueeze(1) if y_knee.dim() == 2 else y_knee  # (B,1,T)
        ankle = y_ankle.unsqueeze(1) if y_ankle.dim() == 2 else y_ankle
        return torch.cat([foot, knee, ankle], dim=1)  # (B, 4, T)

    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """沿时间维做逐通道标准化 + 可学习仿射，batch-size 无关。"""
        if x.size(-1) <= 1:
            return x * self.channel_scale + self.channel_bias
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        x = (x - mean) / std
        return x * self.channel_scale + self.channel_bias

    def forward(self, y_foot: torch.Tensor, y_knee: torch.Tensor, y_ankle: torch.Tensor) -> torch.Tensor:
        """前向传播：组织通道 -> 实例归一化 -> 1D-CNN -> 全局池化 -> 条件向量。"""
        x = self._stack_channels(y_foot, y_knee, y_ankle)  # (B, 4, T)
        x = self._instance_normalize(x)
        feat = self.conv(x)  # (B, feature_dim, T)
        avg_pool = feat.mean(dim=-1)  # (B, feature_dim)
        max_pool = feat.max(dim=-1).values  # (B, feature_dim)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2*feature_dim)
        return self.head(pooled)
