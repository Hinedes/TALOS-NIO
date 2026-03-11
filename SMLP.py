"""
SMLP.py — Overlord (ResNet1D)
TALOS NIO Neural Backbone — Rev. 2

Architecture: 1D Residual CNN on raw IMU time-domain data.
Replaces SpectralMLP (Rev. 1) which used FFT magnitude features.

Input:  (batch, 6, 64)  — raw mean-subtracted IMU window [accel(3) + gyro(3), 64 samples]
Output: translation (3), quaternion (4), log_var (3)

RKNN INT8 NPU constraints (RK3588):
  - Conv1d → reshaped to Conv2d (1×W) by RKNN compiler. Fully supported.
  - BatchNorm1d → folded into preceding conv weights at compile time. Zero runtime cost.
  - ReLU → native INT8 activation. GELU is NOT used.
  - Residual addition → native INT8 element-wise op. Fully supported.
  - No LSTM, no GRU, no attention, no LayerNorm, no Sigmoid/Tanh in hot path.

Channel sizes (32→64→128→256) are multiples of 32 for NPU MAC block alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """
    Standard residual block for 1D sequences.
    Two Conv1d layers with BN + ReLU. Skip connection with optional projection.
    All ops are RKNN INT8 native.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        # Projection shortcut — only when shape changes
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class BigSpectralMLP(nn.Module):
    """
    Overlord — ResNet1D backbone.
    Despite the class name kept for checkpoint compatibility, this is a
    purely convolutional residual network operating on raw IMU time-domain input.

    Encoder: 4 stages of increasing channel depth with stride-2 downsampling.
             Input (B, 6, 64) → bottleneck (B, 256, 4) → global avg pool → (B, 256)

    Heads:
      head_trans : Linear(256→3)  ego-centric XYZ displacement
      head_quat  : Linear(256→4)  → L2 normalize → unit quaternion
      head_cov   : Linear(256→3)  log-variance per axis (init: zeros → exp(0)=1.0)
    """

    def __init__(self, input_channels=6, base_ch=32):
        super().__init__()

        # Stem: project input channels into base feature space
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(),
        )

        # Residual stages — stride=2 halves sequence length each stage
        # (B, 6, 64) → (B, 32, 64) → (B, 64, 32) → (B, 128, 16) → (B, 256, 8) → (B, 256, 4)
        self.layer1 = ResBlock1D(base_ch,      base_ch * 2,  stride=2)   # 64→32
        self.layer2 = ResBlock1D(base_ch * 2,  base_ch * 4,  stride=2)   # 32→16
        self.layer3 = ResBlock1D(base_ch * 4,  base_ch * 8,  stride=2)   # 16→8
        self.layer4 = ResBlock1D(base_ch * 8,  base_ch * 8,  stride=2)   # 8→4

        feat_dim = base_ch * 8  # 256

        # Global average pooling collapses temporal dimension → (B, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Output heads
        self.head_trans = nn.Linear(feat_dim, 3)
        self.head_quat  = nn.Linear(feat_dim, 4)
        self.head_cov   = nn.Linear(feat_dim, 3)

        # Zero-init covariance head → exp(0) = 1.0 unit variance at round 1
        nn.init.zeros_(self.head_cov.weight)
        nn.init.zeros_(self.head_cov.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, 6, 64) — raw mean-subtracted IMU [accel(3)+gyro(3), T=64]

        Returns:
            translation : (batch, 3)  ego-centric displacement
            quaternion  : (batch, 4)  unit quaternion [W, X, Y, Z]
            log_var     : (batch, 3)  per-axis log-variance for NLL loss / R_obs
        """
        x = self.stem(x)       # (B, 32, 64)
        x = self.layer1(x)     # (B, 64, 32)
        x = self.layer2(x)     # (B, 128, 16)
        x = self.layer3(x)     # (B, 256, 8)
        x = self.layer4(x)     # (B, 256, 4)

        x = self.gap(x)        # (B, 256, 1)
        x = x.squeeze(-1)      # (B, 256)

        translation = self.head_trans(x)                              # (B, 3)
        quaternion  = F.normalize(self.head_quat(x), p=2, dim=-1)    # (B, 4)
        log_var     = self.head_cov(x)                                # (B, 3)

        return translation, quaternion, log_var


# Smoke test
if __name__ == '__main__':
    model = BigSpectralMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    dummy = torch.randn(4, 6, 64)
    t, q, lv = model(dummy)
    print(f"translation : {t.shape}  {t.dtype}")
    print(f"quaternion  : {q.shape}  norm={q.norm(dim=-1).mean():.6f}")
    print(f"log_var     : {lv.shape}  init_mean={lv.mean():.6f}  (should be ~0)")
