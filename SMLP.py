"""
SMLP.py — SpectralMLP (Rev. 2)
TALOS NIO Neural Backbone

Architecture:
  - Wrapper: Handles CPU-side FFT for seamless training from (B, 6, 64) raw IMU.
  - Core: The INT8-compatible MLP backbone running on 198 spectral bins.

Covariance head outputs raw log-variance (log σ²).
Caller applies exp() to get σ². Convention unchanged from Rev. 1.

Changes from Rev. 1:
  - head_cov weight: zeros_ → normal_(0, 0.01)
      Restores gradient flow from covariance loss into the shared backbone.
      zeros_ prevented the covariance branch from influencing trunk updates.
  - head_cov bias: zeros_ → constant_(-2.0)
      log σ² = -2.0 → σ² ≈ 0.135 at init, matching observed actual error ~0.15.
      Previously init at 0.0 → σ² = 1.0, far above actual error, inflating R from
      the first forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMLPNPU(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone
        self.fc1 = nn.Linear(198, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # Dropout on hidden embedding -- NOT on raw spectral input
        self.drop = nn.Dropout(0.15)

        # Translation branch (kinematics)
        self.fc_trans   = nn.Linear(128, 64)
        self.bn_trans   = nn.BatchNorm1d(64)
        self.head_trans = nn.Linear(64, 3)

        # Covariance branch (uncertainty)
        self.fc_cov   = nn.Linear(128, 64)
        self.bn_cov   = nn.BatchNorm1d(64)
        self.head_cov = nn.Linear(64, 3)

        # Calibrated covariance init -- do not change
        nn.init.normal_(self.head_cov.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_cov.bias, -2.0)

    def forward(self, x):
        # Shared feature extraction -- no dropout on raw 198-bin input
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))

        # Translation path
        t = F.relu(self.bn_trans(self.fc_trans(x)))
        pred_vel = self.head_trans(t)

        # Covariance path
        c = F.relu(self.bn_cov(self.fc_cov(x)))
        pred_cov = self.head_cov(c)

        return pred_vel, pred_cov


class SpectralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.npu_core = SpectralMLPNPU()

    def forward(self, x_raw):
        B = x_raw.size(0)
        fft_c  = torch.fft.rfft(x_raw, dim=-1)
        x_spec = torch.log1p(torch.abs(fft_c)).view(B, -1)
        return self.npu_core(x_spec)


BigSpectralMLP = SpectralMLP


if __name__ == '__main__':
    model = SpectralMLP()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(4, 6, 64)
    t, lv = model(dummy)
    print(f"Translation : {t.shape}")
    print(f"LogVar      : {lv.shape}")
    print(f"LogVar init mean : {lv.mean():.4f}  (target: ~-2.0)")
    print(f"Sigma² init mean : {lv.exp().mean():.4f}  (target: ~0.135)")