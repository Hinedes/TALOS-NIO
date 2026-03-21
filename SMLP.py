"""
SMLP.py — Phase-Aware SpectralMLP (Rev. 3)
TALOS NIO Neural Backbone

Architecture:
    - Wrapper: Handles CPU-side FFT.
    - Phase-Aware Extraction: Separates Real and Imaginary components to preserve
        the sign of the DC bin (Gravity projection/Turn direction) and temporal phase.
    - Core: Expanded 396-input MLP backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMLPNPU(nn.Module):
    def __init__(self):
        super().__init__()
        # Expanded backbone to accept both Real and Imaginary components (6 channels * 66 = 396)
        self.fc1 = nn.Linear(396, 256)
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

        nn.init.normal_(self.head_cov.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.head_cov.bias, -2.0)

    def forward(self, x):
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
        fft_c  = torch.fft.rfft(x_raw, dim=-1)  # (B, 6, 33) complex

        # Isolate Real and Imaginary to preserve sign (direction) and phase (timing)
        real_part = fft_c.real
        imag_part = fft_c.imag

        # Symmetric Log-Compression: sign(x) * log1p(abs(x))
        # Preserves the polarity of the DC bins while compressing magnitude
        real_scaled = torch.sign(real_part) * torch.log1p(torch.abs(real_part))
        imag_scaled = torch.sign(imag_part) * torch.log1p(torch.abs(imag_part))

        # Concatenate and flatten -> (B, 6, 66) -> (B, 396)
        x_spec = torch.cat([real_scaled, imag_scaled], dim=-1).view(B, -1)
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