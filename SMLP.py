"""
SMLP.py — SpectralMLP (Rev. 1 Restored)
TALOS NIO Neural Backbone
Architecture:
  - Wrapper: Handles CPU-side FFT for seamless training from (B, 6, 64) raw IMU.
  - Core: The INT8-compatible MLP backbone running on 198 spectral bins.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralMLPNPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(198, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.head_trans = nn.Linear(64, 3)
        self.head_quat  = nn.Linear(64, 4)
        self.head_cov   = nn.Linear(64, 3)
        nn.init.zeros_(self.head_cov.weight)
        nn.init.zeros_(self.head_cov.bias)

    def forward(self, x):
        x = self.drop(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.head_trans(x), F.normalize(self.head_quat(x), p=2, dim=-1), self.head_cov(x)

class SpectralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.npu_core = SpectralMLPNPU()

    def forward(self, x_raw):
        B = x_raw.size(0)
        fft_c = torch.fft.rfft(x_raw, dim=-1)
        x_spec = torch.log1p(torch.abs(fft_c)).view(B, -1)
        return self.npu_core(x_spec)

BigSpectralMLP = SpectralMLP

if __name__ == '__main__':
    model = SpectralMLP()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    dummy = torch.randn(4, 6, 64)
    t, q, lv = model(dummy)
    print(f"Translation: {t.shape}, Quat: {q.shape}, LogVar: {lv.shape}")
    print(f"Quat norm: {q.norm(dim=-1).mean():.6f}, LogVar init: {lv.mean():.6f}")
