"""
Phase 8: Multi-Scale Dilated Residual Depth Refinement Adapter.
Input : [base_depth (1ch) | RGB (3ch) | Sobel_energy (1ch)] -> 5 channels
Output: Refined depth (1ch) = base_depth + 0.30 * tanh(Delta)
"""
import torch
import torch.nn as nn

class MultiScaleDilatedAdapter(nn.Module):
    def __init__(self, in_channels: int = 5, hidden: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
        )
        fuse_in = (hidden // 2) * 3
        self.fusion = nn.Sequential(
            nn.Conv2d(fuse_in, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.scale = 0.30

    def forward(self, base_depth: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([base_depth, rgb, energy], dim=1)
        x = self.stem(feat)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b4 = self.b4(x)
        fused = torch.cat([b1, b2, b4], dim=1)
        delta = self.scale * torch.tanh(self.fusion(fused))
        return torch.clamp(base_depth + delta, 0.0, 1.0)
