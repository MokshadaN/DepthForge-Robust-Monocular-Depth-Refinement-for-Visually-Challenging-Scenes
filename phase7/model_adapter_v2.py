"""
Phase 7: Multi-Scale Dilated Residual Adapter for UniDAC.
Captures both wide-area planar smoothness and razor-thin boundary contours.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDilatedBlock(nn.Module):
    """
    Atrous / Dilated Residual Convolution Block with 3 parallel receptive fields.
    Dilation 1: local pixel boundary transitions
    Dilation 2: intermediate object silhouettes
    Dilation 4: broad contextual / surface support
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_dim = out_channels // 3
        rem = out_channels - (branch_dim * 3)

        self.conv_d1 = nn.Conv2d(in_channels, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(in_channels, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv2d(in_channels, branch_dim + rem, kernel_size=3, padding=4, dilation=4)

        self.gn = nn.GroupNorm(4, out_channels)
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.conv_d1(x)
        b2 = self.conv_d2(x)
        b4 = self.conv_d4(x)
        out = torch.cat([b1, b2, b4], dim=1)
        out = F.gelu(self.gn(out))
        return self.proj(out)


class MultiScaleDilatedAdapter(nn.Module):
    """
    Deep Multi-Scale Residual Adapter for Camouflaged, Textureless, and Transparent Depth.
    """
    def __init__(self, in_channels: int = 5, hidden_dim: int = 36):
        super().__init__()
        # Input: 1 (base depth) + 3 (RGB) + 1 (Sobel energy) = 5
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.block1 = MultiScaleDilatedBlock(hidden_dim, hidden_dim)
        self.block2 = MultiScaleDilatedBlock(hidden_dim, hidden_dim)
        self.block3 = MultiScaleDilatedBlock(hidden_dim, hidden_dim)
        
        self.out_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        
        # Zero-initialization for stable identity starting point
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, base_depth: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([base_depth, rgb, energy], dim=1)
        feat = F.gelu(self.in_conv(x))
        feat = self.block1(feat) + feat
        feat = self.block2(feat) + feat
        feat = self.block3(feat) + feat
        
        delta = self.out_conv(feat)
        # Refined depth = Base Depth + Bounded Learned Correction
        refined_depth = F.relu(base_depth + 0.30 * torch.tanh(delta))
        return refined_depth
