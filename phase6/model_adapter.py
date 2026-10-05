"""
Phase 6: Texture & Camouflage Aware Adapter for UniDAC.
Adds a lightweight residual edge-and-plane refinement head on top of UniDAC depth features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class TextureAwareResidualHead(nn.Module):
    """
    Lightweight Residual Boundary & Planar Refinement Head.
    Input: Depth Map + Normalized Ray Coordinates + RGB Features.
    Output: Residual Depth Correction (delta_D) enforcing sharp boundaries and flat planes.
    """
    def __init__(self, in_channels: int = 5, hidden_dim: int = 32):
        super().__init__()
        # in_channels: 1 (raw depth) + 3 (RGB) + 1 (gradient energy) = 5
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(4, hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(4, hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gn3   = nn.GroupNorm(4, hidden_dim)
        
        # Final residual prediction (delta depth)
        self.out_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        
        # Initialize out_conv to near-zero so initial output is identical to baseline UniDAC
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, depth: torch.Tensor, rgb: torch.Tensor, grad_energy: torch.Tensor) -> torch.Tensor:
        # depth: (B, 1, H, W), rgb: (B, 3, H, W), grad_energy: (B, 1, H, W)
        x = torch.cat([depth, rgb, grad_energy], dim=1)
        feat = F.gelu(self.gn1(self.conv1(x)))
        feat = F.gelu(self.gn2(self.conv2(feat))) + feat
        feat = F.gelu(self.gn3(self.conv3(feat))) + feat
        delta = self.out_conv(feat)
        
        # Refined depth = Base Depth + Learned Delta (bounded by tanh)
        refined_depth = F.relu(depth + 0.35 * torch.tanh(delta))
        return refined_depth


class CamouflageFineTunedUniDAC(nn.Module):
    """
    End-to-End Fine-Tuned Model combining Pretrained UniDAC with the Texture-Aware Head.
    """
    def __init__(self):
        super().__init__()
        self.adapter = TextureAwareResidualHead()

    def forward(self, base_depth: torch.Tensor, rgb: torch.Tensor, grad_energy: torch.Tensor) -> torch.Tensor:
        return self.adapter(base_depth, rgb, grad_energy)
