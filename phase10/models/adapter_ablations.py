"""
Phase 10: Model variants for targeted ablation studies.
Isolates:
  1. Sobel Gradient Energy Prior (5-channel vs 4-channel input)
  2. Soft Domain Routing (Dynamic softmax weights vs Fixed uniform 1/3 weights)
  3. Expert Head Specialization (3 Specialized Heads vs 1 Monolithic Head)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase10.models.adapter_multihead import DilatedBranch, DomainHead

class NoSobelAdapter(nn.Module):
    """Ablation 1: Multi-head adapter WITHOUT Sobel energy prior (in_ch = 4)."""
    def __init__(self, in_ch: int = 4, hidden: int = 64, p_drop: float = 0.10, scale: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop)
        )
        self.b1 = DilatedBranch(hidden, hidden // 2, 1, p_drop)
        self.b2 = DilatedBranch(hidden, hidden // 2, 2, p_drop)
        self.b4 = DilatedBranch(hidden, hidden // 2, 4, p_drop)
        trunk_ch = (hidden // 2) * 3

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop * 2),
            nn.Linear(trunk_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

        self.head_camo  = DomainHead(trunk_ch, p_drop)
        self.head_glass = DomainHead(trunk_ch, p_drop)
        self.head_tex   = DomainHead(trunk_ch, p_drop)
        self.scale = scale

    def forward(self, base_d: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor = None):
        x = self.stem(torch.cat([base_d, rgb], dim=1))
        trunk = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)
        logits = self.clf(trunk)
        alpha = torch.softmax(logits, dim=1)
        ac = alpha[:, 0].view(-1, 1, 1, 1)
        ag = alpha[:, 1].view(-1, 1, 1, 1)
        at = alpha[:, 2].view(-1, 1, 1, 1)
        delta = (ac * self.head_camo(trunk) +
                 ag * self.head_glass(trunk) +
                 at * self.head_tex(trunk))
        refined = torch.clamp(base_d + self.scale * torch.tanh(delta), 0.0, 1.0)
        return refined, logits


class UniformRoutingAdapter(nn.Module):
    """Ablation 2: Multi-head adapter WITHOUT dynamic router (fixed uniform weights 1/3 each)."""
    def __init__(self, in_ch: int = 5, hidden: int = 64, p_drop: float = 0.10, scale: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop)
        )
        self.b1 = DilatedBranch(hidden, hidden // 2, 1, p_drop)
        self.b2 = DilatedBranch(hidden, hidden // 2, 2, p_drop)
        self.b4 = DilatedBranch(hidden, hidden // 2, 4, p_drop)
        trunk_ch = (hidden // 2) * 3

        self.head_camo  = DomainHead(trunk_ch, p_drop)
        self.head_glass = DomainHead(trunk_ch, p_drop)
        self.head_tex   = DomainHead(trunk_ch, p_drop)
        self.scale = scale

    def forward(self, base_d: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor):
        x = self.stem(torch.cat([base_d, rgb, energy], dim=1))
        trunk = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)
        # Uniform fixed blend
        delta = (self.head_camo(trunk) + self.head_glass(trunk) + self.head_tex(trunk)) / 3.0
        refined = torch.clamp(base_d + self.scale * torch.tanh(delta), 0.0, 1.0)
        dummy_logits = torch.zeros((base_d.shape[0], 3), device=base_d.device)
        return refined, dummy_logits


class SingleHeadDilatedAdapter(nn.Module):
    """Ablation 3: Monolithic single-head adapter (Phase 8 architecture, ~83.4K params)."""
    def __init__(self, in_ch: int = 5, hidden: int = 64, p_drop: float = 0.10, scale: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop)
        )
        self.b1 = DilatedBranch(hidden, hidden // 2, 1, p_drop)
        self.b2 = DilatedBranch(hidden, hidden // 2, 2, p_drop)
        self.b4 = DilatedBranch(hidden, hidden // 2, 4, p_drop)
        trunk_ch = (hidden // 2) * 3
        self.head = nn.Sequential(
            nn.Conv2d(trunk_ch, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(48, 1, 1)
        )
        self.scale = scale

    def forward(self, base_d: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor):
        x = self.stem(torch.cat([base_d, rgb, energy], dim=1))
        trunk = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)
        delta = self.head(trunk)
        refined = torch.clamp(base_d + self.scale * torch.tanh(delta), 0.0, 1.0)
        dummy_logits = torch.zeros((base_d.shape[0], 3), device=base_d.device)
        return refined, dummy_logits
