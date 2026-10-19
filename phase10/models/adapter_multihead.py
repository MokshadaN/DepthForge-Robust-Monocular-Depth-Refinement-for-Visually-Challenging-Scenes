"""
Phase 10: Multi-Head Dilated Residual Adapter Architecture with Soft Domain Routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, d: int, p_drop: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainHead(nn.Module):
    def __init__(self, in_ch: int = 96, p_drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadDilatedAdapter(nn.Module):
    """
    Phase 10 Multi-Head Dilated Adapter.
    Combines:
      - 5-channel multi-modal stem: [Base_Depth, R, G, B, Sobel_Energy]
      - Multi-scale dilated trunk (d = 1, 2, 4)
      - Soft domain classifier router (Camouflage, Glass, Textureless)
      - 3 domain-specialized expert residual heads
      - Tanh-gated residual scaling
    """
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
        trunk_ch = (hidden // 2) * 3   # 96

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

    def forward(self, base_d: torch.Tensor, rgb: torch.Tensor, energy: torch.Tensor):
        x = self.stem(torch.cat([base_d, rgb, energy], dim=1))
        trunk = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)  # (B, 96, H, W)
        logits = self.clf(trunk)  # (B, 3)
        alpha = torch.softmax(logits, dim=1)
        ac = alpha[:, 0].view(-1, 1, 1, 1)
        ag = alpha[:, 1].view(-1, 1, 1, 1)
        at = alpha[:, 2].view(-1, 1, 1, 1)
        delta = (ac * self.head_camo(trunk) +
                 ag * self.head_glass(trunk) +
                 at * self.head_tex(trunk))
        refined = torch.clamp(base_d + self.scale * torch.tanh(delta), 0.0, 1.0)
        return refined, logits
