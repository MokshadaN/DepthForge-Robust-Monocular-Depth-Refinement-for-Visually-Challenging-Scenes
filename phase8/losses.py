"""
Phase 8 — Losses.
Fixed loss functions that replace Phase 7's contradicting L1+edge combo.

SILog:       Scale-Invariant Log loss (standard for monocular depth).
             Training against pseudo-GT is valid because SILog is scale-invariant:
             multiplying GT by any constant does not change the gradient direction.
GradLoss:    Multi-scale gradient consistency in log-depth space.
             Penalises wrong depth gradient structure without fighting SILog.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def silog_loss(pred: torch.Tensor, target: torch.Tensor,
               lamda: float = 0.85, eps: float = 1e-6) -> torch.Tensor:
    """
    Scale-Invariant Logarithmic loss (Eigen et al., NeurIPS 2014).

    Args:
        pred:   predicted depth  (B,1,H,W) or (H,W), positive floats
        target: pseudo/GT depth  same shape, positive floats
        lamda:  variance focus weight (0.85 is standard)
    """
    mask = (target > eps) & (pred > eps)
    if mask.sum() < 10:
        return pred.sum() * 0.0   # no valid pixels

    d = torch.log(pred[mask] + eps) - torch.log(target[mask] + eps)
    return torch.sqrt((d ** 2).mean() - lamda * (d.mean() ** 2) + eps)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  scales: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-scale log-gradient consistency loss.
    Encourages the adapter to match the *structure* of the teacher's depth,
    not just scale, across multiple resolutions.

    Args:
        pred, target: (B,1,H,W) positive depth tensors
        scales:       number of downsampling levels (1×, 2×, 4×)
    """
    total = pred.new_zeros(1)
    for s in range(scales):
        if s > 0:
            p = F.avg_pool2d(pred,   2 ** s, 2 ** s)
            t = F.avg_pool2d(target, 2 ** s, 2 ** s)
        else:
            p, t = pred, target

        # Work in log-depth space to match SILog
        lp = torch.log(p.clamp(min=eps))
        lt = torch.log(t.clamp(min=eps))

        gx_p = lp[:, :, :, 1:] - lp[:, :, :, :-1]
        gx_t = lt[:, :, :, 1:] - lt[:, :, :, :-1]
        gy_p = lp[:, :, 1:, :] - lp[:, :, :-1, :]
        gy_t = lt[:, :, 1:, :] - lt[:, :, :-1, :]

        total = total + (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()

    return total / scales


def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  w_silog: float = 1.0, w_grad: float = 0.5) -> torch.Tensor:
    """Weighted sum of SILog + gradient consistency (no contradictions)."""
    return w_silog * silog_loss(pred, target) + w_grad * gradient_loss(pred, target)
