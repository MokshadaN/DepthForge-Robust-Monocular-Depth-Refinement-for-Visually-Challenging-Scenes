"""
Phase 8 — Metrics.
Replaces Phase 7's BSI (which rewarded noise and hallucinated edges).

Standard monocular depth evaluation metrics (after median scale alignment):
  AbsRel   — absolute relative error       (lower is better)
  SqRel    — squared relative error        (lower is better)
  RMSE     — root mean squared error       (lower is better)
  SILog    — scale-invariant log error     (lower is better)
  delta1   — % pixels with ratio < 1.25   (higher is better)
  delta2   — % pixels with ratio < 1.25²  (higher is better)
  delta3   — % pixels with ratio < 1.25³  (higher is better)
"""
from __future__ import annotations
import torch
import numpy as np


def _align_scale(pred: torch.Tensor, gt: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """Median scale alignment: multiply pred by median(gt)/median(pred)."""
    scale = torch.median(gt) / (torch.median(pred) + eps)
    return pred * scale


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor,
                    align_scale: bool = True,
                    eps: float = 1e-6) -> dict[str, float]:
    """
    Compute standard depth estimation metrics.

    Args:
        pred: predicted depth tensor (any shape), positive floats
        gt:   ground-truth / pseudo-GT depth tensor (same shape)
        align_scale: if True, median-align pred to gt before computing
                     AbsRel/RMSE/δ metrics (fair when using pseudo-GT)
    Returns:
        dict with AbsRel, SqRel, RMSE, SILog, δ₁, δ₂, δ₃
    """
    pred = pred.float().flatten()
    gt   = gt.float().flatten()

    mask = (gt > eps) & (pred > eps)
    if mask.sum() < 5:
        return {k: float("nan") for k in
                ["AbsRel", "SqRel", "RMSE", "SILog", "d1", "d2", "d3"]}

    p, g = pred[mask], gt[mask]

    if align_scale:
        p = _align_scale(p, g, eps)

    # -- Error metrics --
    abs_rel = ((p - g).abs() / (g + eps)).mean().item()
    sq_rel  = (((p - g) ** 2) / (g + eps)).mean().item()
    rmse    = torch.sqrt(((p - g) ** 2).mean()).item()

    # SILog (no scale alignment needed — it is inherently scale-invariant)
    dp = pred[mask]   # un-aligned for SILog
    dg = gt[mask]
    d  = torch.log(dp + eps) - torch.log(dg + eps)
    silog = torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2) + eps).item()

    # -- Threshold accuracy --
    ratio  = torch.max(p / (g + eps), g / (p + eps))
    delta1 = (ratio < 1.25      ).float().mean().item()
    delta2 = (ratio < 1.25 ** 2 ).float().mean().item()
    delta3 = (ratio < 1.25 ** 3 ).float().mean().item()

    return {
        "AbsRel": abs_rel,
        "SqRel":  sq_rel,
        "RMSE":   rmse,
        "SILog":  silog,
        "d1":     delta1,
        "d2":     delta2,
        "d3":     delta3,
    }


def format_metrics(m: dict[str, float], prefix: str = "") -> str:
    """Return a single-line human-readable metrics string."""
    return (
        f"{prefix}AbsRel={m['AbsRel']:.4f}  SqRel={m['SqRel']:.4f}  "
        f"RMSE={m['RMSE']:.4f}  SILog={m['SILog']:.4f}  "
        f"δ₁={m['d1']:.4f}  δ₂={m['d2']:.4f}  δ₃={m['d3']:.4f}"
    )


def aggregate_metrics(metric_list: list[dict]) -> dict[str, float]:
    """Average a list of per-sample metric dicts (skip NaN entries)."""
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
