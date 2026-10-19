"""
Phase 10: Rigorous Evaluation Metrics Engine.

Addresses Step 1:
- Separates Raw (Unaligned), Global Scale-Aligned, and Affine-Fitted alignment.
- Validates strict positive-depth masks with configurable [min_depth, max_depth] sensor bounds.
- Computes standard depth metrics: AbsRel, SqRel, RMSE, RMSElog, SILog, δ1, δ2, δ3.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any, Optional

def compute_raw_metrics(pred: np.ndarray,
                        gt: np.ndarray,
                        min_depth: float = 1e-3,
                        max_depth: float = np.inf,
                        eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute raw (unaligned) metrics directly between prediction and ground truth.
    """
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt >= min_depth) & (gt <= max_depth) & (pred > 0)
    if valid.sum() < 10:
        return {k: float("nan") for k in ["AbsRel", "SqRel", "RMSE", "RMSElog", "SILog", "d1", "d2", "d3"]}

    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)

    # AbsRel & SqRel
    abs_rel = np.mean(np.abs(p - g) / (g + eps))
    sq_rel = np.mean(((p - g) ** 2) / (g + eps))
    rmse = np.sqrt(np.mean((p - g) ** 2))

    # Log metrics
    p_safe = np.clip(p, eps, None)
    g_safe = np.clip(g, eps, None)
    log_p = np.log(p_safe)
    log_g = np.log(g_safe)
    rmse_log = np.sqrt(np.mean((log_p - log_g) ** 2))

    # SILog (Scale Invariant Logarithmic Error)
    diff_log = log_p - log_g
    silog = np.sqrt(max(0.0, np.mean(diff_log ** 2) - 0.85 * (np.mean(diff_log) ** 2)))

    # Threshold accuracy
    ratio = np.maximum(p_safe / g_safe, g_safe / p_safe)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "RMSElog": float(rmse_log),
        "SILog": float(silog),
        "d1": d1,
        "d2": d2,
        "d3": d3,
    }


def align_median_scale(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Global median scale alignment: pred * (median(gt) / median(pred)).
    Standard for monocular depth with unknown physical scale.
    """
    p_val = pred[valid_mask]
    g_val = gt[valid_mask]
    if len(p_val) < 10 or np.median(p_val) < eps:
        return pred
    scale = np.median(g_val) / (np.median(p_val) + eps)
    return pred * scale


def align_affine_least_squares(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Per-image least-squares affine fit: s * pred + t.
    NOTE: In Step 1 audit, this is target-fitted alignment and must NOT be used as primary score.
    """
    p_val = pred[valid_mask]
    g_val = gt[valid_mask]
    if len(p_val) < 10:
        return pred
    A = np.stack([p_val, np.ones_like(p_val)], axis=1)
    s, t = np.linalg.lstsq(A, g_val, rcond=None)[0]
    return np.clip(s * pred + t, 0.0, 1.0)


def evaluate_sample_dual(pred: np.ndarray,
                         gt: np.ndarray,
                         min_depth: float = 1e-3,
                         max_depth: float = np.inf) -> Dict[str, Dict[str, float]]:
    """
    Compute dual evaluation:
    1. 'raw': unaligned raw output
    2. 'median_aligned': median-scale aligned
    3. 'affine_aligned': target-fitted per-image least-squares
    """
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt >= min_depth) & (gt <= max_depth) & (pred > 0)

    raw_metrics = compute_raw_metrics(pred, gt, min_depth, max_depth)

    pred_med = align_median_scale(pred.copy(), gt, valid)
    med_metrics = compute_raw_metrics(pred_med, gt, min_depth, max_depth)

    pred_aff = align_affine_least_squares(pred.copy(), gt, valid)
    aff_metrics = compute_raw_metrics(pred_aff, gt, min_depth, max_depth)

    return {
        "raw": raw_metrics,
        "median_aligned": med_metrics,
        "affine_aligned": aff_metrics,
    }


def aggregate_metric_dicts(metric_list: list[Dict[str, float]]) -> Dict[str, float]:
    """Average metric dictionaries across a dataset, ignoring NaNs."""
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
