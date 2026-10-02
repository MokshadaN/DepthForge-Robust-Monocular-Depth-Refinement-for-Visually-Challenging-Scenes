"""Mask-wise depth metrics with both global and per-region scale alignment, depth binning, and bootstrap CIs."""

import numpy as np


def _stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    if pred.size == 0 or gt.size == 0:
        return dict(absrel=np.nan, rmse=np.nan, d1=np.nan, d2=np.nan, d3=np.nan, n=0)
    r = np.maximum(pred / gt, gt / pred)
    return dict(
        absrel=float(np.mean(np.abs(pred - gt) / gt)),
        rmse=float(np.sqrt(np.mean((pred - gt) ** 2))),
        d1=float(np.mean(r < 1.25)),
        d2=float(np.mean(r < 1.25**2)),
        d3=float(np.mean(r < 1.25**3)),
        n=int(pred.size),
    )


def compute_per_image_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    lt_mask: np.ndarray,
    valid_gt: np.ndarray,
    min_px: int = 100,
    depth_bins: list = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)],
) -> dict:
    """Compute Raw, Global-Scale-Aligned, and Region-Scale-Aligned metrics.

    Global Alignment fits s_global ONCE on all valid pixels in the image:
        s_global = median(gt[valid]) / median(pred[valid])
    and evaluates (pred * s_global) on all sub-regions (lowtex, textured, bins).
    """
    valid_all = valid_gt & (pred > 1e-3) & (gt > 1e-3) & np.isfinite(pred) & np.isfinite(gt)
    if valid_all.sum() < min_px:
        return None

    p_all = pred[valid_all].astype(np.float64)
    g_all = gt[valid_all].astype(np.float64)
    s_global = float(np.median(g_all) / np.median(p_all))

    regions = {
        "all": np.ones_like(lt_mask, dtype=bool),
        "lowtex": lt_mask,
        "textured": ~lt_mask,
    }
    # Add depth bins
    for b_min, b_max in depth_bins:
        b_name = f"depth_{b_min:.1f}_{b_max:.1f}m"
        regions[b_name] = (gt >= b_min) & (gt < b_max)

    out = {
        "info": {"s_global": s_global, "n_valid": int(valid_all.sum()), "pred_med": float(np.median(p_all)), "gt_med": float(np.median(g_all))},
        "raw": {},
        "global_aligned": {},
        "region_aligned": {},
    }

    for r_name, r_mask in regions.items():
        m = valid_all & r_mask
        if m.sum() < min_px:
            continue
        p = pred[m].astype(np.float64)
        g = gt[m].astype(np.float64)

        # 1. Raw
        out["raw"][r_name] = _stats(p, g)
        # 2. Global-Scale Aligned (preserves spatial scale inconsistencies)
        out["global_aligned"][r_name] = _stats(p * s_global, g)
        # 3. Region-Scale Aligned (per-mask independent fit)
        s_region = float(np.median(g) / np.median(p))
        al_reg = _stats(p * s_region, g)
        al_reg["scale"] = s_region
        out["region_aligned"][r_name] = al_reg

    return out


def bootstrap_ci(per_scene_vals, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and (1-alpha) percentile bootstrap CI over scenes."""
    v = np.asarray([x for x in per_scene_vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(1)
    return (
        float(v.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def summarize(records: list, align_type: str = "global_aligned", region: str = "lowtex", metric: str = "absrel"):
    """Extract and bootstrap-summarize a specific metric across scene records."""
    vals = []
    for r in records:
        if r and (align_type in r) and (region in r[align_type]):
            val = r[align_type][region].get(metric)
            if val is not None and np.isfinite(val):
                vals.append(val)
    return bootstrap_ci(vals)