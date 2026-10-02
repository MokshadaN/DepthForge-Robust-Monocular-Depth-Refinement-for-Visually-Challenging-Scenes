"""Low-texture mask: T = a*grad + b*var + c*entropy (each normalised), mask = T <= tau.
Fit tau on a VALIDATION split only (see percentile_tau)."""
import numpy as np, cv2
from skimage.filters.rank import entropy as _entropy
from skimage.morphology import disk

def _norm(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

def texture_map(rgb, k=15, w=(1/3, 1/3, 1/3)):
    """rgb: HxWx3 uint8 or float in [0,1]. Returns T in [0,1], higher = more texture."""
    img = (rgb * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gf = g.astype(np.float32)
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    grad = cv2.blur(np.sqrt(gx**2 + gy**2), (k, k))
    mean = cv2.blur(gf, (k, k)); var = np.maximum(cv2.blur(gf**2, (k, k)) - mean**2, 0)
    ent = _entropy(g, disk(k // 2)).astype(np.float32)
    return w[0]*_norm(grad) + w[1]*_norm(var) + w[2]*_norm(ent)

def percentile_tau(T_list, pct=20):
    """Global threshold = pct-th percentile of T over a validation set."""
    return float(np.percentile(np.concatenate([t.ravel() for t in T_list]), pct))

def low_texture_mask(T, tau, min_area=500, close_k=7):
    m = (T <= tau).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)