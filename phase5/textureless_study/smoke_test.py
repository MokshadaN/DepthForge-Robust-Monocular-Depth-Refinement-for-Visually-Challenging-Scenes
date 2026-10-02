"""Synthetic smoke test of the harness (no GPU / checkpoints). Run: python smoke_test.py"""
import numpy as np
from evaluate import fit_tau, run, fmt
rng = np.random.default_rng(1)
def make(i):
    img = np.full((240, 320, 3), 128, np.uint8); img[:, 160:] = rng.integers(0, 255, (240, 160, 3))
    img[0, 0] = (i, 0, 0)                                   # scene id, so dummy models can look up GT
    return dict(id=i, rgb=img, gt=np.full((240, 320), 2 + 0.3 * i, np.float32), K=(300, 300, 160, 120))
S = [make(i) for i in range(12)]; tau = fit_tau(S[:4], 40)
gt_of = lambda rgb: 2 + 0.3 * int(rgb[0, 0, 0])
def scale_biased(rgb, K):                                   # right structure, 1.2x scale
    return np.full(rgb.shape[:2], 1.2 * gt_of(rgb), np.float32)
from tar_refine import refine_tar

def flat_noisy(rgb, K):                                     # exact on texture, bumpy on flat half
    d = np.full(rgb.shape[:2], gt_of(rgb), np.float32)
    d[:, :160] *= 1 + 0.3 * np.sin(np.linspace(0, 12, 160))[None, :]
    return d

def tar_refined(rgb, K):
    d = flat_noisy(rgb, K)
    d_ref, _ = refine_tar(d, rgb, K, tau=tau)
    return d_ref

models = {
    "scale_biased": scale_biased,
    "flat_noisy": flat_noisy,
    "tar_refined": tar_refined,
}
t = run(models, S[4:], tau)
for k in ("raw_absrel", "aligned_absrel", "raw_rmse"):
    print(f"\n--- {k} ---")
    fmt(t, k)