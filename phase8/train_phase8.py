"""
Phase 8 — Supervised Fine-Tuning of UniDAC with Proper Metrics & Logging.

Fixes every issue identified in the Phase 7 audit:
  ✓ Cross-model supervision: DPT_Large pseudo-GT (not UniDAC self-distillation)
  ✓ Loss: SILog + gradient consistency (no contradicting losses)
  ✓ Metrics: AbsRel, δ₁, δ₂, δ₃, SILog (not BSI)
  ✓ Disk caching: UniDAC base-depth cached to .pt files (run once)
  ✓ Per-domain evaluation: separate metrics for camo / transparent / textureless
  ✓ Logging: per-epoch CSV + final summary report + visual comparison panels
  ✓ Best checkpoint by val AbsRel (not val loss)

Architecture (unchanged — adapter is good):
  Input  : [UniDAC_base_depth (1ch) | RGB (3ch) | Sobel_energy (1ch)] = 5ch
  Output : Refined depth (1ch) = base_depth + 0.30 * tanh(Δ)
  Adapter: Multi-Scale Dilated Residual (dilation 1, 2, 4 in parallel)

Run:
    # Step 1 — build data (one time, ~25 min on CPU)
    python phase8/build_phase8_data.py

    # Step 2 — train
    python phase8/train_phase8.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import load_sample_config, UniDACDepthRunner
from phase4.compare_all_models import MiDaSModelBackend
from phase7.model_adapter_v2 import MultiScaleDilatedAdapter
from phase8.losses import combined_loss, silog_loss
from phase8.metrics import compute_metrics, aggregate_metrics, format_metrics

# ── Paths ─────────────────────────────────────────────────────────────────────
P8_DIR       = Path("phase8").resolve()
DATA_DIR     = P8_DIR / "data"
SPLITS_F     = DATA_DIR / "splits.json"
CACHE_DIR    = P8_DIR / "cache"
RESULTS_DIR  = P8_DIR / "results"
CHARTS_DIR   = RESULTS_DIR / "charts"
PANELS_DIR   = RESULTS_DIR / "panels"
CKPT_DIR     = Path("checkpoints").resolve()
BEST_CKPT    = CKPT_DIR / "phase8_best.pt"
TRAIN_LOG    = RESULTS_DIR / "training_log.csv"
EVAL_REPORT  = RESULTS_DIR / "evaluation_report.txt"

for d in [CACHE_DIR, CHARTS_DIR, PANELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_EPOCHS   = 40
LR           = 1.2e-3
WEIGHT_DECAY = 1e-4
IMG_SIZE     = (480, 640)   # (H, W)
W_SILOG      = 1.0
W_GRAD       = 0.5


# ── Sobel energy ──────────────────────────────────────────────────────────────
def sobel_energy(img_bgr: np.ndarray) -> torch.Tensor:
    """Compute normalised Sobel gradient magnitude → (1,1,H,W) tensor."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    norm = mag / (mag.max() + 1e-6)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()


# ── Disk cache ────────────────────────────────────────────────────────────────
def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pt"


def build_cache(records: list[dict], unidac: UniDACDepthRunner) -> None:
    """
    Run UniDAC once per image, save (rgb_tensor, base_depth, sobel) to disk.
    Skips images that are already cached (resumes after interruption).
    """
    n      = len(records)
    to_run = [r for r in records if not cache_path(r["name"]).exists()]
    print(f"  Cache: {n - len(to_run)}/{n} already cached. Building {len(to_run)} new...")

    tmp = CACHE_DIR / "_tmp_infer.jpg"
    for idx, r in enumerate(to_run, 1):
        rgb_p = REPO_ROOT / r["rgb"]
        img   = cv2.imread(str(rgb_p))
        if img is None:
            print(f"    [SKIP] {r['name']} — cannot read RGB")
            continue

        img_res = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        cv2.imwrite(str(tmp), img_res)

        base_d = unidac.predict(tmp, output_size=IMG_SIZE)   # (H, W) float32

        rgb_t   = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        depth_t = torch.from_numpy(base_d).unsqueeze(0).float()               # (1,H,W)
        energy_t= sobel_energy(img_res).squeeze(0)                             # (1,H,W)

        torch.save({"rgb": rgb_t, "depth": depth_t, "energy": energy_t},
                   str(cache_path(r["name"])))

        if idx % 25 == 0 or idx == len(to_run):
            print(f"    Cached {idx}/{len(to_run)} — {r['name']}")

    if tmp.exists():
        tmp.unlink()


def load_sample(r: dict) -> dict | None:
    """Load cached tensor bundle + pseudo-GT depth for one sample."""
    cp = cache_path(r["name"])
    if not cp.exists():
        return None

    bundle  = torch.load(str(cp), weights_only=True)
    depth_p = REPO_ROOT / r["depth_gt"]
    if not depth_p.exists():
        return None

    gt_np   = np.load(str(depth_p)).astype(np.float32)
    gt_t    = torch.from_numpy(gt_np).unsqueeze(0)              # (1,H,W)

    return {
        "name":   r["name"],
        "domain": r["domain"],
        "rgb":    bundle["rgb"].unsqueeze(0),                    # (1,3,H,W)
        "depth":  bundle["depth"].unsqueeze(0),                  # (1,1,H,W)
        "energy": bundle["energy"].unsqueeze(0),                 # (1,1,H,W)
        "gt":     gt_t.unsqueeze(0),                             # (1,1,H,W)
    }


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(adapter, samples, optimizer):
    adapter.train()
    losses = []
    for s in samples:
        optimizer.zero_grad()
        refined = adapter(s["depth"], s["rgb"], s["energy"])
        loss    = combined_loss(refined, s["gt"], W_SILOG, W_GRAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(adapter, samples) -> dict[str, float]:
    adapter.eval()
    all_metrics = []
    for s in samples:
        refined = adapter(s["depth"], s["rgb"], s["energy"])
        m = compute_metrics(refined.squeeze(), s["gt"].squeeze())
        all_metrics.append(m)
    return aggregate_metrics(all_metrics)


@torch.no_grad()
def evaluate_by_domain(adapter, samples) -> dict[str, dict]:
    adapter.eval()
    by_domain: dict[str, list] = {}
    for s in samples:
        refined = adapter(s["depth"], s["rgb"], s["energy"])
        m = compute_metrics(refined.squeeze(), s["gt"].squeeze())
        by_domain.setdefault(s["domain"], []).append(m)
    return {dom: aggregate_metrics(ms) for dom, ms in by_domain.items()}


# ── Visualisation helpers ─────────────────────────────────────────────────────
def _to_magma(d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    norm   = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def _banner(img: np.ndarray, line1: str, line2: str = "") -> np.ndarray:
    bar = np.full((46, img.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(bar, line1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (255, 255, 255), 1, cv2.LINE_AA)
    if line2:
        cv2.putText(bar, line2, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (170, 200, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_comparison_panel(s: dict, adapter, midas, panel_path: Path,
                           m_base: dict, m_ours: dict, m_midas: dict):
    """Save 5-column panel: RGB | Base UniDAC | Ours | GT depth | MiDaS."""
    W = 320
    img_bgr = (s["rgb"].squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    H       = img_bgr.shape[0]

    refined = adapter(s["depth"], s["rgb"], s["energy"])

    d_base  = s["depth"].squeeze().numpy()
    d_ours  = refined.squeeze().numpy()
    d_gt    = s["gt"].squeeze().numpy()

    # MiDaS prediction
    mid_raw = midas.predict(img_bgr, sample_cfg={}, output_size=(H, W))
    scale   = float(np.median(d_base) / (np.median(mid_raw) + 1e-6))
    d_mid   = mid_raw * scale

    def col(d, w=W, h=H): return cv2.resize(_to_magma(d), (w, h))

    p_rgb  = cv2.resize(img_bgr, (W, H))
    domain = s["domain"].capitalize()

    p1 = _banner(p_rgb,          "1. Input RGB",       domain)
    p2 = _banner(col(d_base),    "2. UniDAC Base",
                 f"AbsRel={m_base['AbsRel']:.4f}  δ₁={m_base['d1']:.4f}")
    p3 = _banner(col(d_ours),    "3. Phase 8 (Ours)",
                 f"AbsRel={m_ours['AbsRel']:.4f}  δ₁={m_ours['d1']:.4f}")
    p4 = _banner(col(d_gt),      "4. DPT_Large GT",
                 f"AbsRel={m_midas['AbsRel']:.4f}  δ₁={m_midas['d1']:.4f}")
    p5 = _banner(col(d_mid),     "5. MiDaS v3.1",      "")

    panel = np.hstack([p1, p2, p3, p4, p5])
    cv2.imwrite(str(panel_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ── Main training pipeline ────────────────────────────────────────────────────
def run():
    print("=" * 88)
    print("  PHASE 8: SUPERVISED FINE-TUNING — FIXED LOSSES + PROPER METRICS")
    print("=" * 88)

    # ── Load splits ──
    if not SPLITS_F.exists():
        print(f"\n[ERROR] {SPLITS_F} not found.")
        print("  Run:  python phase8/build_phase8_data.py   first.\n")
        sys.exit(1)

    with open(SPLITS_F) as fp:
        splits = json.load(fp)

    train_recs = splits["train"]
    val_recs   = splits["val"]
    test_recs  = splits["test"]

    print(f"\nDataset  : {splits['total']} images | "
          f"Train {splits['split_counts']['train']} | "
          f"Val {splits['split_counts']['val']} | "
          f"Test {splits['split_counts']['test']}")
    print(f"Teacher  : {splits['teacher_model']}")
    print(f"Domains  : {splits['domain_counts']}\n")

    # ── UniDAC base model ──
    print("Loading UniDAC foundation model...")
    sample_cfg  = load_sample_config(Path("demo/input/scannetpp_sample.json").resolve())
    unidac      = UniDACDepthRunner(
        checkpoint=CKPT_DIR / "unidac.pt",
        sample=sample_cfg
    )

    # ── Build feature cache ──
    print("\nBuilding / verifying UniDAC feature cache...")
    all_recs = train_recs + val_recs + test_recs
    build_cache(all_recs, unidac)

    # ── Load all cached samples into memory ──
    print("\nLoading cached samples into memory...")
    def load_split(recs, label):
        loaded = [s for r in recs if (s := load_sample(r)) is not None]
        print(f"  {label}: {len(loaded)}/{len(recs)} loaded")
        return loaded

    train_s = load_split(train_recs, "Train")
    val_s   = load_split(val_recs,   "Val  ")
    test_s  = load_split(test_recs,  "Test ")

    # ── Adapter + optimiser ──
    adapter   = MultiScaleDilatedAdapter()
    optimizer = torch.optim.AdamW(adapter.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    # ── CSV log header ──
    with open(TRAIN_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss",
                         "val_AbsRel", "val_SILog", "val_d1",
                         "val_SqRel",  "val_RMSE",  "lr"])

    # ── Training loop ──
    print(f"\nTraining {NUM_EPOCHS} epochs | AdamW lr={LR} | "
          f"Loss: SILog×{W_SILOG} + GradLoss×{W_GRAD}\n")

    best_val_absrel = float("inf")
    train_losses, val_absrels, val_d1s = [], [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Shuffle training samples each epoch
        import random as _random
        _random.shuffle(train_s)

        tr_loss = train_one_epoch(adapter, train_s, optimizer)
        val_m   = evaluate(adapter, val_s)
        scheduler.step()

        train_losses.append(tr_loss)
        val_absrels.append(val_m["AbsRel"])
        val_d1s.append(val_m["d1"])

        # Save best checkpoint
        if val_m["AbsRel"] < best_val_absrel:
            best_val_absrel = val_m["AbsRel"]
            torch.save(adapter.state_dict(), str(BEST_CKPT))
            best_tag = " ← best"
        else:
            best_tag = ""

        # CSV log
        with open(TRAIN_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}",
                f"{val_m['AbsRel']:.6f}", f"{val_m['SILog']:.6f}",
                f"{val_m['d1']:.6f}",     f"{val_m['SqRel']:.6f}",
                f"{val_m['RMSE']:.6f}",
                f"{scheduler.get_last_lr()[0]:.7f}"
            ])

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
                  f"Loss={tr_loss:.5f} | "
                  f"AbsRel={val_m['AbsRel']:.4f} | "
                  f"δ₁={val_m['d1']:.4f} | "
                  f"SILog={val_m['SILog']:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.6f} | "
                  f"{elapsed:.1f}s{best_tag}")

    print(f"\n[TRAINING DONE] Best val AbsRel = {best_val_absrel:.4f} → {BEST_CKPT}\n")

    # ── Plot loss + metric curves ──
    epochs = np.arange(1, NUM_EPOCHS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 8: Training Curves — Supervised Depth Fine-Tuning",
                 fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_losses, color="#e74c3c", lw=2.2, label="Train Loss (SILog+Grad)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(alpha=0.35, ls="--")

    ax2.plot(epochs, val_absrels, color="#2980b9", lw=2.2, label="Val AbsRel (↓)")
    ax2_r = ax2.twinx()
    ax2_r.plot(epochs, val_d1s, color="#27ae60", lw=2.2, ls="--", label="Val δ₁ (↑)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AbsRel", color="#2980b9")
    ax2_r.set_ylabel("δ₁ accuracy", color="#27ae60")
    ax2.set_title("Validation Metrics")

    lines = [plt.Line2D([0], [0], color="#2980b9", lw=2),
             plt.Line2D([0], [0], color="#27ae60", lw=2, ls="--")]
    ax2.legend(lines, ["Val AbsRel (↓)", "Val δ₁ (↑)"], loc="upper right")
    ax2.grid(alpha=0.35, ls="--")

    plt.tight_layout()
    chart_p = CHARTS_DIR / "training_curves.png"
    plt.savefig(str(chart_p), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {chart_p}")

    # ── Blind test evaluation ──
    print("\n" + "=" * 88)
    print("  BLIND TEST SET EVALUATION")
    print("=" * 88)

    adapter.load_state_dict(torch.load(str(BEST_CKPT), weights_only=True))
    midas = MiDaSModelBackend("DPT_Large", device="cpu")

    # Per-sample metrics for three models
    base_metrics_all, ours_metrics_all, midas_metrics_all = [], [], []

    for idx, s in enumerate(test_s, 1):
        gt      = s["gt"].squeeze()
        base_d  = s["depth"].squeeze()
        refined = adapter(s["depth"], s["rgb"], s["energy"]).squeeze()

        img_bgr = (s["rgb"].squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mid_raw = midas.predict(img_bgr, sample_cfg={}, output_size=(IMG_SIZE[0], IMG_SIZE[1]))
        scale   = float(np.median(base_d.numpy()) / (np.median(mid_raw) + 1e-6))
        mid_t   = torch.from_numpy(mid_raw * scale).float()

        m_base  = compute_metrics(base_d,  gt)
        m_ours  = compute_metrics(refined, gt)
        m_midas = compute_metrics(mid_t,   gt)

        base_metrics_all.append(m_base)
        ours_metrics_all.append(m_ours)
        midas_metrics_all.append(m_midas)

        # Save panel for first 8 test samples
        if idx <= 8:
            panel_p = PANELS_DIR / f"panel_{idx:02d}_{s['name']}.jpg"
            save_comparison_panel(s, adapter, midas, panel_p,
                                  m_base, m_ours, m_midas)

    # Aggregate
    m_base_agg  = aggregate_metrics(base_metrics_all)
    m_ours_agg  = aggregate_metrics(ours_metrics_all)
    m_midas_agg = aggregate_metrics(midas_metrics_all)

    # Per-domain
    domain_results = evaluate_by_domain(adapter, test_s)

    # ── Print + save report ──
    def pct_change(new, old, metric_higher_better=False):
        if metric_higher_better:
            delta = (new - old) / (old + 1e-9) * 100
            arrow = "↑" if delta > 0 else "↓"
        else:
            delta = (old - new) / (old + 1e-9) * 100
            arrow = "↑" if delta > 0 else "↓"
        return f"{arrow}{abs(delta):.1f}%"

    report_lines = [
        "=" * 88,
        "  PHASE 8 FINAL EVALUATION REPORT — BLIND TEST SET",
        "=" * 88,
        f"  Dataset       : {splits['total']} images, {splits['unique_scenes']} unique scenes",
        f"  Teacher model : {splits['teacher_model']}",
        f"  Test samples  : {len(test_s)}",
        "",
        f"  {'Model':<36} {'AbsRel':>8} {'SqRel':>8} {'RMSE':>8} "
        f"{'SILog':>8} {'δ₁':>8} {'δ₂':>8} {'δ₃':>8}",
        "  " + "-" * 84,
        f"  {'MiDaS DPT_Large (Baseline)':<36} "
        f"{m_midas_agg['AbsRel']:>8.4f} {m_midas_agg['SqRel']:>8.4f} "
        f"{m_midas_agg['RMSE']:>8.4f} {m_midas_agg['SILog']:>8.4f} "
        f"{m_midas_agg['d1']:>8.4f} {m_midas_agg['d2']:>8.4f} "
        f"{m_midas_agg['d3']:>8.4f}",
        f"  {'UniDAC Base (Pretrained)':<36} "
        f"{m_base_agg['AbsRel']:>8.4f} {m_base_agg['SqRel']:>8.4f} "
        f"{m_base_agg['RMSE']:>8.4f} {m_base_agg['SILog']:>8.4f} "
        f"{m_base_agg['d1']:>8.4f} {m_base_agg['d2']:>8.4f} "
        f"{m_base_agg['d3']:>8.4f}",
        f"  {'UniDAC Phase 8 Fine-Tuned (Ours)':<36} "
        f"{m_ours_agg['AbsRel']:>8.4f} {m_ours_agg['SqRel']:>8.4f} "
        f"{m_ours_agg['RMSE']:>8.4f} {m_ours_agg['SILog']:>8.4f} "
        f"{m_ours_agg['d1']:>8.4f} {m_ours_agg['d2']:>8.4f} "
        f"{m_ours_agg['d3']:>8.4f}",
        "",
        f"  Gain vs UniDAC Base "
        f"→ AbsRel {pct_change(m_ours_agg['AbsRel'], m_base_agg['AbsRel'])} | "
        f"δ₁ {pct_change(m_ours_agg['d1'], m_base_agg['d1'], True)} | "
        f"SILog {pct_change(m_ours_agg['SILog'], m_base_agg['SILog'])}",
        "",
        "  Per-Domain Breakdown (Ours — Phase 8 Fine-Tuned):",
        "  " + "-" * 84,
    ]

    for dom, dm in domain_results.items():
        report_lines.append(
            f"  {dom.capitalize():<18} AbsRel={dm['AbsRel']:.4f}  "
            f"δ₁={dm['d1']:.4f}  δ₂={dm['d2']:.4f}  SILog={dm['SILog']:.4f}"
        )

    report_lines += [
        "",
        "  Training Log  -> phase8/results/training_log.csv",
        "  Curves Chart  -> phase8/results/charts/training_curves.png",
        "  Test Panels   -> phase8/results/panels/",
        "  Best Ckpt     -> checkpoints/phase8_best.pt",
        "=" * 88,
    ]

    report_str = "\n".join(report_lines)
    print(report_str)

    with open(EVAL_REPORT, "w") as fp:
        fp.write(report_str + "\n")

    print(f"\nFull report saved → {EVAL_REPORT}\n")


if __name__ == "__main__":
    run()
