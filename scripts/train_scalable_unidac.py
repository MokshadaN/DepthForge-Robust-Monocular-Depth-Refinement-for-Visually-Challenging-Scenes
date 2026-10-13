"""
UniDAC Multi-Domain Scalable Training & Verification Pipeline.

Supports:
- Biological Camouflage (COD10K / CAMO-FS)
- Transparent Glass & Refractive Objects (ClearGrasp / SeeClear / Trans10K)
- Monochromatic & Textureless Architecture (ScanNet++ / NYU Depth v2)

Features:
- Automatic CUDA / CPU detection
- Multi-Scale Dilated Residual Adapter (Atrous Convolutions d=1,2,4)
- Train & Validation Loss tracking per epoch
- Validation Boundary Sharpness (BSI) tracking per epoch
- Automated Dual-Axis Loss Curve plotting (chart_train_val_loss_curves.png)
- Blind evaluation & tangible visual panel generation on held-out test partition
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import load_sample_config, UniDACDepthRunner
from phase4.compare_all_models import MiDaSModelBackend
from phase7.model_adapter_v2 import MultiScaleDilatedAdapter


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def compute_energy_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    norm = grad / (grad.max() + 1e-5)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Texture & Transparency Adapter for UniDAC")
    parser.add_argument("--data_dir", type=str, default="phase7/data/unified_500", help="Path to unified dataset directory")
    parser.add_argument("--splits_file", type=str, default="phase7/data/unified_splits.json", help="Path to dataset splits JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--output_dir", type=str, default="phase7/results", help="Directory to save visual results & plots")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max_train_samples", type=int, default=50, help="Max cached samples for train set (set higher on GPU)")
    parser.add_argument("--max_val_samples", type=int, default=20, help="Max cached samples for val set")
    parser.add_argument("--max_test_samples", type=int, default=20, help="Max cached samples for test set")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    print("=" * 88)
    print("  UNIDAC MULTI-DOMAIN ADAPTIVE TRAINING PIPELINE")
    print(f"  Target Device: {device} | Total Epochs: {args.epochs} | Learning Rate: {args.lr}")
    print("=" * 88)

    data_dir = Path(args.data_dir).resolve()
    splits_p = Path(args.splits_file).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoint_dir).resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if not splits_p.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_p}. Please run phase7/build_unified_multi_domain_dataset.py first.")

    with open(splits_p, "r") as fp:
        splits = json.load(fp)

    train_files = splits["train_files"]
    val_files = splits["val_files"]
    test_files = splits["test_files"]

    print(f"Dataset Partitions: Total={splits.get('total', len(train_files)+len(val_files)+len(test_files))} | Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}\n")

    # 1. Initialize Base UniDAC Runner
    print("Loading Pretrained UniDAC Foundation Model...")
    sample_cfg = load_sample_config(Path("demo/input/scannetpp_sample.json").resolve())
    unidac_runner = UniDACDepthRunner(
        checkpoint=checkpoints_dir / "unidac.pt",
        sample=sample_cfg,
        device=str(device)
    )

    # 2. Pre-cache feature tensors
    temp_p = out_dir / "temp_cache_in.jpg"

    def cache_partition(files, max_count, name):
        batch = []
        selected = files[:max_count]
        print(f"Caching {len(selected)} samples for {name} partition...")
        t0 = time.time()
        for idx, fname in enumerate(selected, 1):
            img_p = data_dir / fname
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            tw, th = 320, int(h * 320 / w)
            tw = (tw // 16) * 16
            th = (th // 16) * 16
            img_res = cv2.resize(img, (tw, th))

            cv2.imwrite(str(temp_p), img_res)
            base_d = unidac_runner.predict(temp_p, output_size=(th, tw))

            rgb_t = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            depth_t = torch.from_numpy(base_d).unsqueeze(0).unsqueeze(0).float()
            energy_t = compute_energy_tensor(img_res)

            batch.append({
                "name": fname,
                "rgb": rgb_t.to(device),
                "depth": depth_t.to(device),
                "energy": energy_t.to(device),
                "shape": (th, tw)
            })
        print(f"  -> Finished {name} in {time.time()-t0:.1f}s ({len(batch)} active tensors)")
        return batch

    train_batch = cache_partition(train_files, args.max_train_samples, "Train")
    val_batch = cache_partition(val_files, args.max_val_samples, "Validation")
    test_batch = cache_partition(test_files, args.max_test_samples, "Test")

    if temp_p.exists():
        temp_p.unlink()

    # 3. Model, Optimizer, Scheduler
    adapter = MultiScaleDilatedAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loss_hist, val_loss_hist, val_bsi_hist = [], [], []
    best_val_loss = float("inf")
    best_ckpt_p = checkpoints_dir / "unidac_multi_domain_500.pt"

    print(f"\nBeginning Training Loop ({args.epochs} Epochs)...")

    for epoch in range(1, args.epochs + 1):
        # A. Training
        adapter.train()
        total_train_loss = 0.0

        for sample in train_batch:
            optimizer.zero_grad()
            rgb, base_d, energy = sample["rgb"], sample["depth"], sample["energy"]
            refined_d = adapter(base_d, rgb, energy)

            l_scale = F.l1_loss(refined_d, base_d)
            gx_ref = refined_d[:, :, :, 1:] - refined_d[:, :, :, :-1]
            gy_ref = refined_d[:, :, 1:, :] - refined_d[:, :, :-1, :]

            b_mask_x = (energy[:, :, :, 1:] > 0.15).float()
            b_mask_y = (energy[:, :, 1:, :] > 0.15).float()
            l_edge = -0.06 * (torch.mean(torch.abs(gx_ref) * b_mask_x) + torch.mean(torch.abs(gy_ref) * b_mask_y))

            flat_mask_x = (energy[:, :, :, 1:] < 0.10).float()
            flat_mask_y = (energy[:, :, 1:, :] < 0.10).float()
            l_flat = 0.18 * (torch.mean(torch.abs(gx_ref) * flat_mask_x) + torch.mean(torch.abs(gy_ref) * flat_mask_y))

            loss = l_scale + l_edge + l_flat
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_batch)
        train_loss_hist.append(avg_train_loss)

        # B. Validation
        adapter.eval()
        total_val_loss = 0.0
        val_bsi_sum = 0.0

        with torch.no_grad():
            for sample in val_batch:
                rgb, base_d, energy = sample["rgb"], sample["depth"], sample["energy"]
                refined_d = adapter(base_d, rgb, energy)

                l_scale = F.l1_loss(refined_d, base_d)
                gx_ref = refined_d[:, :, :, 1:] - refined_d[:, :, :, :-1]
                gy_ref = refined_d[:, :, 1:, :] - refined_d[:, :, :-1, :]

                b_mask_x = (energy[:, :, :, 1:] > 0.15).float()
                b_mask_y = (energy[:, :, 1:, :] > 0.15).float()
                l_edge = -0.06 * (torch.mean(torch.abs(gx_ref) * b_mask_x) + torch.mean(torch.abs(gy_ref) * b_mask_y))

                flat_mask_x = (energy[:, :, :, 1:] < 0.10).float()
                flat_mask_y = (energy[:, :, 1:, :] < 0.10).float()
                l_flat = 0.18 * (torch.mean(torch.abs(gx_ref) * flat_mask_x) + torch.mean(torch.abs(gy_ref) * flat_mask_y))

                v_loss = (l_scale + l_edge + l_flat).item()
                total_val_loss += v_loss

                d_np = refined_d.squeeze().cpu().numpy()
                gx = cv2.Sobel(d_np, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(d_np, cv2.CV_32F, 0, 1, ksize=3)
                val_bsi_sum += float(np.percentile(np.sqrt(gx**2 + gy**2), 90))

        avg_val_loss = total_val_loss / len(val_batch)
        avg_val_bsi = val_bsi_sum / len(val_batch)
        val_loss_hist.append(avg_val_loss)
        val_bsi_hist.append(avg_val_bsi)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(adapter.state_dict(), str(best_ckpt_p))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val BSI: {avg_val_bsi:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n[SUCCESS] Best checkpoint saved -> {best_ckpt_p}\n")

    # 4. Save Loss Curve Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs_arr = np.arange(1, args.epochs + 1)

    l1 = ax1.plot(epochs_arr, train_loss_hist, color="#e74c3c", linewidth=2.5, label="Train Loss (Multi-Objective)")
    l2 = ax1.plot(epochs_arr, val_loss_hist, color="#2980b9", linewidth=2.5, linestyle="--", label="Validation Loss")
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Loss (Train / Val)", color="#2c3e50")

    ax2 = ax1.twinx()
    l3 = ax2.plot(epochs_arr, val_bsi_hist, color="#27ae60", linewidth=2.2, label="Validation Boundary Sharpness (BSI)")
    ax2.set_ylabel("Validation Boundary Sharpness", color="#27ae60")

    all_l = l1 + l2 + l3
    ax1.legend(all_l, [l.get_label() for l in all_l], loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.title("Train & Validation Loss Dynamics across Multi-Domain Benchmark")
    plt.tight_layout()
    chart_p = out_dir / "chart_train_val_loss_curves.png"
    plt.savefig(str(chart_p), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss Curves Saved -> {chart_p}")

    # 5. Held-Out Blind Test Set Evaluation
    print("\n" + "=" * 80)
    print("  BLIND EVALUATION ON HELD-OUT TEST SET (UNSEEN SCENES)")
    print("=" * 80)
    adapter.load_state_dict(torch.load(str(best_ckpt_p), map_location=device, weights_only=True))
    adapter.eval()
    midas_backend = MiDaSModelBackend("DPT_Hybrid", device="cpu")

    bsi_base, bsi_ours, bsi_mid = [], [], []
    test_panels = []

    for idx, sample in enumerate(test_batch, 1):
        fname = sample["name"]
        rgb, base_d, energy = sample["rgb"], sample["depth"], sample["energy"]
        with torch.no_grad():
            refined_d = adapter(base_d, rgb, energy)

        d_base = base_d.squeeze().cpu().numpy()
        d_ours = refined_d.squeeze().cpu().numpy()

        img_np = (rgb.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        mid_raw = midas_backend.predict(img_np, {}, output_size=sample["shape"])
        scale = float(np.median(d_base) / (np.median(mid_raw) + 1e-6))
        d_mid = mid_raw * scale

        def calc_bsi(dm):
            gx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
            return float(np.percentile(np.sqrt(gx**2 + gy**2), 90))

        b1, b2, b3 = calc_bsi(d_base), calc_bsi(d_ours), calc_bsi(d_mid)
        bsi_base.append(b1)
        bsi_ours.append(b2)
        bsi_mid.append(b3)

        if idx <= 6:
            def to_magma(d_map):
                d_min, d_max = np.percentile(d_map, 2), np.percentile(d_map, 98)
                norm = np.clip((d_map - d_min) / (d_max - d_min + 1e-5), 0, 1)
                return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            def to_turbo(d_map):
                norm = np.clip(d_map / (np.percentile(d_map, 99) + 1e-5), 0, 1)
                return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

            tw, th = 320, sample["shape"][0]
            p_rgb = cv2.resize(img_np, (tw, th))
            p_base = cv2.resize(to_magma(d_base), (tw, th))
            p_ours = cv2.resize(to_magma(d_ours), (tw, th))
            p_diff = cv2.resize(to_turbo(np.abs(d_ours - d_base)), (tw, th))
            p_mid = cv2.resize(to_magma(d_mid), (tw, th))

            def add_banner(img, title, subtitle):
                banner = np.zeros((46, img.shape[1], 3), dtype=np.uint8)
                banner[:] = [22, 22, 22]
                cv2.putText(banner, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(banner, subtitle, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                return np.vstack([banner, img])

            tag = "Transparent Glass" if "transparent" in fname else ("Textureless Wall" if "textureless" in fname else "Biological Camouflage")
            p1 = add_banner(p_rgb, "1. Unseen Test RGB", tag)
            p2 = add_banner(p_base, "2. UniDAC Base", f"BSI = {b1:.4f}")
            p3 = add_banner(p_ours, "3. Fine-Tuned (Ours)", f"BSI = {b2:.4f}")
            p4 = add_banner(p_diff, "4. Learned Delta", "Delta Refinement Map")
            p5 = add_banner(p_mid, "5. MiDaS v3.1", f"BSI = {b3:.4f}")

            panel = np.hstack([p1, p2, p3, p4, p5])
            test_panels.append(panel)

    if test_panels:
        target_w = 1600
        resized = []
        for p in test_panels:
            h, w = p.shape[:2]
            scaled_h = int(h * target_w / w)
            resized.append(cv2.resize(p, (target_w, scaled_h)))

        banner = np.zeros((80, target_w, 3), dtype=np.uint8)
        banner[:] = [15, 22, 30]
        cv2.putText(banner, "Held-Out Blind Test Benchmark: Multi-Domain 500 Dataset Generalization", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(banner, "Zero-Shot Evaluation on Completely Unseen Camouflage, Glass Refraction & Textureless Interiors", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 240), 1, cv2.LINE_AA)
        divider = np.zeros((6, target_w, 3), dtype=np.uint8)
        divider[:] = [30, 90, 140]

        all_blocks = [banner, divider]
        for idx, tp in enumerate(resized):
            all_blocks.append(tp)
            if idx < len(resized) - 1:
                all_blocks.append(divider)

        master_showcase = np.vstack(all_blocks)
        showcase_p = out_dir / "master_test_set_showcase.jpg"
        cv2.imwrite(str(showcase_p), master_showcase, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Master Blind Test Showcase Saved -> {showcase_p}")

    m_base = np.mean(bsi_base)
    m_ours = np.mean(bsi_ours)
    m_mid = np.mean(bsi_mid)

    print("\n" + "=" * 80)
    print("  FINAL HELD-OUT BLIND TEST RESULTS")
    print("=" * 80)
    print(f"  MiDaS v3.1 (Baseline)          | Mean BSI: {m_mid:.4f}")
    print(f"  UniDAC Base (Pre-Trained)      | Mean BSI: {m_base:.4f}")
    print(f"  UniDAC Multi-Scale Fine-Tuned  | Mean BSI: {m_ours:.4f}  <-- +{((m_ours-m_base)/m_base)*100:.2f}% Boundary Gain")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
