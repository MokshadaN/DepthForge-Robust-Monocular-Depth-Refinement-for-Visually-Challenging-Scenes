"""
Phase 10: Authoritative Rigorous Evaluator.

Addresses Step 1 Deliverable:
- 'one reliable evaluation script and baseline results'
- Computes Raw, Median-Aligned, and Affine-Aligned metrics.
- Separates domains (Camouflage, Glass/Transparent, Textureless).
- Compares Baseline vs Single-Head (Phase 8) vs Multi-Head (Phase 10).
"""
from __future__ import annotations
import json
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase10.models.adapter_multihead import MultiHeadDilatedAdapter
from phase10.models.adapter_ablations import SingleHeadDilatedAdapter
from phase10.eval.metrics_engine import evaluate_sample_dual, aggregate_metric_dicts

IMG_SIZE = (480, 640)

def compute_sobel_energy(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    norm = mag / (mag.max() + 1e-6)
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float()


def run_evaluation(splits_path: Path,
                   p10_ckpt_path: Path,
                   p8_ckpt_path: Path,
                   output_dir: Path,
                   device_str: str = "cpu"):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # 1. Load Data Splits
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_records = splits.get("test", [])
    print(f"[Phase 10 Eval] Loaded {len(test_records)} test samples from {splits_path}")

    # 2. Load Base Model (MiDaS small)
    print("[Phase 10 Eval] Loading MiDaS Small baseline...")
    base_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    # 3. Load Phase 10 Multi-Head Adapter
    p10_adapter = MultiHeadDilatedAdapter().to(device)
    if p10_ckpt_path.exists():
        state = torch.load(p10_ckpt_path, map_location=device, weights_only=True)
        p10_adapter.load_state_dict(state)
        print(f"[Phase 10 Eval] Loaded Phase 10 Multi-Head weights from {p10_ckpt_path}")
    else:
        print(f"[Phase 10 Eval] WARNING: Checkpoint not found at {p10_ckpt_path}")
    p10_adapter.eval()

    # 4. Load Phase 8 Single-Head Adapter
    p8_adapter = SingleHeadDilatedAdapter().to(device)
    if p8_ckpt_path.exists():
        state8 = torch.load(p8_ckpt_path, map_location=device, weights_only=True)
        # Adapt keys if necessary
        p8_adapter.load_state_dict(state8, strict=False)
        print(f"[Phase 10 Eval] Loaded Phase 8 Single-Head weights from {p8_ckpt_path}")
    p8_adapter.eval()

    # Accumulators: {model_name: {domain: {mode: [metric_dict]}}}
    models = ["baseline", "singlehead_p8", "multihead_p10"]
    modes = ["raw", "median_aligned", "affine_aligned"]
    domains = ["all", "camo", "transparent", "textureless"]

    results = {m: {d: {mode: [] for mode in modes} for d in domains} for m in models}
    per_image_rows = []

    print("[Phase 10 Eval] Processing test samples...")
    for idx, rec in enumerate(test_records, 1):
        rgb_path = REPO_ROOT / rec["rgb"]
        gt_path = REPO_ROOT / rec["depth_gt"]
        dom = rec["domain"]
        name = rec["name"]

        if not rgb_path.exists() or not gt_path.exists():
            continue

        img_bgr = cv2.imread(str(rgb_path))
        if img_bgr is None:
            continue
        img_res = cv2.resize(img_bgr, (IMG_SIZE[1], IMG_SIZE[0]))
        gt_np = np.load(str(gt_path)).astype(np.float32)

        # Base inference
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        inp = transform(img_rgb).to(device)
        with torch.no_grad():
            b_pred = base_midas(inp)
            b_pred = F.interpolate(b_pred.unsqueeze(1), size=IMG_SIZE, mode="bicubic", align_corners=False).squeeze()
            b_min, b_max = b_pred.min(), b_pred.max()
            if b_max > b_min:
                b_pred = (b_pred - b_min) / (b_max - b_min)
            base_np = b_pred.cpu().numpy()

            # Tensors for adapters
            rgb_t = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            base_t = torch.from_numpy(base_np).unsqueeze(0).unsqueeze(0).float().to(device)
            energy = compute_sobel_energy(img_res).to(device)

            # Phase 10 Multi-Head Forward
            p10_out, logits = p10_adapter(base_t, rgb_t, energy)
            p10_np = p10_out.squeeze().cpu().numpy()

            # Phase 8 Single-Head Forward
            p8_out, _ = p8_adapter(base_t, rgb_t, energy)
            p8_np = p8_out.squeeze().cpu().numpy()

        # Compute dual metrics
        m_base = evaluate_sample_dual(base_np, gt_np)
        m_p8   = evaluate_sample_dual(p8_np, gt_np)
        m_p10  = evaluate_sample_dual(p10_np, gt_np)

        eval_pairs = [("baseline", m_base), ("singlehead_p8", m_p8), ("multihead_p10", m_p10)]
        row = {"name": name, "domain": dom}

        for m_name, m_dict in eval_pairs:
            for mode in modes:
                for metric_k, metric_v in m_dict[mode].items():
                    row[f"{m_name}_{mode}_{metric_k}"] = metric_v

            # Record in accumulators
            for mode in modes:
                results[m_name]["all"][mode].append(m_dict[mode])
                if dom in results[m_name]:
                    results[m_name][dom][mode].append(m_dict[mode])

        per_image_rows.append(row)
        if idx % 20 == 0 or idx == len(test_records):
            print(f"  Processed [{idx}/{len(test_records)}]")

    # Save Per-Image CSV
    csv_path = output_dir / "phase10_rigorous_per_image_metrics.csv"
    if per_image_rows:
        fieldnames = list(per_image_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"[Phase 10 Eval] Saved per-image metrics to {csv_path}")

    # Generate Summary Aggregations
    summary_data = {}
    for dom in domains:
        summary_data[dom] = {}
        for mode in modes:
            summary_data[dom][mode] = {}
            for m_name in models:
                summary_data[dom][mode][m_name] = aggregate_metric_dicts(results[m_name][dom][mode])

    json_path = output_dir / "phase10_rigorous_summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Phase 10 Eval] Saved summary JSON to {json_path}")

    # Print Formatted Markdown Table for Step 1
    print_markdown_table(summary_data, output_dir / "evaluation_benchmark_report.md")
    return summary_data


def print_markdown_table(summary: dict, out_md_path: Path = None):
    lines = []
    lines.append("# Step 1: Rigorous Evaluation Report (Raw vs Median-Scale vs Affine-Fitted)")
    lines.append("\nEvaluation comparing Baseline vs Single-Head (Phase 8) vs Multi-Head (Phase 10) across optical edge cases.\n")

    for mode_name, mode_key, desc in [
        ("1. Primary Raw Unaligned Metrics", "raw", "Evaluated directly without test-time target fitting or scaling."),
        ("2. Standard Median Scale-Aligned Metrics", "median_aligned", "Normalized using global median scale (standard monocular benchmark protocol)."),
        ("3. Affine-Fitted Least-Squares (Historical Diagnostic)", "affine_aligned", "Target-fitted affine alignment (s*pred + t). Preserved for audit comparison.")
    ]:
        lines.append(f"\n### {mode_name}\n*{desc}*\n")
        lines.append(f"| {'Domain':<15} | {'Model':<20} | {'AbsRel (v)':<12} | {'RMSE (v)':<10} | {'SILog (v)':<10} | {'d1 < 1.25 (^)':<14} |")
        lines.append("|" + "-" * 17 + "|" + "-" * 22 + "|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 16 + "|")

        for dom in ["all", "camo", "transparent", "textureless"]:
            dom_name = "Overall (All)" if dom == "all" else dom.capitalize()
            for m_key, m_label in [("baseline", "Base MiDaS"),
                                   ("singlehead_p8", "Phase 8 SingleHead"),
                                   ("multihead_p10", "Phase 10 MultiHead (Ours)")]:
                m = summary[dom][mode_key][m_key]
                lines.append(f"| {dom_name:<15} | {m_label:<20} | {m.get('AbsRel', 0):.4f}       | {m.get('RMSE', 0):.4f}     | {m.get('SILog', 0):.4f}     | {m.get('d1', 0):.4f}         |")
            lines.append("|" + "-" * 17 + "|" + "-" * 22 + "|" + "-" * 14 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 16 + "|")

    md_content = "\n".join(lines)
    if out_md_path:
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[Phase 10 Eval] Formatted benchmark report saved to {out_md_path}")
    return md_content


if __name__ == "__main__":
    splits_file = REPO_ROOT / "phase8" / "data" / "splits.json"
    p10_weights = REPO_ROOT / "checkpoints" / "phase10_multihead_best.pt"
    p8_weights  = REPO_ROOT / "checkpoints" / "phase8_best.pt"
    out_dir     = REPO_ROOT / "phase10" / "eval" / "results"

    run_evaluation(splits_file, p10_weights, p8_weights, out_dir, device_str="cpu")
