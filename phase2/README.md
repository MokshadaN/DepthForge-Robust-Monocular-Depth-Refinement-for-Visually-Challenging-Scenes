# Phase 2 — Robustness Study: YOLO + UniDAC Across Camera Types

## Research Question

> **How does camera projection geometry affect object-level distance estimation accuracy
> using pretrained YOLO and UniDAC, and which depth-aggregation strategy is most robust?**

---

> 📄 **Detailed Diagnostic Report:** See [PHASE2_FINDINGS.md](PHASE2_FINDINGS.md) for full failure mode breakdown, indoor reflection issues, and 360° ERP projection analysis.

---

## Three Experiments

| Exp | Question | Status |
|---|---|---|
| **1** | Cross-camera robustness (Fisheye / 360° ERP) | ✅ All 3 demo samples run |
| **2** | Depth aggregation strategy comparison (5 modes) | ✅ Best: center cf=0.35 |
| **3** | Object size × distance error grid | ✅ Live in evaluate.py |

---

## Experiment 1 Results (YOLO + UniDAC, center cf=0.35)

| Camera | Dataset | MAE (m) | AbsRel | δ<1.25 | N |
|---|---|---|---|---|---|
| Fisheye outdoor | KITTI-360 | **0.281** | **0.085** | 1.000 | 2 |
| Fisheye indoor | ScanNet++ | 0.558 | 0.252 | 0.333 | 3 |
| 360° ERP | Matterport3D | 3.951 | 1.907 | 0.000 | 2 |

> Note: ScanNet++ has harder targets (bottles, TV) vs. KITTI-360 (cars).
> A fair per-class comparison requires equal object-type distribution.

---

## Experiment 2 — ROI Aggregation (KITTI-360, 2 confident cars)

| Mode | AbsRel | MAE |
|---|---|---|
| **center cf=0.35** ✅ | **0.085** | **0.281 m** |
| center cf=0.50 | 0.101 | 0.336 m |
| center cf=0.65 | 0.117 | 0.396 m |
| weighted cf=0.50 | 0.126 | 0.419 m |

**Insight:** Tighter crop avoids background depth bleed into the object ROI.
Gaussian weighting adds bias for large close objects (fisheye cars fill the full box).

---

## Model Comparison (UniDAC vs Baselines)

Run with:
```powershell
python phase2/compare_models.py `
  --image demo/input/kitti360_rgb.png `
  --gt    demo/input/kitti360_depth.png `
  --depth-scale 256 `
  --sample-config demo/input/kitti360_sample.json `
  --models unidac midas zoe `
  --output-dir phase2/results/kitti360_comparison
```

Models compared:

| Model | Type | Depth type | Scale-align? |
|---|---|---|---|
| UniDAC | DINOv3 + DPT | Metric | No (raw metric) |
| MiDaS v3.1 | DPT-BEiT-L | Relative (inverse) | ✅ auto |
| ZoeDepth-N | MiDaS + ZoeDepth | Metric | No |
| Depth Anything V2 | ViT-L | Relative | ✅ auto |

---

## Quick Start

```powershell
# Experiment 1 — run all 3 camera types
python phase1/pipeline.py `
  --sample-config demo/input/kitti360_sample.json `
  --image demo/input/kitti360_rgb.png `
  --gt-depth demo/input/kitti360_depth.png `
  --yolo-model yolo11n.pt --unidac-checkpoint checkpoints/unidac.pt `
  --roi-mode center --center-fraction 0.35 `
  --output-dir phase1/output

python phase1/pipeline.py `
  --sample-config demo/input/scannetpp_sample.json `
  --image demo/input/scannetpp_rgb.jpg `
  --gt-depth demo/input/scannetpp_depth.png `
  --yolo-model yolo11n.pt --unidac-checkpoint checkpoints/unidac.pt `
  --roi-mode center --center-fraction 0.35 `
  --output-dir phase1/output_scannetpp

python phase1/pipeline.py `
  --sample-config demo/input/matterport3d_sample.json `
  --image demo/input/matterport3d_rgb.png `
  --gt-depth demo/input/matterport3d_depth.exr `
  --yolo-model yolo11n.pt --unidac-checkpoint checkpoints/unidac.pt `
  --roi-mode center --center-fraction 0.35 `
  --output-dir phase1/output_matterport3d

# Aggregate Experiment 1 table
python phase2/summarise_exp1.py

# Model comparison
python phase2/compare_models.py `
  --image demo/input/kitti360_rgb.png `
  --gt demo/input/kitti360_depth.png `
  --sample-config demo/input/kitti360_sample.json `
  --models unidac midas zoe `
  --output-dir phase2/results/comparison
```

---

## Output Structure

```
phase2/
├── compare_models.py          ← multi-model comparison runner
├── summarise_exp1.py          ← Exp 1 cross-camera table builder
├── results/
│   ├── exp1_camera_metrics.csv
│   └── comparison/
│       ├── comparison_metrics.json    ← ranked model metrics
│       ├── comparison_results.csv     ← per-object all-models
│       ├── sidebyside_depth.jpg       ← depth map grid
│       ├── depth_UniDAC_color.png
│       ├── depth_MiDaS_color.png
│       └── depth_ZoeDepth_color.png
```

---

## Phase 3 Directions (contingent on Phase 2 findings)

| Finding | Next step |
|---|---|
| 360° error >> fisheye | Projection-aware ROI sampling |
| Small objects worst | Depth uncertainty weighting |
| Seg mask >> bbox by >20% AbsRel | Replace bbox with YOLO-seg mask everywhere |
| ZoeDepth/MiDaS beats UniDAC on indoor | Fine-tune UniDAC on ScanNet++ |
