# Phase 3 — Advanced Geometry & Instance Segmentation Pipeline

## Overview

Phase 3 introduces advanced geometric unwarping and instance segmentation modules to resolve the primary failure modes identified in Phase 2:
1. **360° Tangential Perspective Unwarping (Cubemap Patching):** Eliminates equirectangular curve distortion by extracting rectilinear $90^\circ$ FoV perspective patches around the $360^\circ$ horizon before running object detection.
2. **YOLOv11 Instance Segmentation (`yolo11n-seg.pt`):** Replaces 2D bounding boxes with exact polygon instance masks, preventing background wall/desk leakage into object depth averages.
3. **Gradient-Aware & Outlier-Trimmed Depth Aggregation:** Computes spatial depth gradients ($\nabla D$) inside the object mask to strip high-gradient boundary silhouettes and specular highlights.

---

## 📊 Cross-Phase Progression Benchmark

| Camera Projection | Pipeline Phase | MAE (m) | RMSE (m) | AbsRel Error | $\delta < 1.25$ Acc | Evaluated Objects ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fisheye Outdoor (KITTI-360)** | Phase 1 (BBox) | 0.2814 m | 0.2881 m | 0.0848 (8.5%) | 1.0000 | 2 |
| | Phase 2 (Center 0.35) | 0.2814 m | 0.2881 m | 0.0848 (8.5%) | 1.0000 | 2 |
| | **Phase 3 (Seg Mask)** ⭐ | **0.2121 m** | **0.2308 m** | **0.0354 (3.5%)** | **1.0000** | **4** |
| **Fisheye Indoor (ScanNet++)** | Phase 1 (BBox) | 0.5582 m | 0.5931 m | 0.2524 (25.2%) | 0.3333 | 3 |
| | Phase 2 (Center 0.35) | 0.5582 m | 0.5931 m | 0.2524 (25.2%) | 0.3333 | 3 |
| | **Phase 3 (Seg Mask)** ⭐ | **0.4815 m** | **0.4915 m** | **0.2161 (21.6%)** | **0.6667** | **3** |
| **360° ERP (Matterport3D)** | Phase 1 (BBox) | 3.9513 m | 3.9679 m | 1.9065 (190.6%) | 0.0000 | 2 (Hallucinated `train`) |
| | Phase 2 (Center 0.35) | 3.9513 m | 3.9679 m | 1.9065 (190.6%) | 0.0000 | 2 (Hallucinated `train`) |
| | **Phase 3 (Cubemap+Seg)** ⭐| **3.8585 m** | **3.8797 m** | **1.7390 (173.9%)** | **0.0000** | **6 (True: `bed`, `couch`, `chair`)** |

---

## 🎯 Key Breakthroughs in Phase 3

1. **Outdoor Fisheye Accuracy Jump:**
   - AbsRel improved from **8.5% down to 3.5%** on KITTI-360.
   - Mean absolute error reduced to **21.2 cm**.
   - Segmentation masks enabled accurate depth estimation on distant vehicles (truck at 18.9 m with 1.8% error).

2. **Indoor Object Segmentation Gains:**
   - ScanNet++ $\delta < 1.25$ threshold accuracy **doubled from 33.3% to 66.7%**.
   - Mask-guided extraction filtered out table and wall background pixels from small items (`bottle`, `refrigerator`).

3. **360° Panoramic Hallucination Elimination:**
   - In Phase 1 & 2, standard 2D YOLO hallucinated indoor furniture as `train` cars due to equirectangular warping.
   - In Phase 3, cubemap perspective unwarping accurately identified real indoor items (`bed`, `couch`, `chair`, `person`) and mapped polygon masks back to equirectangular coordinates.

---

## 🚀 Reproduction Commands

Execute all commands from `d:/IIITP/Intern_2_Month/UniDAC`:

### 1. Run Phase 3 Pipeline on All Datasets

```powershell
# 1. KITTI-360 (Outdoor Fisheye)
python phase3/pipeline_v3.py `
  --sample-config demo/input/kitti360_sample.json `
  --image demo/input/kitti360_rgb.png `
  --gt-depth demo/input/kitti360_depth.png `
  --yolo-model yolo11n-seg.pt `
  --unidac-checkpoint checkpoints/unidac.pt `
  --output-dir phase3/output_kitti360

# 2. ScanNet++ (Indoor Fisheye)
python phase3/pipeline_v3.py `
  --sample-config demo/input/scannetpp_sample.json `
  --image demo/input/scannetpp_rgb.jpg `
  --gt-depth demo/input/scannetpp_depth.png `
  --yolo-model yolo11n-seg.pt `
  --unidac-checkpoint checkpoints/unidac.pt `
  --output-dir phase3/output_scannetpp

# 3. Matterport3D (360° ERP Panorama with Cubemap Unwarping)
python phase3/pipeline_v3.py `
  --sample-config demo/input/matterport3d_sample.json `
  --image demo/input/matterport3d_rgb.png `
  --gt-depth demo/input/matterport3d_depth.exr `
  --yolo-model yolo11n-seg.pt `
  --unidac-checkpoint checkpoints/unidac.pt `
  --output-dir phase3/output_matterport3d
```

### 2. Generate Progression Benchmark Table

```powershell
python phase3/benchmark_phase3.py
```
> Outputs the unified multi-phase comparison table and saves `phase3/results/phase_progression_metrics.csv`.

---

## 📂 Phase 3 Artifact Structure

```
phase3/
├── erp_patch.py               <- Tangential perspective unwarper & mask reprojector
├── pipeline_v3.py             <- End-to-end segmentation & depth aggregation pipeline
├── benchmark_phase3.py        <- Multi-phase benchmark progression aggregator
├── output_kitti360/           <- Annotated visual, depth map, CSV, and metrics
├── output_scannetpp/          <- Annotated visual, depth map, CSV, and metrics
├── output_matterport3d/       <- Annotated visual, depth map, CSV, and metrics
└── results/
    └── phase_progression_metrics.csv
```
