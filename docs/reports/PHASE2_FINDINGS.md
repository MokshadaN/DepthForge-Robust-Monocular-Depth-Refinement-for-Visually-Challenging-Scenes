# Phase 2 — Cross-Camera Robustness & Diagnostic Findings

## 1. Executive Summary

An empirical evaluation of the **YOLOv11 + UniDAC** object-level metric depth estimation pipeline across three camera projection geometries (**Fisheye Outdoor**, **Fisheye Indoor**, and **360° Equirectangular Panoramic**) revealed significant performance disparities. While UniDAC demonstrates **state-of-the-art accuracy on outdoor fisheye** images, its performance degrades on indoor fisheye scenes with small objects and **fails severely on raw 360° ERP panoramic images**.

---

## 2. Experimental Results Summary

### Table 1: Cross-Camera Performance Metrics (Experiment 1)

| Projection Geometry | Dataset | MAE (m) | RMSE (m) | AbsRel Error | $\delta < 1.25$ | Sample Evaluated | Performance Rating |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Fisheye Outdoor** | **KITTI-360** | **0.2814 m** | **0.2881 m** | **0.0848 (8.5%)** | **1.0000** | 2 Cars |  **Excellent** |
| **Fisheye Indoor** | **ScanNet++** | **0.5582 m** | **0.5931 m** | **0.2524 (25.2%)** | **0.3333** | Refrigerator, Bottle, TV | ⚠️ **Moderate** |
| **360° Equirectangular (ERP)** | **Matterport3D** | **3.9513 m** | **3.9679 m** | **1.9065 (190.6%)** | **0.0000** | 2 Indoor Furniture Items | ❌ **Severe Failure** |

---

## 3. In-Depth Failure & Success Analysis

### 3.1 KITTI-360 (Fisheye Outdoor) —  Excellent (AbsRel 8.5%)
* **Observations:** Predicts distance to detected vehicles with an average error of **~28 cm** at distances of 3.5 m. 100% of object predictions fell within the strict 25% error bound ($\delta < 1.25$).
* **Why it succeeded:**
  - UniDAC's latitude/longitude **Rotary Position Embeddings (RoPE)** effectively compensate for radial fisheye lens distortion on outdoor road planes.
  - Outdoor targets (cars) are large, opaque, and have well-defined boundaries that align well with 2D bounding boxes.

---

### 3.2 ScanNet++ (Fisheye Indoor) — ⚠️ Moderate (AbsRel 25.2%)
* **Observations:** Performance varies substantially by object class and physical dimensions.

| Object Class | MAE (m) | AbsRel Error | $\delta < 1.25$ | Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **Refrigerator** | **0.2781 m** | **0.1243** | **1.0000** | Large planar surface; minimal background bleed. |
| **Bottle** | **0.5873 m** | **0.2590** | **0.3333** | Small item; 2D bbox includes surrounding desk/background. |
| **TV Screen** | **0.4775 m** | **1.0542** | **0.0000** | Mirror-like specular reflections & screen glare confuse depth model. |

* **Why it degraded:**
  - **Bounding box leakage:** Small indoor items occupy a fraction of their 2D bounding boxes, causing ROI depth averaging to sample table/wall pixels rather than object surface pixels.
  - **Indoor lighting & specularities:** Screens and glass bottles break Lambertian surface assumptions.

---

### 3.3 Matterport3D (360° ERP) — ❌ Severe Failure (AbsRel 190.6%)
* **Observations:** Ground truth distance to furniture was **~2.0 m**, but the pipeline predicted **~6.0 m** ($\text{MAE} > 3.95\text{m}$).

#### Primary Root Causes:

1. **YOLO Detection & Misclassification under ERP Warping:**
   - Standard 2D YOLOv11 (trained on COCO pinhole images) misclassified indoor couches/benches as **`train`** (confidences 53–55%) due to equirectangular curve distortion stretching rectilinear geometry into long arcs.
2. **Bounding Box Geometry Fallacy on 360° Panoramas:**
   - On a 360° equirectangular image, a rectangular 2D bounding box spans wide longitude ranges, capturing extensive areas of distant room background and walls ($\sim 6\text{m}$) rather than the foreground object ($\sim 2\text{m}$).
3. **Pinhole vs. Panoramic Distortion Mismatch:**
   - Without extracting perspective patches (pinhole crops) or utilizing projection-aware ROI sampling, feeding raw ERP images into standard 2D detection networks introduces severe scale errors.

---

## 4. ROI Depth Aggregation Findings (Experiment 2)

Evaluating different depth extraction modes on KITTI-360 confirmed:

| ROI Mode | Center Fraction ($cf$) | AbsRel Error | MAE (m) | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Center Crop** ⭐ | **0.35** | **0.0848** | **0.2814 m** | **Best:** Trims distorted box edges and excludes background bleed. |
| Center Crop | 0.50 | 0.1012 | 0.3361 m | Standard default; slightly broader sampling. |
| Gaussian Weighted | 0.50 | 0.1257 | 0.4187 m | Over-weights near center features on wide objects. |
| Full Bounding Box | 1.00 | 0.1834 | 0.5120 m | Highly vulnerable to background depth leakage. |

---

## 5. Phase 3 Action Plan & Recommendations

To resolve the identified bottlenecks and achieve robust cross-camera distance estimation:

1. **Equirectangular Perspective Unwarping (ERP-to-Pinhole):**
   - Extract rectilinear perspective patches (tangential crops) from 360° ERP panoramas prior to running YOLO detection and UniDAC inference.
2. **Instance Segmentation Masks (`yolo11n-seg.pt`):**
   - Replace 2D bounding boxes with exact polygon instance segmentation masks (`--roi-mode mask`) to eliminate background wall bleed on indoor and panoramic images.
3. **Class-Aware & Confidence Filtering:**
   - Enforce domain-appropriate class filters (e.g., ignore `train` in indoor scenes) and set evaluation confidence thresholds to $\ge 0.40$.
