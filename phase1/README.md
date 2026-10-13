# Phase 1 — YOLO + UniDAC Object-Level Distance Estimation

This is a small integration layer around the official UniDAC repository. It does **not** train or modify UniDAC or YOLO.

## Pipeline

```text
RGB image
   ├──────────────> YOLO11n ────────> object boxes
   │                                      │
   └──────────────> UniDAC ────────> metric depth map
                                          │
                          object box + depth ROI
                                          │
                                          v
                               representative depth
                                          │
                                          v
                                annotated image + CSV
```

## Official repositories

- UniDAC: https://github.com/girish1511/UniDAC
- UniDAC paper: https://arxiv.org/abs/2603.27105
- Ultralytics: https://github.com/ultralytics/ultralytics

## 1. Get UniDAC

Clone the official repository and put this `phase1/` folder into its root:

```bash
git clone https://github.com/girish1511/UniDAC.git
cd UniDAC
# copy phase1/ here
```

The official UniDAC repo documents Python 3.10.18 and a PyTorch 2.7.0 + CUDA 11.8 setup, followed by `pip install -r requirements.txt`.

## 2. Install UniDAC dependencies

Use the official commands from the UniDAC README, then install the Phase-1 detector dependency:

```bash
pip install -r requirements.txt
pip install -r phase1/../phase1_yolounidac/requirements_phase1.txt
```

When the folder is already copied into the UniDAC repo, the simpler command is:

```bash
pip install ultralytics
```

## 3. Get the UniDAC checkpoint

Download the official pretrained checkpoint and place it at:

```text
UniDAC/checkpoints/unidac.pt
```

## 4. First end-to-end run

The script defaults to the official KITTI-360 sample included by UniDAC:

```bash
python phase1/pipeline.py
```

or explicitly:

```bash
python phase1/pipeline.py \
  --image demo/input/kitti360_rgb.png \
  --yolo-model yolo11n.pt \
  --unidac-checkpoint checkpoints/unidac.pt \
  --device cuda
```

The first YOLO run downloads `yolo11n.pt` automatically if it is not already present.

## Outputs

```text
phase1/output/
├── annotated_result.jpg    # RGB image + object boxes + estimated depth
├── depth.npy               # raw remapped UniDAC depth array
├── depth_color.png         # depth visualization
├── object_depths.csv       # object-level results
└── run_config.json
```

## Object-level depth aggregation

The baseline uses the central 50% of each YOLO bounding box, removes invalid values and trims the lowest/highest 10% when enough pixels are available, then reports the median.

You can compare against the full-box baseline:

```bash
python phase1/pipeline.py --roi-mode bbox
```

## Important coordinate detail

UniDAC internally processes the camera image in its canonical/ERP representation. The script explicitly maps the predicted depth back to the original KITTI-360 camera coordinates before using YOLO bounding boxes. This keeps the YOLO and depth coordinate systems aligned.

## If model construction tries to download DINOv3

By default the script sets the DINOv3 `pretrained` field to `None` and then loads the full released UniDAC checkpoint. If your checkpoint/version requires the separate DINOv3 backbone, rerun with:

```bash
python phase1/pipeline.py --download-dinov3-backbone
```

## Phase-1 scope

Do not train anything yet. The goal is simply:

```text
one RGB image
  -> YOLO object detection
  -> UniDAC metric depth
  -> object-level depth/distance estimate
  -> visualization + CSV
```

After this is stable, Phase 2 can evaluate robustness across perspective, fisheye and 360° imagery and under challenging visual conditions.
