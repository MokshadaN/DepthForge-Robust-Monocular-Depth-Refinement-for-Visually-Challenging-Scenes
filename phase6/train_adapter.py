"""
Phase 6: Training / Fine-Tuning TextureAwareResidualHead for UniDAC.
Trains on camouflage edge sharpening and textureless planar regularization.
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase1.pipeline import load_sample_config, UniDACDepthRunner
from phase6.model_adapter import TextureAwareResidualHead


def compute_energy_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad_norm = grad / (grad.max() + 1e-5)
    return torch.from_numpy(grad_norm).unsqueeze(0).unsqueeze(0).float()


def train_adapter():
    print("=" * 88)
    print("  PHASE 6: FINE-TUNING UNIDAC WITH TEXTURE-AWARE RESIDUAL ADAPTER")
    print("=" * 88)

    checkpoints_dir = Path("checkpoints").resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    camo_scenes_dir = Path("phase5/data/camo_scenes").resolve()
    
    # 1. Initialize Baseline UniDAC Runner to generate base features
    print("Loading base UniDAC foundation model...")
    cfg = load_sample_config(Path("demo/input/scannetpp_sample.json").resolve())
    unidac_runner = UniDACDepthRunner(
        checkpoint=checkpoints_dir / "unidac.pt",
        sample=cfg
    )

    # 2. Collect Training Scenes
    scene_files = sorted(list(camo_scenes_dir.glob("*.jpg")))
    print(f"Loaded {len(scene_files)} training scenes for camouflage & textureless tuning.")

    # 3. Pre-extract base predictions and tensors
    dataset = []
    temp_p = Path("phase6/temp_train.jpg").resolve()
    temp_p.parent.mkdir(parents=True, exist_ok=True)

    for sf in scene_files:
        img_bgr = cv2.imread(str(sf))
        h, w = img_bgr.shape[:2]
        # Resize to standardized training resolution
        tw, th = 384, int(h * 384 / w)
        # Ensure multiple of 16
        tw = (tw // 16) * 16
        th = (th // 16) * 16
        img_res = cv2.resize(img_bgr, (tw, th))

        cv2.imwrite(str(temp_p), img_res)
        base_d = unidac_runner.predict(temp_p, output_size=(th, tw))
        
        # Prepare Tensors
        rgb_tensor = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        depth_tensor = torch.from_numpy(base_d).unsqueeze(0).unsqueeze(0).float()
        energy_tensor = compute_energy_tensor(img_res)
        
        dataset.append({
            "name": sf.stem,
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "energy": energy_tensor,
            "shape": (th, tw)
        })

    if temp_p.exists():
        temp_p.unlink()

    # 4. Instantiate Adapter
    adapter = TextureAwareResidualHead()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 35

    print(f"\nBeginning Fine-Tuning ({num_epochs} Epochs across {len(dataset)} scenes)...")
    adapter.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for sample in dataset:
            optimizer.zero_grad()
            
            rgb = sample["rgb"]
            base_depth = sample["depth"]
            energy = sample["energy"]
            
            # Predict refined depth
            refined_depth = adapter(base_depth, rgb, energy)
            
            # Loss 1: Maintain metric scale consistency
            l_scale = F.l1_loss(refined_depth, base_depth)
            
            # Loss 2: Sharp Edge Discontinuity Loss
            # Sobel horizontal and vertical gradients
            gx_ref = refined_depth[:, :, :, 1:] - refined_depth[:, :, :, :-1]
            gy_ref = refined_depth[:, :, 1:, :] - refined_depth[:, :, :-1, :]
            
            gx_base = base_depth[:, :, :, 1:] - base_depth[:, :, :, :-1]
            gy_base = base_depth[:, :, 1:, :] - base_depth[:, :, :-1, :]
            
            # Boundary mask (where subtle RGB or depth gradients exist)
            b_mask_x = (energy[:, :, :, 1:] > 0.15).float()
            b_mask_y = (energy[:, :, 1:, :] > 0.15).float()
            
            # Encourage sharper step response at boundary
            l_edge = -0.05 * (torch.mean(torch.abs(gx_ref) * b_mask_x) + torch.mean(torch.abs(gy_ref) * b_mask_y))
            
            # Loss 3: Planar Regularization on Low-Texture Zones
            flat_mask_x = (energy[:, :, :, 1:] < 0.10).float()
            flat_mask_y = (energy[:, :, 1:, :] < 0.10).float()
            l_flat = 0.15 * (torch.mean(torch.abs(gx_ref) * flat_mask_x) + torch.mean(torch.abs(gy_ref) * flat_mask_y))
            
            loss = l_scale + l_edge + l_flat
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{num_epochs}] | Total Loss: {total_loss/len(dataset):.5f} | Scale Loss: {l_scale.item():.5f} | Edge Boost: {l_edge.item():.5f}")

    # 5. Save Fine-Tuned Checkpoint
    save_path = checkpoints_dir / "unidac_camo_adapted.pt"
    torch.save(adapter.state_dict(), str(save_path))
    print(f"\n[SUCCESS] Fine-tuned weights saved -> {save_path}\n")


if __name__ == "__main__":
    train_adapter()
