"""
=============================================================================
  PHASE 9 — MULTI-HEAD DILATED ADAPTER FINE-TUNING
  Copy each CELL block into a separate Kaggle notebook cell.
  GPU Runtime: T4 / P100 recommended.
=============================================================================

ARCHITECTURE ANSWER:
  Phase 8 → SINGLE-HEAD  (one unified residual delta for all 3 domains)
  Phase 9 → MULTI-HEAD   (shared trunk + 3 domain-expert heads)
             Domain auto-detected by a lightweight 3-class classifier
             no manual label needed at inference time.

DATASET: 1000+ images per domain
  - Camouflage   : CAMO-FS (HuggingFace: Thanhdat3010/CAMO-FS)
  - Glass        : Trans10K (HuggingFace: kadirnar/Trans10k)
  - Textureless  : ScanNet (HuggingFace: jxu124/scannet)

TEACHER: DPT_Large (cross-model pseudo-GT, NO self-distillation)
LOSS:    SILog + Gradient Consistency + Domain CE
PARAMS:  ~200K  (vs 83.4K in Phase 8)
EPOCHS:  60     (vs 40 in Phase 8)
=============================================================================
"""

# ===========================================================================
# CELL 1 — Environment & GPU
# ===========================================================================
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install('timm')

import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU      : {torch.cuda.get_device_name(0)}')
    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device   : {DEVICE}')


# ===========================================================================
# CELL 2 — Imports & Directories
# ===========================================================================
import os, json, random, shutil, zipfile
import urllib.request
import concurrent.futures
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42); np.random.seed(42); random.seed(42)

ROOT      = Path('/kaggle/working')
DATA_DIR  = ROOT / 'data'
CKPT_DIR  = ROOT / 'checkpoints'
RESULTS   = ROOT / 'results'
PANELS    = RESULTS / 'panels'

for d in [DATA_DIR, CKPT_DIR, RESULTS, PANELS]:
    d.mkdir(parents=True, exist_ok=True)

DOMAIN_NAMES = {0: 'camouflage', 1: 'glass', 2: 'textureless'}
print('Directories ready.')



# ===========================================================================
# CELL 3A — Corrected URL Reachability Test
# Range requests prevent large files from being downloaded for testing.
# ===========================================================================

import urllib.request
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 UniDACPhase9/1.0",
    "Range": "bytes=0-0",
}

URL_TESTS = [
    (
        "CAMO-FS image",
        "https://huggingface.co/datasets/Thanhdat3010/CAMO-FS/resolve/main/images/images/camouflage_00000000.jpg",
    ),
    (
        "GSD Parquet",
        "https://huggingface.co/datasets/garrying/GSD/resolve/main/data/train-00000-of-00001.parquet",
    ),
    (
        "GSD-S Parquet (corrected)",
        "https://huggingface.co/datasets/garrying/GSD-S/resolve/main/data/train-00000-of-00002.parquet",
    ),
    (
        "NYUv2 train Parquet",
        "https://huggingface.co/datasets/tanganke/nyuv2/resolve/main/data/train-00000-of-00008.parquet",
    ),
    (
        "NYUv2 validation Parquet",
        "https://huggingface.co/datasets/tanganke/nyuv2/resolve/main/data/val-00000-of-00006.parquet",
    ),
    (
        "ScanNet archive (reachability only)",
        "https://huggingface.co/datasets/maksimko123/scannet/resolve/main/1.zip",
    ),
    (
        "Trans10K-v2 Parquet",
        "https://huggingface.co/datasets/rdyzakya/Trans10K-v2/resolve/main/data/train-00000-of-00002.parquet",
    ),
]


def probe(label, url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)

        with urllib.request.urlopen(req, timeout=30) as response:
            response.read(1)
            status = response.status

        print(f"  ✅ {label:<38} → HTTP {status}")
        return True

    except Exception as exc:
        print(f"  ❌ {label:<38} → {exc}")
        return False


print("=== Corrected Hugging Face URL Test ===")

results = {}
for label, url in URL_TESTS:
    results[label] = probe(label, url)

print("\n=== Summary ===")
print(f"Passed: {sum(results.values())}/{len(results)}")
print("Note: A reachable archive or Parquet URL does not mean the")
print("entire file has been downloaded or the dataset has been decoded.")


# ===========================================================================
# CELL 3B — Dataset Download (NO synthetic images — real datasets only)
# ===========================================================================
TARGET    = 1000
CAMO_DIR  = DATA_DIR / 'camouflage';  CAMO_DIR.mkdir(exist_ok=True)
GLASS_DIR = DATA_DIR / 'glass';       GLASS_DIR.mkdir(exist_ok=True)
TEX_DIR   = DATA_DIR / 'textureless'; TEX_DIR.mkdir(exist_ok=True)

def try_dl(url, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            dest.write_bytes(r.read())
        return True
    except:
        return False


def hf_stream_save(
    hf_path,
    dest_dir,
    img_key="image",
    limit=1000,
    split="train",
    label="",
):
    """
    Stream images from a standard Hugging Face dataset.
    Supports both PIL images (HWC) and NumPy arrays (CHW/HWC).
    Returns the number of NEW images saved.
    """
    from datasets import load_dataset

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    inspected = 0

    try:
        ds = load_dataset(
            hf_path,
            split=split,
            streaming=True,
        )

        print(f"  Reading {hf_path} [{split}]...")

        for i, sample in enumerate(ds):
            if count >= limit:
                break

            inspected += 1
            dest = dest_dir / f"{label}_{i:05d}.jpg"

            # A previous run may already have saved this exact image.
            if dest.exists() and dest.stat().st_size > 1000:
                continue

            try:
                if img_key not in sample:
                    if inspected == 1:
                        print("  Available columns:", list(sample.keys()))
                    continue

                arr = np.asarray(sample[img_key])

                # Convert CHW to HWC, e.g. (3, 288, 384) -> (288, 384, 3).
                if (
                    arr.ndim == 3
                    and arr.shape[0] in (1, 3, 4)
                    and arr.shape[-1] not in (1, 3, 4)
                ):
                    arr = np.transpose(arr, (1, 2, 0))

                # Convert grayscale to RGB.
                if arr.ndim == 2:
                    arr = np.repeat(arr[..., None], 3, axis=2)
                elif arr.ndim != 3:
                    continue

                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=2)

                # Discard alpha if present.
                if arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                else:
                    continue

                # Convert floating-point images to uint8 correctly.
                arr = np.nan_to_num(arr)

                if np.issubdtype(arr.dtype, np.floating):
                    if arr.min() >= 0 and arr.max() <= 1.001:
                        arr = arr * 255.0

                arr = np.clip(arr, 0, 255).astype(np.uint8)

                # Convert RGB to BGR for OpenCV.
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                if cv2.imwrite(str(dest), img_bgr):
                    count += 1

                if count > 0 and count % 200 == 0:
                    print(f"    {label}: {count}/{limit} newly saved")

            except Exception as sample_error:
                if count < 3:
                    print(f"  Skipping sample {i}: {sample_error}")

        print(f"  {label}: {count} new images saved from {hf_path} [{split}]")
        return count

    except Exception as e:
        print(f"  {label} FAILED ({hf_path}, split={split}): {e}")
        return 0

# ────────────────────────────────────────────────────────────────────────────
# CAMOUFLAGE: CAMO-FS (direct URL, already confirmed working)
# ────────────────────────────────────────────────────────────────────────────
CAMO_BASE = 'https://huggingface.co/datasets/Thanhdat3010/CAMO-FS/resolve/main/images/images'
def dl_camo(i):
    fn = f'camouflage_{i:08d}.jpg'
    return try_dl(f'{CAMO_BASE}/{fn}', CAMO_DIR / fn)

print(f'\n[1/3] Camouflage — CAMO-FS ({TARGET} images)...')
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
    ok_camo = sum(ex.map(dl_camo, range(TARGET)))
print(f'  Result: {ok_camo}/{TARGET}  ({time.time()-t0:.1f}s)')

# ────────────────────────────────────────────────────────────────────────────
# GLASS: garrying/GSD (CVPR 2022 Glass Surface Detection, 3523 images)
#   → Fallback 1: garrying/GSD-S (4519 images with semantic labels)
#   → Fallback 2: Trans10K direct URL
# ────────────────────────────────────────────────────────────────────────────
print(f'\n[2/3] Glass — GSD dataset ({TARGET} images)...')
n_glass = len(list(GLASS_DIR.glob('*.jpg')))

if n_glass < TARGET:
    n_glass += hf_stream_save('garrying/GSD', GLASS_DIR, img_key='image',
                              limit=TARGET - n_glass, split='train', label='gsd_train')
if n_glass < TARGET:
    print(f'  Trying GSD test split for {TARGET-n_glass} more...')
    n_glass += hf_stream_save('garrying/GSD', GLASS_DIR, img_key='image',
                              limit=TARGET - n_glass, split='test', label='gsd_test')
if n_glass < TARGET:
    print(f'  Trying GSD-S fallback for {TARGET-n_glass} more...')
    n_glass += hf_stream_save('garrying/GSD-S', GLASS_DIR, img_key='image',
                              limit=TARGET - n_glass, split='train', label='gsds')
if n_glass < TARGET:
    print(f'  Trying Trans10K direct URL for {TARGET-n_glass} more...')
    GLASS_BASE = 'https://huggingface.co/datasets/kadirnar/Trans10k/resolve/main/images'
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(
            lambda i: try_dl(f'{GLASS_BASE}/glass_{i:05d}.jpg', GLASS_DIR / f'trans_{i:05d}.jpg'),
            range(1, TARGET - n_glass + 1)))
    n_glass += sum(results)

print(f'  Glass total: {n_glass}')


# ---------------------------------------------------------------------------
# TEXTURELESS PROXY: NYUv2 indoor RGB images
# Source: tanganke/nyuv2 (standard Parquet dataset)
# Splits: train (795) + val (654)
# ---------------------------------------------------------------------------

print(f"\n[3/3] Textureless proxy — NYUv2 ({TARGET} images)...")

n_tex = len(list(TEX_DIR.glob("*.jpg")))

if n_tex < TARGET:
    remaining = TARGET - n_tex

    print(f"  Downloading training images: up to {remaining}...")

    added = hf_stream_save(
        "tanganke/nyuv2",
        TEX_DIR,
        img_key="image",
        limit=remaining,
        split="train",
        label="nyuv2_train",
    )
    n_tex += added

if n_tex < TARGET:
    remaining = TARGET - n_tex

    print(f"  Training split insufficient; trying validation: {remaining}...")

    added = hf_stream_save(
        "tanganke/nyuv2",
        TEX_DIR,
        img_key="image",
        limit=remaining,
        split="val",
        label="nyuv2_val",
    )
    n_tex += added

print(f"  Textureless proxy total: {n_tex}/{TARGET}")

# ────────────────────────────────────────────────────────────────────────────
# Final inventory check — STOP if any domain is empty
# ────────────────────────────────────────────────────────────────────────────
n_c = len(list(CAMO_DIR.glob('*.jpg')) + list(CAMO_DIR.glob('*.png')))
n_g = len(list(GLASS_DIR.glob('*.jpg')) + list(GLASS_DIR.glob('*.png')))
n_t = len(list(TEX_DIR.glob('*.jpg')) + list(TEX_DIR.glob('*.png')))
print(f'\n{"="*55}')
print(f'  FINAL: Camo={n_c} | Glass={n_g} | Tex={n_t} | TOTAL={n_c+n_g+n_t}')
print(f'{"="*55}')

for domain, count, name in [(n_c, TARGET, 'Camouflage'), (n_g, TARGET, 'Glass'), (n_t, TARGET, 'Textureless')]:
    status = '✅' if domain >= count * 0.8 else ('⚠️ LOW' if domain >= 100 else '❌ EMPTY')
    print(f'  {name:<15}: {domain:>5}  {status}')
print()
if n_g < 100 or n_t < 100:
    raise RuntimeError('❌ Not enough real images. Share the CELL 3A tester output for diagnosis.')


# ===========================================================================
# CELL 4 — Load MiDaS (base) + DPT_Large (teacher)
# ===========================================================================
print('Loading MiDaS_small...')
midas_small = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True).to(DEVICE).eval()
midas_xfms  = torch.hub.load('intel-isl/MiDaS', 'transforms',  trust_repo=True)
small_xfm   = midas_xfms.small_transform
print('  MiDaS_small OK')

print('Loading DPT_Large (teacher)...')
dpt_large = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True).to(DEVICE).eval()
dpt_xfm   = midas_xfms.dpt_transform
print('  DPT_Large OK')

IMG_H, IMG_W = 384, 512   # fixed inference resolution

@torch.no_grad()
def get_base(img_rgb):
    inp = small_xfm(img_rgb).to(DEVICE)
    out = midas_small(inp)
    out = F.interpolate(out.unsqueeze(1), (IMG_H, IMG_W), mode='bicubic', align_corners=False).squeeze()
    mn, mx = out.min(), out.max()
    return ((out - mn) / (mx - mn + 1e-6)).cpu().numpy()

@torch.no_grad()
def get_teacher(img_rgb):
    inp = dpt_xfm(img_rgb).to(DEVICE)
    out = dpt_large(inp)
    out = F.interpolate(out.unsqueeze(1), (IMG_H, IMG_W), mode='bicubic', align_corners=False).squeeze()
    mn, mx = out.min(), out.max()
    return ((out - mn) / (mx - mn + 1e-6)).cpu().numpy()

def sobel(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag  = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)**2 +
                   cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)**2)
    return mag / (mag.max() + 1e-6)

print(f'Models ready. Inference: {IMG_H}x{IMG_W}')


# ===========================================================================
# CELL 5 — Pre-cache all features (one-time forward pass → CPU RAM)
# ===========================================================================
LIMIT = 1000   # images per domain

domain_cfg = [
    (CAMO_DIR,  0, 'camouflage'),
    (GLASS_DIR, 1, 'glass'),
    (TEX_DIR,   2, 'textureless'),
]

all_samples = []
for ddir, didx, dname in domain_cfg:
    files = sorted(list(ddir.glob('*.jpg')) + list(ddir.glob('*.png')))[:LIMIT]
    print(f'\nCaching {len(files)} {dname} images...')
    for i, fp in enumerate(files):
        img_bgr = cv2.imread(str(fp))
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        base_np    = get_base(img_rgb)
        teacher_np = get_teacher(img_rgb)
        sobel_np   = sobel(img_bgr)
        all_samples.append({
            'rgb':     torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.,
            'base_d':  torch.from_numpy(base_np).unsqueeze(0).float(),
            'teacher': torch.from_numpy(teacher_np).unsqueeze(0).float(),
            'energy':  torch.from_numpy(sobel_np).unsqueeze(0).float(),
            'domain':  didx,
            'name':    fp.name,
        })
        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            print(f'  [{dname}] {i+1}/{len(files)}')

print(f'\nTotal cached: {len(all_samples)}')

# ── Stratified split (70/15/15 per domain) ─────────────────────────────────
# Fixes: random shuffle caused camouflage to land almost entirely in train,
# producing 0 camouflage test samples and inflating per-domain routing stats.
from collections import defaultdict
by_domain = defaultdict(list)
for s in all_samples:
    by_domain[s['domain']].append(s)

train_data, val_data, test_data = [], [], []
for di, slist in by_domain.items():
    random.shuffle(slist)
    n = len(slist)
    n_tr = int(n * 0.70)
    n_vl = int(n * 0.15)
    train_data += slist[:n_tr]
    val_data   += slist[n_tr:n_tr + n_vl]
    test_data  += slist[n_tr + n_vl:]

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

# Diagnostic: verify domain balance in each split
def split_stats(split, name):
    counts = defaultdict(int)
    for s in split: counts[s['domain']] += 1
    parts = '  '.join(f'{DOMAIN_NAMES[d]}={counts[d]}' for d in sorted(counts))
    print(f'  {name:6s}: total={len(split)}  [{parts}]')

print('Stratified split:')
split_stats(train_data, 'Train')
split_stats(val_data,   'Val')
split_stats(test_data,  'Test')
print(f'Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}')


# ===========================================================================
# CELL 6 — Multi-Head Dilated Adapter Architecture
#
#  Architecture diagram:
#
#  Input [base_depth(1) | RGB(3) | Sobel(1)] = 5-channel
#         ↓
#  Stem (Conv-BN-ReLU, 64ch)
#         ↓
#  Parallel dilated branches: d=1, d=2, d=4  → concat → trunk (96ch)
#         ↓
#  Domain Classifier (global-pool → 32 → 3 logits → softmax α)
#         ↙             ↓             ↘
#  head_camo       head_glass       head_tex       ← 3 expert heads
#  (Conv×3, 48→32→1)
#         ↘             ↓             ↙
#  Δ = α₀·Δcamo + α₁·Δglass + α₂·Δtex     ← soft weighted mix
#         ↓
#  D_refined = clamp(D_base + 0.3·tanh(Δ), 0, 1)
# ===========================================================================
class DilatedBranch(nn.Module):
    def __init__(self, in_ch, out_ch, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout))          # ← spatial dropout
    def forward(self, x): return self.net(x)


class DomainHead(nn.Module):
    """Per-domain expert refinement head with dropout for regularization."""
    def __init__(self, in_ch=96, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(48, 32, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))
    def forward(self, x): return self.net(x)


class MultiHeadDilatedAdapter(nn.Module):
    """
    Phase 9: Multi-Head Dilated Residual Adapter.
    ~200K parameters. With dropout regularization.
    At inference: domain auto-inferred via soft classifier — no label needed.
    """
    def __init__(self, in_ch=5, hidden=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout))
        self.b1 = DilatedBranch(hidden, hidden // 2, 1, dropout)
        self.b2 = DilatedBranch(hidden, hidden // 2, 2, dropout)
        self.b4 = DilatedBranch(hidden, hidden // 2, 4, dropout)
        trunk_ch = (hidden // 2) * 3   # 96

        # Lightweight domain classifier
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout * 2),
            nn.Linear(trunk_ch, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 3))

        # 3 domain expert heads
        self.head_camo  = DomainHead(trunk_ch, dropout)
        self.head_glass = DomainHead(trunk_ch, dropout)
        self.head_tex   = DomainHead(trunk_ch, dropout)
        self.scale = 0.30

    def forward(self, base_d, rgb, energy):
        x     = self.stem(torch.cat([base_d, rgb, energy], dim=1))
        trunk = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)  # (B,96,H,W)

        logits = self.clf(trunk)                   # (B,3)
        alpha  = torch.softmax(logits, dim=1)

        ac = alpha[:, 0].view(-1, 1, 1, 1)
        ag = alpha[:, 1].view(-1, 1, 1, 1)
        at = alpha[:, 2].view(-1, 1, 1, 1)

        delta  = (ac * self.head_camo(trunk) +
                  ag * self.head_glass(trunk) +
                  at * self.head_tex(trunk))

        return torch.clamp(base_d + self.scale * torch.tanh(delta), 0., 1.), logits


adapter = MultiHeadDilatedAdapter().to(DEVICE)
total_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
print(f'Multi-Head Adapter: {total_params:,} params ({total_params/1e3:.1f}K)')

# Sanity check
with torch.no_grad():
    _r, _l = adapter(torch.rand(2,1,IMG_H,IMG_W).to(DEVICE),
                     torch.rand(2,3,IMG_H,IMG_W).to(DEVICE),
                     torch.rand(2,1,IMG_H,IMG_W).to(DEVICE))
    print(f'Output: {_r.shape}  Logits: {_l.shape}  ✓')


# ===========================================================================
# CELL 7 — Loss Functions
# ===========================================================================
def silog_loss(pred, target, lamda=0.5):
    """Scale-Invariant Log loss (mathematically sound, no self-distillation)."""
    mask = (target > 1e-3) & (pred > 1e-3)
    if mask.sum() < 10:
        return pred.mean() * 0
    d = torch.log(pred[mask] + 1e-6) - torch.log(target[mask] + 1e-6)
    return d.pow(2).mean() - lamda * d.mean().pow(2)

def grad_loss(pred, target):
    """Gradient consistency — preserves edge sharpness."""
    gxp = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gyp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gxt = target[:, :, :, 1:] - target[:, :, :, :-1]
    gyt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(gxp, gxt) + F.l1_loss(gyp, gyt)

def total_loss(pred, teacher, logits, dom_lbl):
    return (1.0 * silog_loss(pred, teacher) +
            0.5 * grad_loss(pred, teacher) +
            0.1 * F.cross_entropy(logits, dom_lbl))

print('Losses: SILog + GradConsistency + DomainCE  ✓')


# ===========================================================================
# CELL 8 — Evaluation Metrics
# ===========================================================================
def align(pred, target):
    valid = (target > 1e-3) & (pred > 1e-3)
    p, g  = pred[valid].ravel(), target[valid].ravel()
    if len(p) < 10: return pred
    A = np.stack([p, np.ones_like(p)], 1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return np.clip(c[0] * pred + c[1], 0, 1)

def compute_metrics(pred_np, gt_np):
    p = align(pred_np, gt_np)
    m = gt_np > 1e-3
    if m.sum() < 10: return {}
    p, g = p[m], gt_np[m]
    d    = np.log(p + 1e-6) - np.log(g + 1e-6)
    r    = np.maximum(p / (g + 1e-6), g / (p + 1e-6))
    return {
        'AbsRel': np.mean(np.abs(p - g) / (g + 1e-6)),
        'SqRel':  np.mean((p - g)**2 / (g + 1e-6)),
        'RMSE':   np.sqrt(np.mean((p - g)**2)),
        'SILog':  np.sqrt(max(0, np.mean(d**2) - 0.5 * np.mean(d)**2)),
        'd1': (r < 1.25).mean(),
        'd2': (r < 1.25**2).mean(),
        'd3': (r < 1.25**3).mean(),
    }

print('Metrics: AbsRel, SqRel, RMSE, SILog, δ1/δ2/δ3  ✓')


# ===========================================================================
# CELL 9 — Training Loop (60 epochs, AdamW, Cosine LR)
#   Improvements v2:
#     • Mini-batch training (BS=8) — 6-8x faster than sample-by-sample
#     • Higher weight_decay (5e-4) — stronger L2 regularization
#     • Augmentation: random hflip + depth/brightness jitter
#     • Val metric = SILog-only (not affected by domain CE imbalance)
#     • Early stopping with patience=12
# ===========================================================================
import math

NUM_EPOCHS = 60
BATCH_SIZE = 8
PATIENCE   = 12      # early stopping patience
BEST_CKPT  = CKPT_DIR / 'phase9_multihead_best.pt'
LAST_CKPT  = CKPT_DIR / 'phase9_multihead_last.pt'

# Higher weight_decay for stronger regularization
optimizer  = torch.optim.AdamW(adapter.parameters(), lr=1.5e-3, weight_decay=5e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-6)

# ── Augmentation helpers ──────────────────────────────────────────────────
def augment_sample(s):
    """Apply random horizontal flip and depth scale jitter."""
    rgb   = s['rgb'].clone()
    bd    = s['base_d'].clone()
    tch   = s['teacher'].clone()
    en    = s['energy'].clone()
    dl    = s['domain']
    # Random horizontal flip
    if random.random() < 0.5:
        rgb = torch.flip(rgb, [-1])
        bd  = torch.flip(bd,  [-1])
        tch = torch.flip(tch, [-1])
        en  = torch.flip(en,  [-1])
    # Depth scale jitter ±10% (prevents scale memorization)
    scale = random.uniform(0.90, 1.10)
    bd  = torch.clamp(bd  * scale, 0., 1.)
    tch = torch.clamp(tch * scale, 0., 1.)
    return rgb, bd, tch, en, dl


def collate_batch(samples):
    """Collate a list of samples into batched tensors with augmentation."""
    rgbs, bds, tchs, ens, dls = [], [], [], [], []
    for s in samples:
        rgb, bd, tch, en, dl = augment_sample(s)
        rgbs.append(rgb); bds.append(bd); tchs.append(tch)
        ens.append(en);   dls.append(dl)
    return (torch.stack(rgbs).to(DEVICE),
            torch.stack(bds).to(DEVICE),
            torch.stack(tchs).to(DEVICE),
            torch.stack(ens).to(DEVICE),
            torch.tensor(dls, dtype=torch.long).to(DEVICE))


train_hist, val_hist, absrel_hist, silog_hist = [], [], [], []
best_val     = float('inf')
no_improve   = 0

print('=' * 72)
print('  PHASE 9 v2: MULTI-HEAD ADAPTER — REGULARIZED TRAINING')
print(f'  Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}')
print(f'  BatchSize={BATCH_SIZE} | WeightDecay=5e-4 | Patience={PATIENCE}')
print('=' * 72)

for ep in range(1, NUM_EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────────────────
    adapter.train()
    tl = 0.
    random.shuffle(train_data)
    batches = [train_data[i:i+BATCH_SIZE] for i in range(0, len(train_data), BATCH_SIZE)]
    for batch in batches:
        rgb, bd, tch, en, dl = collate_batch(batch)
        optimizer.zero_grad()
        ref, logits = adapter(bd, rgb, en)
        loss = total_loss(ref, tch, logits, dl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()
    scheduler.step()
    avg_tl = tl / len(batches)
    train_hist.append(avg_tl)

    # ── Validate (SILog-only for stable tracking) ──────────────────────────
    adapter.eval()
    vl = 0.; va = 0.; vs = 0.
    with torch.no_grad():
        val_batches = [val_data[i:i+BATCH_SIZE] for i in range(0, len(val_data), BATCH_SIZE)]
        for batch in val_batches:
            rgb = torch.stack([s['rgb'] for s in batch]).to(DEVICE)
            bd  = torch.stack([s['base_d'] for s in batch]).to(DEVICE)
            tch = torch.stack([s['teacher'] for s in batch]).to(DEVICE)
            en  = torch.stack([s['energy'] for s in batch]).to(DEVICE)
            dl  = torch.tensor([s['domain'] for s in batch], dtype=torch.long).to(DEVICE)
            ref, logits = adapter(bd, rgb, en)
            # Val loss = SILog only (no DomainCE, prevents label-imbalance inflation)
            vl += silog_loss(ref, tch).item()
            for i in range(len(batch)):
                m = compute_metrics(ref[i].squeeze().cpu().numpy(),
                                    tch[i].squeeze().cpu().numpy())
                if m:
                    va += m['AbsRel']
                    vs += m['SILog']
    n_val_batches = len(val_batches)
    n_val = len(val_data)
    avg_vl = vl / n_val_batches
    avg_va = va / n_val
    avg_vs = vs / n_val
    val_hist.append(avg_vl)
    absrel_hist.append(avg_va)
    silog_hist.append(avg_vs)

    # ── Checkpoint & Early Stopping ────────────────────────────────────────
    if avg_vl < best_val:
        best_val   = avg_vl
        no_improve = 0
        torch.save(adapter.state_dict(), str(BEST_CKPT))
    else:
        no_improve += 1

    if ep % 5 == 0 or ep == 1:
        print(f'  [{ep:03d}/{NUM_EPOCHS}] Train={avg_tl:.5f}  ValSILog={avg_vl:.5f}  '
              f'AbsRel={avg_va:.4f}  SILog={avg_vs:.4f}  LR={scheduler.get_last_lr()[0]:.2e}  '
              f'NoImprove={no_improve}/{PATIENCE}')

    if no_improve >= PATIENCE:
        print(f'\n⚠️  Early stopping at epoch {ep} (no improvement for {PATIENCE} epochs)')
        break

torch.save(adapter.state_dict(), str(LAST_CKPT))
print(f'\nDone. Best Val SILog={best_val:.5f} → {BEST_CKPT}')


# ===========================================================================
# CELL 10 — Training Curves (enhanced: 3 subplots)
# ===========================================================================
actual_epochs = len(train_hist)
ep_ax = np.arange(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Phase 9 Multi-Head Adapter v2 — Training Dynamics', fontsize=14, fontweight='bold')

# Plot 1: Train vs Val SILog loss
ax1 = axes[0]
ax1.plot(ep_ax, train_hist, '#e74c3c', lw=2.5, label='Train Loss')
ax1.plot(ep_ax, val_hist,   '#2980b9', lw=2.5, ls='--', label='Val SILog Loss')
best_ep = int(np.argmin(val_hist)) + 1
ax1.axvline(best_ep, color='gray', ls=':', lw=1.5, label=f'Best @ ep {best_ep}')
ax1.set_xlabel('Epoch'); ax1.set_title('SILog Loss (Train vs Val)')
ax1.legend(); ax1.grid(alpha=.3)

# Plot 2: AbsRel
ax2 = axes[1]
ax2.plot(ep_ax, absrel_hist, '#27ae60', lw=2.5, label='Val AbsRel ↓')
ax2.axvline(best_ep, color='gray', ls=':', lw=1.5)
ax2.set_xlabel('Epoch'); ax2.set_title('Validation AbsRel (↓ better)')
ax2.legend(); ax2.grid(alpha=.3)

# Plot 3: SILog metric (eval)
ax3 = axes[2]
ax3.plot(ep_ax, silog_hist, '#8e44ad', lw=2.5, label='Val SILog Metric ↓')
ax3.axvline(best_ep, color='gray', ls=':', lw=1.5)
ax3.set_xlabel('Epoch'); ax3.set_title('Validation SILog Metric (↓ better)')
ax3.legend(); ax3.grid(alpha=.3)

plt.tight_layout()
out_p = RESULTS / 'phase9_training_curves.png'
plt.savefig(str(out_p), dpi=200, bbox_inches='tight')
plt.show()
print(f'Saved → {out_p}')


# ===========================================================================
# CELL 11 — Test Evaluation (per domain, all metrics)
# ===========================================================================
adapter.load_state_dict(torch.load(str(BEST_CKPT), map_location=DEVICE, weights_only=True))
adapter.eval()

res_ref  = {0: [], 1: [], 2: []}
res_base = {0: [], 1: [], 2: []}

with torch.no_grad():
    for s in test_data:
        rgb = s['rgb'].unsqueeze(0).to(DEVICE)
        bd  = s['base_d'].unsqueeze(0).to(DEVICE)
        tch = s['teacher'].unsqueeze(0).to(DEVICE)
        en  = s['energy'].unsqueeze(0).to(DEVICE)
        dom = s['domain']
        ref, _ = adapter(bd, rgb, en)
        mr = compute_metrics(ref.squeeze().cpu().numpy(), tch.squeeze().cpu().numpy())
        mb = compute_metrics(bd.squeeze().cpu().numpy(),  tch.squeeze().cpu().numpy())
        if mr: res_ref[dom].append(mr)
        if mb: res_base[dom].append(mb)

print('\n' + '=' * 72)
print('  PHASE 9 TEST EVALUATION')
print('=' * 72)
print(f'{"Domain":<15}{"Model":<22}{"AbsRel":>9}{"SILog":>9}{"δ1":>8}')
print('-' * 72)

all_r, all_b = [], []
for di, dn in DOMAIN_NAMES.items():
    rr, rb = res_ref[di], res_base[di]
    if not rr: continue
    mr = {k: np.mean([x[k] for x in rr]) for k in rr[0]}
    mb = {k: np.mean([x[k] for x in rb]) for k in rb[0]}
    gain = (mb['AbsRel'] - mr['AbsRel']) / mb['AbsRel'] * 100
    all_r.append(mr['AbsRel']); all_b.append(mb['AbsRel'])
    print(f'{dn:<15}{"Baseline (MiDaS)":<22}{mb["AbsRel"]:>9.4f}{mb["SILog"]:>9.4f}{mb["d1"]:>8.4f}')
    print(f'{"":<15}{"Phase9 Multi-Head":<22}{mr["AbsRel"]:>9.4f}{mr["SILog"]:>9.4f}{mr["d1"]:>8.4f}  ← +{gain:.1f}%')
    print('-' * 72)

ov_gain = (np.mean(all_b) - np.mean(all_r)) / np.mean(all_b) * 100
print(f'\nOverall: Baseline={np.mean(all_b):.4f}  Phase9={np.mean(all_r):.4f}  Gain=+{ov_gain:.1f}%')
print('=' * 72)


# ===========================================================================
# CELL 12 — 6-Column Visual Diagnostic Panels
#   RGB | Sobel | Base | Refined | DPT GT | |Error|
# ===========================================================================
N = 4
domain_pool = {0: [], 1: [], 2: []}
for s in test_data:
    d = s['domain']
    if len(domain_pool[d]) < N:
        domain_pool[d].append(s)

def cmap(arr, cm):
    mn, mx = np.percentile(arr, 2), np.percentile(arr, 98)
    norm = np.clip((arr - mn) / (mx - mn + 1e-5), 0, 1)
    return plt.get_cmap(cm)(norm)[..., :3]

def make_panel(samples, title, out_path):
    n = len(samples)
    fig, axes = plt.subplots(n, 6, figsize=(26, 4.2 * n))
    if n == 1: axes = axes[np.newaxis, :]
    hdrs = ['RGB Input', 'Sobel Energy', 'Base MiDaS',
            'Phase 9 Refined', 'DPT_Large GT', '|Refined − GT|']
    for row, s in enumerate(samples):
        rgb = s['rgb'].unsqueeze(0).to(DEVICE)
        bd  = s['base_d'].unsqueeze(0).to(DEVICE)
        en  = s['energy'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ref, _ = adapter(bd, rgb, en)
        imgs = [
            s['rgb'].permute(1, 2, 0).numpy(),
            s['energy'].squeeze().numpy(),
            cmap(s['base_d'].squeeze().numpy(), 'inferno'),
            cmap(ref.squeeze().cpu().numpy(), 'inferno'),
            cmap(s['teacher'].squeeze().numpy(), 'inferno'),
            cmap(np.abs(ref.squeeze().cpu().numpy() - s['teacher'].squeeze().numpy()), 'plasma'),
        ]
        for col in range(6):
            ax = axes[row, col]
            if col == 1:
                ax.imshow(imgs[col], cmap='bone')
            else:
                ax.imshow(np.clip(imgs[col], 0, 1))
            if row == 0:
                ax.set_title(hdrs[col], fontsize=9, fontweight='bold')
            ax.axis('off')
    plt.suptitle(f'Phase 9 Multi-Head — {title}', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved → {out_path.name}')

labels = {0: 'Biological Camouflage', 1: 'Transparent Glass & Refraction', 2: 'Textureless Architecture'}
for di, slist in domain_pool.items():
    if slist:
        make_panel(slist, labels[di], PANELS / f'panel_phase9_{DOMAIN_NAMES[di]}.png')


# ===========================================================================
# CELL 13 — Domain Routing Accuracy
# ===========================================================================
adapter.eval()
correct = 0; total = 0
dc = {0:0,1:0,2:0}; dt = {0:0,1:0,2:0}
with torch.no_grad():
    for s in test_data:
        rgb = s['rgb'].unsqueeze(0).to(DEVICE)
        bd  = s['base_d'].unsqueeze(0).to(DEVICE)
        en  = s['energy'].unsqueeze(0).to(DEVICE)
        gt  = s['domain']
        _, logits = adapter(bd, rgb, en)
        pred = logits.argmax(1).item()
        if pred == gt: correct += 1; dc[gt] += 1
        dt[gt] += 1; total += 1

print(f'\nDomain Routing Accuracy: {correct}/{total} = {100*correct/total:.1f}%')
for di, dn in DOMAIN_NAMES.items():
    if dt[di] > 0:
        print(f'  {dn:<15}: {dc[di]}/{dt[di]} = {100*dc[di]/dt[di]:.1f}%')


# ===========================================================================
# CELL 14 — Phase 8 vs Phase 9 Comparison
# ===========================================================================
print('\n' + '=' * 72)
print('  PHASE 8 (Single-Head) vs PHASE 9 (Multi-Head)')
print('=' * 72)
rows = [
    ('Architecture',     'Single-Head',       'Multi-Head (3 heads)'),
    ('Parameters',       '83,400',            f'{total_params:,}'),
    ('Heads',            '1 shared output',   '3 domain-expert heads'),
    ('Domain routing',   'None (implicit)',   'Soft classifier (auto)'),
    ('Dataset',          '~522 images',       f'~{len(all_samples)} images'),
    ('Training epochs',  '40',                str(NUM_EPOCHS)),
    ('Teacher',          'DPT_Large',         'DPT_Large'),
    ('Loss',             'SILog + Grad',      'SILog + Grad + DomCE'),
    ('Phase8 AbsRel',    '4.2002 (reported)', 'see CELL 11 results'),
]
print(f'{"Aspect":<28}{"Phase 8":>22}{"Phase 9":>22}')
print('-' * 72)
for r in rows:
    print(f'{r[0]:<28}{r[1]:>22}{r[2]:>22}')
print('=' * 72)


# ===========================================================================
# CELL 15 — Package Everything for Download
# ===========================================================================
zip_path = ROOT / 'phase9_results.zip'
with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in CKPT_DIR.glob('*.pt'):
        zf.write(str(f), f'checkpoints/{f.name}')
    for f in PANELS.glob('*.png'):
        zf.write(str(f), f'panels/{f.name}')
    for f in RESULTS.glob('*.png'):
        zf.write(str(f), f'results/{f.name}')

print(f'\n✅ Packaged → {zip_path}')
print(f'   Size: {zip_path.stat().st_size / 1e6:.1f} MB')
print('   Download: Kaggle → Output tab ↗')
