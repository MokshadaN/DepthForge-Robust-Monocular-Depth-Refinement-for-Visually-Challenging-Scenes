"""
=============================================================================
  PHASE 9 - MULTI-HEAD DILATED ADAPTER FINE-TUNING
  Run entirely in one Kaggle notebook cell, or split at each # CELL comment.
  GPU Runtime: T4 x2 / P100 recommended.
=============================================================================

ARCHITECTURE:
  Phase 8: SINGLE-HEAD  (one unified residual delta for all 3 domains)
  Phase 9: MULTI-HEAD   (shared trunk + 3 domain-expert heads)
           Domain auto-detected by a lightweight 3-class classifier.
           No manual label needed at inference time.

DATASET: 1000 images per domain
  - Camouflage  : CAMO-FS     (Thanhdat3010/CAMO-FS, direct JPEG URLs)
  - Glass       : GSD / GSD-S (garrying/GSD, garrying/GSD-S, HF streaming)
  - Textureless : NYUv2       (tanganke/nyuv2, HF streaming)

TEACHER : DPT_Large (cross-model pseudo-GT, NO self-distillation)
LOSS    : SILog + Gradient Consistency + Domain CE
PARAMS  : ~228 K  (vs 83.4 K in Phase 8)
EPOCHS  : 60 max  (early-stopping patience=12)
=============================================================================
"""

# ===========================================================================
# CELL 1 - Environment and GPU check
# ===========================================================================
import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_install("timm")

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Enable GPU in Kaggle Settings > Accelerator.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device   : {DEVICE}")


# ===========================================================================
# CELL 2 - Imports and directory setup
# ===========================================================================
import os, csv, json, math, random, shutil, zipfile
import urllib.request
import concurrent.futures
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import display as ipy_display

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

ROOT     = Path("/kaggle/working")
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS  = ROOT / "results"
PANELS   = RESULTS / "panels"
VIZ_DIR  = RESULTS / "visualizations"

for _d in [DATA_DIR, CKPT_DIR, RESULTS, PANELS, VIZ_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DOMAIN_NAMES = {0: "camouflage", 1: "glass", 2: "textureless"}
DOMAIN_IDS   = sorted(DOMAIN_NAMES.keys())
METRIC_KEYS  = ["AbsRel", "SqRel", "RMSE", "SILog", "d1", "d2", "d3"]

print("Directories ready.")


# ===========================================================================
# CELL 3A - URL reachability test (probe only, no full download)
# ===========================================================================
# Two separate header dicts:
#   PROBE_HEADERS    - Range: bytes=0-0  -> confirms URL exists (1 byte response)
#   DOWNLOAD_HEADERS - no Range header   -> downloads the full file
PROBE_HEADERS = {
    "User-Agent": "Mozilla/5.0 UniDACPhase9/1.0",
    "Range": "bytes=0-0",
}
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 UniDACPhase9/1.0",
}

_URL_TESTS = [
    ("CAMO-FS image",
     "https://huggingface.co/datasets/Thanhdat3010/CAMO-FS/resolve/main/images/images/camouflage_00000000.jpg"),
    ("GSD Parquet",
     "https://huggingface.co/datasets/garrying/GSD/resolve/main/data/train-00000-of-00001.parquet"),
    ("GSD-S Parquet",
     "https://huggingface.co/datasets/garrying/GSD-S/resolve/main/data/train-00000-of-00002.parquet"),
    ("NYUv2 train Parquet",
     "https://huggingface.co/datasets/tanganke/nyuv2/resolve/main/data/train-00000-of-00008.parquet"),
    ("NYUv2 val Parquet",
     "https://huggingface.co/datasets/tanganke/nyuv2/resolve/main/data/val-00000-of-00006.parquet"),
    ("Trans10K-v2 Parquet",
     "https://huggingface.co/datasets/rdyzakya/Trans10K-v2/resolve/main/data/train-00000-of-00002.parquet"),
]

def _probe(label, url):
    try:
        req = urllib.request.Request(url, headers=PROBE_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read(1)
            status = resp.status
        print(f"  OK  {label:<38} -> HTTP {status}")
        return True
    except Exception as exc:
        print(f"  FAIL {label:<38} -> {exc}")
        return False

print("=== Hugging Face URL Reachability Test ===")
_probe_results = {lbl: _probe(lbl, url) for lbl, url in _URL_TESTS}
print(f"\nPassed: {sum(_probe_results.values())}/{len(_probe_results)}")
print("(Reachable URL != full dataset downloaded.)\n")


# ===========================================================================
# CELL 3B - Dataset download
# ===========================================================================
TARGET          = 1000
MIN_VALID_BYTES = 5_000

CAMO_DIR  = DATA_DIR / "camouflage";  CAMO_DIR.mkdir(exist_ok=True)
GLASS_DIR = DATA_DIR / "glass";       GLASS_DIR.mkdir(exist_ok=True)
TEX_DIR   = DATA_DIR / "textureless"; TEX_DIR.mkdir(exist_ok=True)

def _is_valid(path):
    return path.exists() and path.stat().st_size > MIN_VALID_BYTES

def try_dl(url, dest):
    """Download url -> dest. Skip if dest is already valid (>5 KB)."""
    if _is_valid(dest):
        return True
    try:
        req = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        if len(data) > MIN_VALID_BYTES:
            dest.write_bytes(data)
            return True
        return False
    except Exception:
        return False

def hf_stream_save(hf_path, dest_dir, img_key="image",
                   limit=1000, split="train", label=""):
    """Stream images from a HuggingFace dataset and save as JPEG.
    Returns the count of valid images saved (including previously saved ones)."""
    from datasets import load_dataset
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        ds = load_dataset(hf_path, split=split, streaming=True)
        print(f"  Reading {hf_path} [{split}]...")
        for i, sample in enumerate(ds):
            if count >= limit:
                break
            dest = dest_dir / f"{label}_{i:05d}.jpg"
            if _is_valid(dest):
                count += 1
                continue
            try:
                if img_key not in sample:
                    if i == 0:
                        print("  Available columns:", list(sample.keys()))
                    continue
                arr = np.asarray(sample[img_key])
                # CHW -> HWC
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 2:
                    arr = np.repeat(arr[..., None], 3, axis=2)
                elif arr.ndim != 3:
                    continue
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                if arr.shape[-1] >= 3:
                    arr = arr[..., :3]
                else:
                    continue
                arr = np.nan_to_num(arr)
                if np.issubdtype(arr.dtype, np.floating):
                    if arr.min() >= 0 and arr.max() <= 1.001:
                        arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if cv2.imwrite(str(dest), img_bgr):
                    count += 1
                if count > 0 and count % 200 == 0:
                    print(f"    {label}: {count}/{limit} saved")
            except Exception as e:
                if count < 3:
                    print(f"  Skip sample {i}: {e}")
        print(f"  {label}: {count} images from {hf_path} [{split}]")
        return count
    except Exception as e:
        print(f"  FAILED {hf_path} [{split}]: {e}")
        return 0


# -- Camouflage: direct JPEG downloads from CAMO-FS --------------------------
CAMO_BASE = "https://huggingface.co/datasets/Thanhdat3010/CAMO-FS/resolve/main/images/images"

def _dl_camo(i):
    fn = f"camouflage_{i:08d}.jpg"
    return try_dl(f"{CAMO_BASE}/{fn}", CAMO_DIR / fn)

print(f"\n[1/3] Camouflage - CAMO-FS ({TARGET} images)...")
_t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as _ex:
    _ok_camo = sum(_ex.map(_dl_camo, range(TARGET)))
print(f"  Downloaded: {_ok_camo}/{TARGET}  ({time.time()-_t0:.1f}s)")

# -- Glass: GSD -> GSD-S fallback --------------------------------------------
print(f"\n[2/3] Glass - GSD ({TARGET} images)...")
n_glass = sum(1 for p in GLASS_DIR.glob("*.jpg") if _is_valid(p))

if n_glass < TARGET:
    n_glass += hf_stream_save("garrying/GSD", GLASS_DIR, img_key="image",
                               limit=TARGET - n_glass, split="train", label="gsd_train")
if n_glass < TARGET:
    n_glass += hf_stream_save("garrying/GSD", GLASS_DIR, img_key="image",
                               limit=TARGET - n_glass, split="test", label="gsd_test")
if n_glass < TARGET:
    n_glass += hf_stream_save("garrying/GSD-S", GLASS_DIR, img_key="image",
                               limit=TARGET - n_glass, split="train", label="gsds_train")
print(f"  Glass total: {n_glass}")

# -- Textureless: NYUv2 train + val ------------------------------------------
print(f"\n[3/3] Textureless - NYUv2 ({TARGET} images)...")
n_tex = sum(1 for p in TEX_DIR.glob("*.jpg") if _is_valid(p))

if n_tex < TARGET:
    added = hf_stream_save("tanganke/nyuv2", TEX_DIR, img_key="image",
                            limit=TARGET - n_tex, split="train", label="nyuv2_train")
    n_tex += added
if n_tex < TARGET:
    print(f"  Training insufficient; using val split for {TARGET - n_tex} more...")
    added = hf_stream_save("tanganke/nyuv2", TEX_DIR, img_key="image",
                            limit=TARGET - n_tex, split="val", label="nyuv2_val")
    n_tex += added
print(f"  Textureless total: {n_tex}/{TARGET}")

# -- Final inventory check (validates file sizes, not just file counts) -------
def _count_valid(folder):
    return sum(1 for p in folder.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png") and _is_valid(p))

n_c = _count_valid(CAMO_DIR)
n_g = _count_valid(GLASS_DIR)
n_t = _count_valid(TEX_DIR)
print(f'\n{"="*57}')
print(f"  VALID IMAGES: Camo={n_c} | Glass={n_g} | Tex={n_t} | TOTAL={n_c+n_g+n_t}")
print(f'{"="*57}')
for cnt, name in [(n_c, "Camouflage"), (n_g, "Glass"), (n_t, "Textureless")]:
    flag = "OK" if cnt >= TARGET * 0.8 else ("LOW" if cnt >= 100 else "EMPTY/CORRUPT")
    print(f"  {name:<15}: {cnt:>5}  {flag}")
print()
if n_c < 100 or n_g < 100 or n_t < 100:
    raise RuntimeError(
        f"Dataset validation failed: Camo={n_c}, Glass={n_g}, Tex={n_t}. "
        "All 3 domains must have >=100 valid images (>5 KB)."
    )


# ===========================================================================
# CELL 3C - Visual verification contact sheet (4 thumbnails per domain)
# ===========================================================================
_COLS = 4
_CW, _CH, _HH = 240, 170, 24
_sheet = Image.new("RGB", (_CW * _COLS, (_CH + _HH) * len(DOMAIN_NAMES)), "white")
_draw  = ImageDraw.Draw(_sheet)
_rng_v = random.Random(42)

for _row_idx, (_dn_id, _dn_label) in enumerate(DOMAIN_NAMES.items()):
    _folder  = [CAMO_DIR, GLASS_DIR, TEX_DIR][_dn_id]
    _files   = sorted(p for p in _folder.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png") and _is_valid(p))
    _samples = _rng_v.sample(_files, min(_COLS, len(_files)))
    _y0      = _row_idx * (_CH + _HH)
    _draw.text((4, _y0 + 4), f"{_dn_label.upper()} - {len(_files)} files", fill="black")
    for _col_idx, _path in enumerate(_samples):
        _x = _col_idx * _CW
        _y = _y0 + _HH
        try:
            with Image.open(_path) as _im:
                _im = _im.convert("RGB")
                _im.thumbnail((_CW - 6, _CH - 30))
                _sheet.paste(_im, (_x + (_CW - _im.width) // 2, _y + 4))
                _draw.text((_x + 4, _y + _CH - 20), _path.name[:28], fill="black")
        except Exception as _e:
            _draw.text((_x + 4, _y + 10), f"ERROR: {_e}", fill="red")

ipy_display(_sheet)
print("Visual verification complete.\n")


# ===========================================================================
# CELL 4 - Load MiDaS (student base) + DPT_Large (teacher)
# ===========================================================================
import os as _os
torch.hub.set_dir("/kaggle/working/torch_hub")
_hub_dir      = torch.hub.get_dir()
_trusted_file = _os.path.join(_hub_dir, "trusted_list")
_repos_trust  = {"intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"}
_existing     = set()
if _os.path.exists(_trusted_file):
    with open(_trusted_file) as _f:
        _existing = {l.strip() for l in _f if l.strip()}
with open(_trusted_file, "w") as _f:
    for _repo in sorted(_existing | _repos_trust):
        _f.write(_repo + "\n")
print("Trusted hub repositories configured.")

print("Loading MiDaS_small...")
midas_small = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(DEVICE).eval()
midas_xfms  = torch.hub.load("intel-isl/MiDaS", "transforms",  trust_repo=True)
small_xfm   = midas_xfms.small_transform
print("  MiDaS_small OK")

print("Loading DPT_Large (teacher)...")
dpt_large = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True).to(DEVICE).eval()
dpt_xfm   = midas_xfms.dpt_transform
print("  DPT_Large OK")

IMG_H, IMG_W = 384, 512

@torch.no_grad()
def get_base(img_rgb):
    inp = small_xfm(img_rgb).to(DEVICE)
    out = midas_small(inp)
    out = F.interpolate(out.unsqueeze(1), (IMG_H, IMG_W), mode="bicubic", align_corners=False).squeeze()
    mn, mx = out.min(), out.max()
    return ((out - mn) / (mx - mn + 1e-6)).cpu().numpy()

@torch.no_grad()
def get_teacher(img_rgb):
    inp = dpt_xfm(img_rgb).to(DEVICE)
    out = dpt_large(inp)
    out = F.interpolate(out.unsqueeze(1), (IMG_H, IMG_W), mode="bicubic", align_corners=False).squeeze()
    mn, mx = out.min(), out.max()
    return ((out - mn) / (mx - mn + 1e-6)).cpu().numpy()

def sobel_energy(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag  = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return mag / (mag.max() + 1e-6)

print(f"Models ready. Inference resolution: {IMG_H}x{IMG_W}")


# ===========================================================================
# CELL 5 - Pre-cache all features into CPU RAM (one-time forward passes)
# ===========================================================================
CACHE_LIMIT = 1000

_domain_cfg = [
    (CAMO_DIR,  0, "camouflage"),
    (GLASS_DIR, 1, "glass"),
    (TEX_DIR,   2, "textureless"),
]

all_samples = []

for _ddir, _didx, _dname in _domain_cfg:
    _files = sorted(
        p for p in (list(_ddir.glob("*.jpg")) + list(_ddir.glob("*.png")))
        if _is_valid(p)
    )[:CACHE_LIMIT]

    print(f"\nCaching {len(_files)} {_dname} images...")
    _cached, _skipped = 0, 0

    for _i, _fp in enumerate(_files):
        _bgr = cv2.imread(str(_fp))
        if _bgr is None:
            _skipped += 1
            continue
        _bgr = cv2.resize(_bgr, (IMG_W, IMG_H))
        _rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)

        all_samples.append({
            "rgb":     torch.from_numpy(_bgr).permute(2, 0, 1).float() / 255.,
            "base_d":  torch.from_numpy(get_base(_rgb)).unsqueeze(0).float(),
            "teacher": torch.from_numpy(get_teacher(_rgb)).unsqueeze(0).float(),
            "energy":  torch.from_numpy(sobel_energy(_bgr)).unsqueeze(0).float(),
            "domain":  _didx,
            "name":    _fp.name,
        })
        _cached += 1
        if _cached % 200 == 0 or (_i + 1) == len(_files):
            print(f"  [{_dname}] {_cached}/{len(_files)}")

    if _cached == 0:
        raise RuntimeError(
            f"CRITICAL: 0 valid images cached for domain '{_dname}'! "
            f"Check directory: {_ddir}"
        )
    if _skipped > 0:
        print(f"  Warning: Skipped {_skipped} unreadable files in {_dname}.")

print(f"\nTotal cached: {len(all_samples)}")

# -- Stratified 70/15/15 split per domain ------------------------------------
_by_domain = defaultdict(list)
for _s in all_samples:
    _by_domain[_s["domain"]].append(_s)

train_data, val_data, test_data = [], [], []
for _di, _slist in _by_domain.items():
    random.shuffle(_slist)
    _n    = len(_slist)
    _n_tr = int(_n * 0.70)
    _n_vl = int(_n * 0.15)
    train_data += _slist[:_n_tr]
    val_data   += _slist[_n_tr:_n_tr + _n_vl]
    test_data  += _slist[_n_tr + _n_vl:]

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

def _split_stats(split, name):
    counts = defaultdict(int)
    for s in split:
        counts[s["domain"]] += 1
    parts = "  ".join(f"{DOMAIN_NAMES[d]}={counts[d]}" for d in sorted(counts))
    print(f"  {name:6s}: total={len(split)}  [{parts}]")

print("Stratified split:")
_split_stats(train_data, "Train")
_split_stats(val_data,   "Val")
_split_stats(test_data,  "Test")
print(f"Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")


# ===========================================================================
# CELL 6 - Multi-Head Dilated Adapter architecture
#
#  Input [base_depth(1) | RGB(3) | Sobel(1)] = 5-channel
#         -> Stem (Conv-BN-ReLU-Drop, 64 ch)
#         -> Dilated branches d=1,2,4 -> concat -> trunk (96 ch)
#         -> Domain classifier (global-pool -> 32 -> 3 logits -> softmax alpha)
#               /           |           \
#        head_camo    head_glass    head_tex   (3 expert heads)
#               \           |           /
#  delta = a0*dcamo + a1*dglass + a2*dtex
#  D_refined = clamp(D_base + 0.3*tanh(delta), 0, 1)
# ===========================================================================
class DilatedBranch(nn.Module):
    def __init__(self, in_ch, out_ch, d, p_drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop))
    def forward(self, x): return self.net(x)


class DomainHead(nn.Module):
    def __init__(self, in_ch=96, p_drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop),
            nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))
    def forward(self, x): return self.net(x)


class MultiHeadDilatedAdapter(nn.Module):
    """Phase 9 adapter: shared trunk + 3 domain-expert heads + soft router."""
    def __init__(self, in_ch=5, hidden=64, p_drop=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop))
        self.b1 = DilatedBranch(hidden, hidden // 2, 1, p_drop)
        self.b2 = DilatedBranch(hidden, hidden // 2, 2, p_drop)
        self.b4 = DilatedBranch(hidden, hidden // 2, 4, p_drop)
        trunk_ch = (hidden // 2) * 3   # 96

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p_drop * 2),
            nn.Linear(trunk_ch, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 3))

        self.head_camo  = DomainHead(trunk_ch, p_drop)
        self.head_glass = DomainHead(trunk_ch, p_drop)
        self.head_tex   = DomainHead(trunk_ch, p_drop)
        self.scale = 0.30

    def forward(self, base_d, rgb, energy):
        x      = self.stem(torch.cat([base_d, rgb, energy], dim=1))
        trunk  = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)  # (B,96,H,W)
        logits = self.clf(trunk)          # (B,3)
        alpha  = torch.softmax(logits, dim=1)
        ac     = alpha[:, 0].view(-1,1,1,1)
        ag     = alpha[:, 1].view(-1,1,1,1)
        at     = alpha[:, 2].view(-1,1,1,1)
        delta  = (ac * self.head_camo(trunk) +
                  ag * self.head_glass(trunk) +
                  at * self.head_tex(trunk))
        return torch.clamp(base_d + self.scale * torch.tanh(delta), 0., 1.), logits


adapter = MultiHeadDilatedAdapter().to(DEVICE)
total_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
print(f"Multi-Head Adapter: {total_params:,} params ({total_params/1e3:.1f} K)")

with torch.no_grad():
    _r, _l = adapter(torch.rand(2,1,IMG_H,IMG_W).to(DEVICE),
                     torch.rand(2,3,IMG_H,IMG_W).to(DEVICE),
                     torch.rand(2,1,IMG_H,IMG_W).to(DEVICE))
    print(f"Sanity check -- Output: {_r.shape}  Logits: {_l.shape}  OK")


# ===========================================================================
# CELL 7 - Loss functions
# ===========================================================================
def silog_loss(pred, target, lamda=0.5):
    """Scale-Invariant Log loss."""
    mask = (target > 1e-3) & (pred > 1e-3)
    if mask.sum() < 10:
        return pred.mean() * 0.
    d = torch.log(pred[mask] + 1e-6) - torch.log(target[mask] + 1e-6)
    return d.pow(2).mean() - lamda * d.mean().pow(2)

def grad_loss(pred, target):
    """Gradient consistency loss."""
    gxp = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gyp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gxt = target[:, :, :, 1:] - target[:, :, :, :-1]
    gyt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(gxp, gxt) + F.l1_loss(gyp, gyt)

def total_loss(pred, teacher, logits, dom_lbl):
    return (1.0 * silog_loss(pred, teacher) +
            0.5 * grad_loss(pred, teacher) +
            0.1 * F.cross_entropy(logits, dom_lbl))

print("Losses: SILog + GradConsistency + DomainCE  OK")


# ===========================================================================
# CELL 8 - Evaluation metrics (scale-aligned)
# ===========================================================================
def align(pred, target):
    valid = (target > 1e-3) & (pred > 1e-3)
    p, g  = pred[valid].ravel(), target[valid].ravel()
    if len(p) < 10:
        return pred
    A      = np.stack([p, np.ones_like(p)], axis=1)
    c, *_  = np.linalg.lstsq(A, g, rcond=None)
    return np.clip(c[0] * pred + c[1], 0., 1.)

def compute_metrics(pred_np, gt_np):
    p = align(pred_np, gt_np)
    m = gt_np > 1e-3
    if m.sum() < 10:
        return {}
    p, g = p[m], gt_np[m]
    d    = np.log(p + 1e-6) - np.log(g + 1e-6)
    r    = np.maximum(p / (g + 1e-6), g / (p + 1e-6))
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / (g + 1e-6))),
        "SqRel":  float(np.mean((p - g)**2 / (g + 1e-6))),
        "RMSE":   float(np.sqrt(np.mean((p - g)**2))),
        "SILog":  float(np.sqrt(max(0., np.mean(d**2) - 0.5 * np.mean(d)**2))),
        "d1":     float((r < 1.25).mean()),
        "d2":     float((r < 1.25**2).mean()),
        "d3":     float((r < 1.25**3).mean()),
    }

print("Metrics: AbsRel, SqRel, RMSE, SILog, d1/d2/d3  OK")


# ===========================================================================
# CELL 9 - Training loop (60 epochs max, batch=8, AdamW, cosine LR,
#           early-stopping patience=12, data augmentation)
# ===========================================================================
NUM_EPOCHS = 60
BATCH_SIZE = 8
PATIENCE   = 12
BEST_CKPT  = CKPT_DIR / "phase9_best.pt"
LAST_CKPT  = CKPT_DIR / "phase9_last.pt"

optimizer = torch.optim.AdamW(adapter.parameters(), lr=1.5e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-6)

def _augment(s):
    rgb = s["rgb"].clone()
    bd  = s["base_d"].clone()
    tch = s["teacher"].clone()
    en  = s["energy"].clone()
    dl  = s["domain"]
    if random.random() < 0.5:
        rgb = torch.flip(rgb, [-1])
        bd  = torch.flip(bd,  [-1])
        tch = torch.flip(tch, [-1])
        en  = torch.flip(en,  [-1])
    sc  = random.uniform(0.90, 1.10)
    bd  = torch.clamp(bd  * sc, 0., 1.)
    tch = torch.clamp(tch * sc, 0., 1.)
    return rgb, bd, tch, en, dl

def _collate(samples):
    rgbs, bds, tchs, ens, dls = [], [], [], [], []
    for s in samples:
        r, b, t, e, d = _augment(s)
        rgbs.append(r); bds.append(b); tchs.append(t); ens.append(e); dls.append(d)
    return (torch.stack(rgbs).to(DEVICE), torch.stack(bds).to(DEVICE),
            torch.stack(tchs).to(DEVICE), torch.stack(ens).to(DEVICE),
            torch.tensor(dls, dtype=torch.long).to(DEVICE))

train_hist, val_hist, absrel_hist, silog_hist = [], [], [], []
best_val, no_improve = float("inf"), 0

print("=" * 72)
print("  PHASE 9 -- MULTI-HEAD ADAPTER  REGULARIZED TRAINING")
print(f"  Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")
print(f"  BatchSize={BATCH_SIZE} | WeightDecay=5e-4 | Patience={PATIENCE}")
print("=" * 72)

for _ep in range(1, NUM_EPOCHS + 1):
    # Train
    adapter.train()
    random.shuffle(train_data)
    _batches = [train_data[i:i+BATCH_SIZE] for i in range(0, len(train_data), BATCH_SIZE)]
    _tl = 0.
    for _batch in _batches:
        _rgb, _bd, _tch, _en, _dl = _collate(_batch)
        optimizer.zero_grad()
        _ref, _logits = adapter(_bd, _rgb, _en)
        _loss = total_loss(_ref, _tch, _logits, _dl)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        _tl += _loss.item()
    scheduler.step()
    _avg_tl = _tl / len(_batches)
    train_hist.append(_avg_tl)

    # Validate
    adapter.eval()
    _vl, _va, _vs = 0., 0., 0.
    with torch.no_grad():
        _vbatches = [val_data[i:i+BATCH_SIZE] for i in range(0, len(val_data), BATCH_SIZE)]
        for _vb in _vbatches:
            _rgb = torch.stack([s["rgb"]     for s in _vb]).to(DEVICE)
            _bd  = torch.stack([s["base_d"]  for s in _vb]).to(DEVICE)
            _tch = torch.stack([s["teacher"] for s in _vb]).to(DEVICE)
            _en  = torch.stack([s["energy"]  for s in _vb]).to(DEVICE)
            _ref, _logits = adapter(_bd, _rgb, _en)
            _vl += silog_loss(_ref, _tch).item()
            for _i in range(len(_vb)):
                _m = compute_metrics(_ref[_i].squeeze().cpu().numpy(),
                                     _tch[_i].squeeze().cpu().numpy())
                if _m:
                    _va += _m["AbsRel"]
                    _vs += _m["SILog"]
    _avg_vl = _vl / len(_vbatches)
    _avg_va = _va / len(val_data)
    _avg_vs = _vs / len(val_data)
    val_hist.append(_avg_vl)
    absrel_hist.append(_avg_va)
    silog_hist.append(_avg_vs)

    if _avg_vl < best_val:
        best_val, no_improve = _avg_vl, 0
        torch.save(adapter.state_dict(), str(BEST_CKPT))
    else:
        no_improve += 1

    if _ep % 5 == 0 or _ep == 1:
        print(f"  [{_ep:03d}/{NUM_EPOCHS}] "
              f"Train={_avg_tl:.5f}  ValSILog={_avg_vl:.5f}  "
              f"AbsRel={_avg_va:.4f}  SILog={_avg_vs:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}  "
              f"NoImprove={no_improve}/{PATIENCE}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {_ep} (no improvement for {PATIENCE} epochs)")
        break

torch.save(adapter.state_dict(), str(LAST_CKPT))
print(f"\nDone. Best Val SILog={best_val:.5f}  ->  {BEST_CKPT}")


# ===========================================================================
# CELL 10 - Load best checkpoint and run comprehensive test evaluation
# ===========================================================================
print("\nLoading best checkpoint for evaluation...")
try:
    adapter.load_state_dict(
        torch.load(str(BEST_CKPT), map_location=DEVICE, weights_only=True))
except TypeError:
    adapter.load_state_dict(torch.load(str(BEST_CKPT), map_location=DEVICE))
adapter.eval()

MAX_VIZ_PER_DOMAIN = 3

per_image_rows = []
routing_true   = []
routing_pred   = []
routing_conf   = []
viz_examples   = {d: [] for d in DOMAIN_IDS}

print(f"\nEvaluating {len(test_data)} test images...")
with torch.no_grad():
    for _idx, _s in enumerate(test_data):
        _dom  = int(_s["domain"])
        _rgb  = _s["rgb"].unsqueeze(0).to(DEVICE)
        _bd   = _s["base_d"].unsqueeze(0).to(DEVICE)
        _en   = _s["energy"].unsqueeze(0).to(DEVICE)
        _tch  = _s["teacher"].squeeze().cpu().numpy()

        _ref, _logits = adapter(_bd, _rgb, _en)
        _base_np = _bd.squeeze().cpu().numpy()
        _ref_np  = _ref.squeeze().cpu().numpy()

        _bm = compute_metrics(_base_np, _tch)
        _rm = compute_metrics(_ref_np,  _tch)

        _row = {"image": _s.get("name", f"test_{_idx:04d}"),
                "domain_id": _dom,
                "domain":    DOMAIN_NAMES[_dom]}
        for _k in METRIC_KEYS:
            _row[f"baseline_{_k}"] = _bm.get(_k, float("nan"))
            _row[f"refined_{_k}"]  = _rm.get(_k, float("nan"))
        per_image_rows.append(_row)

        _probs    = torch.softmax(_logits, dim=1).squeeze(0)
        _pred_dom = int(_probs.argmax().item())
        _conf     = float(_probs.max().item())
        routing_true.append(_dom)
        routing_pred.append(_pred_dom)
        routing_conf.append(_conf)

        # Store examples for visualisation -- IMPORTANT: flip BGR->RGB for display
        if len(viz_examples[_dom]) < MAX_VIZ_PER_DOMAIN:
            _bgr_np = _s["rgb"].permute(1, 2, 0).cpu().numpy()   # stored as BGR
            viz_examples[_dom].append({
                "name":        _s.get("name", f"test_{_idx:04d}"),
                "rgb":         np.clip(_bgr_np[..., ::-1], 0., 1.),  # flip to RGB
                "energy":      _s["energy"].squeeze().cpu().numpy(),
                "base":        _base_np.copy(),
                "refined":     _ref_np.copy(),
                "teacher":     _tch.copy(),
                "pred_domain": _pred_dom,
                "conf":        _conf,
            })

        if (_idx + 1) % 50 == 0 or (_idx + 1) == len(test_data):
            print(f"  {_idx+1}/{len(test_data)} evaluated")

print(f"\nTest evaluation complete: {len(per_image_rows)} samples")

# -- Aggregation helpers -----------------------------------------------------
def _finite(vals):
    return [float(v) for v in vals if v is not None and np.isfinite(v)]

def _safe_mean(vals):
    v = _finite(vals)
    return float(np.mean(v)) if v else float("nan")

def _domain_metric(dom_id, model, key):
    return _safe_mean([r[f"{model}_{key}"] for r in per_image_rows
                       if r["domain_id"] == dom_id])

summary_rows = []
for _di in DOMAIN_IDS:
    _cnt = sum(r["domain_id"] == _di for r in per_image_rows)
    for _model in ("baseline", "refined"):
        _sr = {"domain_id": _di, "domain": DOMAIN_NAMES[_di],
               "model": _model, "n_images": _cnt}
        for _k in METRIC_KEYS:
            _sr[_k] = _domain_metric(_di, _model, _k)
        summary_rows.append(_sr)

# -- Console summary table ---------------------------------------------------
print("\n" + "=" * 72)
print("  PHASE 9 TEST EVALUATION")
print("=" * 72)
print(f"{'Domain':<15}{'Model':<22}{'AbsRel':>9}{'SILog':>9}{'d1':>8}")
print("-" * 72)
_all_b, _all_r = [], []
for _di in DOMAIN_IDS:
    _dn   = DOMAIN_NAMES[_di]
    _babs = _domain_metric(_di, "baseline", "AbsRel")
    _rabs = _domain_metric(_di, "refined",  "AbsRel")
    _bsil = _domain_metric(_di, "baseline", "SILog")
    _rsil = _domain_metric(_di, "refined",  "SILog")
    _bd1  = _domain_metric(_di, "baseline", "d1")
    _rd1  = _domain_metric(_di, "refined",  "d1")
    if not np.isfinite(_babs):
        continue
    _gain = (_babs - _rabs) / _babs * 100
    _all_b.append(_babs); _all_r.append(_rabs)
    print(f"{_dn:<15}{'Baseline (MiDaS)':<22}{_babs:>9.4f}{_bsil:>9.4f}{_bd1:>8.4f}")
    print(f"{'':15}{'Phase9 Multi-Head':<22}{_rabs:>9.4f}{_rsil:>9.4f}{_rd1:>8.4f}  <- +{_gain:.1f}%")
    print("-" * 72)
if _all_b:
    _ov = (np.mean(_all_b) - np.mean(_all_r)) / np.mean(_all_b) * 100
    print(f"\nOverall: Baseline={np.mean(_all_b):.4f}  Phase9={np.mean(_all_r):.4f}  Gain=+{_ov:.1f}%")
print("=" * 72)


# ===========================================================================
# CELL 11 - Domain routing accuracy
# ===========================================================================
_cm = np.zeros((3, 3), dtype=int)
_d2i = {d: i for i, d in enumerate(DOMAIN_IDS)}
for _t, _p in zip(routing_true, routing_pred):
    if _t in _d2i and _p in _d2i:
        _cm[_d2i[_t], _d2i[_p]] += 1
_total_routed = int(_cm.sum())
print(f"\nDomain Routing Accuracy: {np.trace(_cm)}/{_total_routed} = {np.trace(_cm)/_total_routed*100:.1f}%")
for _i, _di in enumerate(DOMAIN_IDS):
    _n = _cm[_i].sum()
    if _n:
        print(f"  {DOMAIN_NAMES[_di]:<15}: {_cm[_i,_i]}/{_n} = {_cm[_i,_i]/_n*100:.1f}%")


# ===========================================================================
# CELL 12 - Phase 8 vs Phase 9 architecture comparison
# ===========================================================================
print("\n" + "=" * 72)
print("  PHASE 8 (Single-Head) vs PHASE 9 (Multi-Head)")
print("=" * 72)
_cmp_rows = [
    ("Architecture",    "Single-Head",       "Multi-Head (3 heads)"),
    ("Parameters",      "83,400",            f"{total_params:,}"),
    ("Heads",           "1 shared output",   "3 domain-expert heads"),
    ("Domain routing",  "None (implicit)",   "Soft classifier (auto)"),
    ("Dataset",         "~522 images",       f"~{len(all_samples)} images"),
    ("Training epochs", "40",                str(NUM_EPOCHS)),
    ("Teacher",         "DPT_Large",         "DPT_Large"),
    ("Loss",            "SILog + Grad",      "SILog + Grad + DomCE"),
    ("Phase8 AbsRel",   "4.2002 (reported)", "see CELL 10 results"),
]
print(f"{'Aspect':<28}{'Phase 8':>22}{'Phase 9':>22}")
print("-" * 72)
for _r in _cmp_rows:
    print(f"{_r[0]:<28}{_r[1]:>22}{_r[2]:>22}")
print("=" * 72)


# ===========================================================================
# CELL 13 - Export CSV reports
# ===========================================================================
_per_image_csv = RESULTS / "phase9_per_image_metrics.csv"
_summary_csv   = RESULTS / "phase9_summary_metrics.csv"

_per_fields = (["image", "domain_id", "domain"] +
               [f"{m}_{k}" for m in ("baseline","refined") for k in METRIC_KEYS])
with open(_per_image_csv, "w", newline="", encoding="utf-8") as _f:
    _w = csv.DictWriter(_f, fieldnames=_per_fields)
    _w.writeheader(); _w.writerows(per_image_rows)

_sum_fields = ["domain_id", "domain", "model", "n_images"] + METRIC_KEYS
with open(_summary_csv, "w", newline="", encoding="utf-8") as _f:
    _w = csv.DictWriter(_f, fieldnames=_sum_fields)
    _w.writeheader(); _w.writerows(summary_rows)

print(f"\nCSV reports saved:")
print(f"  Per-image : {_per_image_csv}")
print(f"  Summary   : {_summary_csv}")


# ===========================================================================
# CELL 14 - All 10 visualizations
# ===========================================================================
_actual_eps = len(train_hist)
_ep_ax      = np.arange(1, _actual_eps + 1)
_best_ep    = int(np.nanargmin(val_hist)) + 1 if val_hist else 1
_dom_labels = [DOMAIN_NAMES[d] for d in DOMAIN_IDS]
_x_lbl      = np.arange(len(DOMAIN_IDS))

def _save_fig(fig, fname, dpi=200):
    _p = VIZ_DIR / fname
    fig.savefig(_p, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {_p.name}")


# 1. Training and validation curves ------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Phase 9 -- Training & Validation Dynamics",
             fontsize=16, fontweight="bold")
for _ax, _hist, _title, _ylabel in [
    (axes[0,0], train_hist,  "Training Total Objective",        "Loss (SILog+Grad+CE)"),
    (axes[0,1], val_hist,    "Validation SILog Loss",           "Val SILog loss"),
    (axes[1,0], absrel_hist, "Validation AbsRel (lower better)","AbsRel"),
    (axes[1,1], silog_hist,  "Validation SILog Metric (lower)", "SILog metric"),
]:
    if _hist:
        _ax.plot(_ep_ax[:len(_hist)], _hist, marker=".", lw=2.0)
        _ax.axvline(_best_ep, ls="--", lw=1.2, alpha=0.6,
                    label=f"Best @ ep {_best_ep}")
        _ax.legend(fontsize=8)
    _ax.set_title(_title); _ax.set_xlabel("Epoch"); _ax.set_ylabel(_ylabel)
    _ax.grid(alpha=0.3)
fig.text(0.5, 0.005,
         "Training objective and val SILog loss use different formulations.",
         ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
_save_fig(fig, "01_training_curves.png")


# 2. Dataset split distribution -----------------------------------------------
_w_bar = 0.26
fig, ax = plt.subplots(figsize=(11, 6))
for _j, (_sn, _sd) in enumerate([("Train", train_data), ("Val", val_data), ("Test", test_data)]):
    _cnts = [sum(int(s["domain"]) == d for s in _sd) for d in DOMAIN_IDS]
    ax.bar(_x_lbl + (_j - 1) * _w_bar, _cnts, _w_bar, label=_sn)
ax.set_xticks(_x_lbl); ax.set_xticklabels(_dom_labels)
ax.set_ylabel("Number of images"); ax.set_title("Dataset Split Distribution per Domain")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
_save_fig(fig, "02_split_distribution.png")


# 3. Per-domain error metrics (AbsRel, SqRel, RMSE, SILog) -------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Baseline MiDaS vs Phase 9 -- Per-Domain Error Metrics",
             fontsize=15, fontweight="bold")
_bw = 0.36
for _ax, _mk in zip(axes.flat, ["AbsRel", "SqRel", "RMSE", "SILog"]):
    _bv = [_domain_metric(d, "baseline", _mk) for d in DOMAIN_IDS]
    _rv = [_domain_metric(d, "refined",  _mk) for d in DOMAIN_IDS]
    _ax.bar(_x_lbl - _bw/2, _bv, _bw, label="Baseline MiDaS", color="#3498db")
    _ax.bar(_x_lbl + _bw/2, _rv, _bw, label="Phase 9 Refined", color="#e74c3c")
    _ax.set_xticks(_x_lbl); _ax.set_xticklabels(_dom_labels)
    _ax.set_title(f"{_mk} (lower is better)"); _ax.set_ylabel(_mk)
    _ax.legend(); _ax.grid(axis="y", alpha=0.25)
fig.tight_layout(rect=[0,0,1,0.95])
_save_fig(fig, "03_error_metrics.png")


# 4. Delta-threshold accuracy (d1, d2) ----------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
_bw2     = 0.14
_offsets = np.array([-1.5, -0.5, 0.5, 1.5])
for _off, (_m, _mk, _lbl) in zip(
    _offsets,
    [("baseline","d1","Base d1"),("refined","d1","Refined d1"),
     ("baseline","d2","Base d2"),("refined","d2","Refined d2")]
):
    _vals = [_domain_metric(d, _m, _mk) for d in DOMAIN_IDS]
    ax.bar(_x_lbl + _off * _bw2, _vals, _bw2, label=_lbl)
ax.set_xticks(_x_lbl); ax.set_xticklabels(_dom_labels)
ax.set_ylabel("Threshold accuracy (higher is better)")
ax.set_title("Per-Domain d1 and d2 Accuracy")
ax.legend(ncol=2); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
_save_fig(fig, "04_delta_accuracy.png")


# 5. Relative improvement bar chart -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Relative Improvement over Baseline (Positive = Better)",
             fontsize=14, fontweight="bold")
for _ax, _mk in zip(axes, ["AbsRel", "SILog"]):
    _pcts   = []
    _colors = []
    for _d in DOMAIN_IDS:
        _bv = _domain_metric(_d, "baseline", _mk)
        _rv = _domain_metric(_d, "refined",  _mk)
        if np.isfinite(_bv) and np.isfinite(_rv) and abs(_bv) > 1e-12:
            _p = (_bv - _rv) / abs(_bv) * 100
        else:
            _p = float("nan")
        _pcts.append(_p)
        _colors.append("#2ecc71" if (not np.isnan(_p) and _p > 0) else "#e74c3c")
    _ax.bar(_dom_labels, _pcts, color=_colors)
    _ax.axhline(0, ls="--", lw=1)
    _ax.set_title(f"{_mk} Reduction vs Baseline (%)")
    _ax.set_ylabel("% reduction"); _ax.grid(axis="y", alpha=0.25)
    _ax.tick_params(axis="x", rotation=10)
fig.tight_layout(rect=[0,0,1,0.92])
_save_fig(fig, "05_relative_improvement.png")


# 6. Per-image AbsRel box plots -----------------------------------------------
_box_d, _box_lbl = [], []
for _d in DOMAIN_IDS:
    for _model in ("baseline", "refined"):
        _vals = _finite([r[f"{_model}_AbsRel"] for r in per_image_rows
                         if r["domain_id"] == _d])
        if _vals:
            _box_d.append(_vals)
            _box_lbl.append(f"{DOMAIN_NAMES[_d]}\n{_model.capitalize()}")
fig, ax = plt.subplots(figsize=(15, 6))
if _box_d:
    ax.boxplot(_box_d, labels=_box_lbl, showfliers=False)
    ax.set_ylabel("AbsRel"); ax.grid(axis="y", alpha=0.25)
    ax.set_title("Per-Image AbsRel Distribution by Domain and Model")
fig.tight_layout()
_save_fig(fig, "06_absrel_boxplots.png")


# 7. Domain-routing confusion matrix ------------------------------------------
_row_tot = _cm.sum(axis=1, keepdims=True)
_cm_pct  = np.divide(_cm, _row_tot,
                      out=np.zeros_like(_cm, dtype=float),
                      where=_row_tot != 0)
fig, ax = plt.subplots(figsize=(8, 7))
_im = ax.imshow(_cm_pct, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3))
ax.set_xticklabels(_dom_labels, rotation=20, ha="right")
ax.set_yticklabels(_dom_labels)
ax.set_xlabel("Predicted Domain"); ax.set_ylabel("True Domain")
ax.set_title("Domain-Routing Confusion Matrix (Row-normalised)")
for _i in range(3):
    for _j in range(3):
        ax.text(_j, _i, f"{_cm[_i,_j]}\n{_cm_pct[_i,_j]:.1%}",
                ha="center", va="center",
                color="white" if _cm_pct[_i,_j] > 0.55 else "black", fontsize=9)
fig.colorbar(_im, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
fig.tight_layout()
_save_fig(fig, "07_routing_confusion.png")


# 8. Routing confidence histogram ---------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(routing_conf, bins=25, edgecolor="black", alpha=0.8, color="#8e44ad")
ax.set_xlabel("Max predicted routing probability")
ax.set_ylabel("Number of test images")
ax.set_title("Domain-Routing Confidence Distribution")
_mean_conf = float(np.mean(routing_conf)) if routing_conf else 0.
ax.axvline(_mean_conf, ls="--", lw=1.5, label=f"Mean = {_mean_conf:.3f}")
ax.legend(); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
_save_fig(fig, "08_routing_confidence.png")


# 9. Prediction-vs-teacher scatter (pixel level) ------------------------------
_rng_sc = np.random.default_rng(42)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for _ax, _d in zip(axes, DOMAIN_IDS):
    _exs = viz_examples.get(_d, [])
    if not _exs:
        _ax.set_title(f"{DOMAIN_NAMES[_d]} -- no examples"); _ax.axis("off"); continue
    for _j, _ex in enumerate(_exs):
        _tgt = _ex["teacher"].ravel()
        _n   = min(3000, len(_tgt))
        _idx_s = _rng_sc.choice(len(_tgt), size=_n, replace=False)
        _ax.scatter(_tgt[_idx_s], _ex["base"][_idx_s],
                    s=3, alpha=0.10, color="#3498db",
                    label="Baseline" if _j == 0 else None)
        _ax.scatter(_tgt[_idx_s], _ex["refined"][_idx_s],
                    s=3, alpha=0.10, color="#e74c3c",
                    label="Refined" if _j == 0 else None)
    _ax.plot([0,1],[0,1], "k--", lw=1.2, label="y=x")
    _ax.set_xlim(0,1); _ax.set_ylim(0,1)
    _ax.set_xlabel("DPT-Large (pseudo-GT)"); _ax.set_ylabel("Model output")
    _ax.set_title(DOMAIN_NAMES[_d])
    _ax.legend(markerscale=4, fontsize=8); _ax.grid(alpha=0.2)
fig.suptitle("Pixelwise Agreement with DPT-Large Pseudo-Targets",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93])
_save_fig(fig, "09_scatter_vs_teacher.png")


# 10. 7-column visual depth panels per domain ---------------------------------
_panel_hdrs = ["RGB Input", "Sobel Energy", "Baseline MiDaS",
               "Phase 9 Refined", "DPT Pseudo-GT",
               "Abs Difference", "Signed Difference"]

def _depth_cmap(arr, cm_name):
    mn, mx = np.percentile(arr, 2), np.percentile(arr, 98)
    norm   = np.clip((arr - mn) / (mx - mn + 1e-5), 0, 1)
    return plt.get_cmap(cm_name)(norm)[..., :3]

for _d in DOMAIN_IDS:
    _exs = viz_examples.get(_d, [])
    if not _exs:
        print(f"  No examples stored for domain {DOMAIN_NAMES[_d]}, skipping panel.")
        continue
    _nr = len(_exs)
    _fig, _axes = plt.subplots(_nr, 7, figsize=(30, 4.8 * _nr), squeeze=False)
    for _r, _ex in enumerate(_exs):
        _ab  = np.abs(_ex["refined"] - _ex["teacher"])
        _sg  = _ex["refined"] - _ex["teacher"]
        _vmax_ab = max(float(np.percentile(_ab, 98)), 1e-6)
        _items = [
            ("rgb",  _ex["rgb"],     None),
            ("map",  _ex["energy"],  "bone"),
            ("map",  _ex["base"],    "inferno"),
            ("map",  _ex["refined"], "inferno"),
            ("map",  _ex["teacher"], "inferno"),
            ("abs",  _ab,            "magma"),
            ("diff", _sg,            "coolwarm"),
        ]
        for _c, (_kind, _arr, _cm_name) in enumerate(_items):
            _ax = _axes[_r, _c]
            if   _kind == "rgb":  _ax.imshow(np.clip(_arr, 0, 1))
            elif _kind == "diff": _ax.imshow(_arr, cmap=_cm_name, vmin=-1, vmax=1)
            elif _kind == "abs":  _ax.imshow(_arr, cmap=_cm_name, vmin=0, vmax=_vmax_ab)
            elif _c == 1:         _ax.imshow(_arr, cmap=_cm_name)
            else:                 _ax.imshow(_arr, cmap=_cm_name, vmin=0, vmax=1)
            if _r == 0:
                _ax.set_title(_panel_hdrs[_c], fontsize=9, fontweight="bold")
            _ax.axis("off")
        _axes[_r, 0].set_ylabel(
            f"{_ex['name'][:18]}\nPred:{DOMAIN_NAMES.get(_ex['pred_domain'], '?')} "
            f"({_ex['conf']:.2f})",
            rotation=90, fontsize=7, labelpad=8)
    _fig.suptitle(
        f"Phase 9 Results -- {DOMAIN_NAMES[_d].upper()}\n"
        "Blue signed diff = under-estimated, Red = over-estimated",
        fontsize=13, fontweight="bold")
    _fig.tight_layout(rect=[0, 0, 1, 0.93])
    _pp = PANELS / f"phase9_panel_{DOMAIN_NAMES[_d]}.png"
    _fig.savefig(_pp, dpi=160, bbox_inches="tight")
    plt.show(); plt.close(_fig)
    print(f"  Saved panel: {_pp.name}")


# ===========================================================================
# CELL 15 - Full text summary table
# ===========================================================================
print("\n" + "=" * 104)
print("PHASE 9 -- COMPLETE TEST METRIC SUMMARY  (vs DPT-Large pseudo-targets)")
print("=" * 104)
_hdr = (f"{'Domain':<14} {'Model':<11} {'N':>5} "
        f"{'AbsRel':>9} {'SqRel':>9} {'RMSE':>9} "
        f"{'SILog':>9} {'d1':>8} {'d2':>8} {'d3':>8}")
print(_hdr)
print("-" * len(_hdr))
for _sr in summary_rows:
    print(f"{_sr['domain']:<14} {_sr['model']:<11} {_sr['n_images']:>5} "
          f"{_sr['AbsRel']:>9.4f} {_sr['SqRel']:>9.4f} {_sr['RMSE']:>9.4f} "
          f"{_sr['SILog']:>9.4f} {_sr['d1']:>8.4f} {_sr['d2']:>8.4f} {_sr['d3']:>8.4f}")
print("=" * 104)


# ===========================================================================
# CELL 16 - Package all outputs into one ZIP for download
# ===========================================================================
_zip_path = ROOT / "phase9_complete_results.zip"

with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED) as _zf:
    for _folder, _arc in [(CKPT_DIR, "checkpoints"),
                           (RESULTS,  "results"),
                           (PANELS,   "panels")]:
        if _folder.exists():
            for _fp in _folder.rglob("*"):
                if _fp.is_file() and _fp.resolve() != _zip_path.resolve():
                    _zf.write(_fp, str(Path(_arc) / _fp.relative_to(_folder)))

print("\n" + "=" * 72)
print("FINAL OUTPUTS")
print("=" * 72)
print(f"Visualizations dir  : {VIZ_DIR}")
print(f"Per-image CSV       : {_per_image_csv}")
print(f"Summary CSV         : {_summary_csv}")
print(f"Qualitative panels  : {PANELS}")
print(f"Complete ZIP        : {_zip_path}")
print(f"ZIP size            : {_zip_path.stat().st_size / 1e6:.2f} MB")
print("\nDownload phase9_complete_results.zip from the Kaggle Output / Files tab.")
