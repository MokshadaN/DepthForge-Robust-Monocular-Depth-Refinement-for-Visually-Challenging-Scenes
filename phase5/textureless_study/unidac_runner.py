"""Adapter around the official UniDAC demo for PERSPECTIVE images.
Tested on kitti360 / scannetpp demo samples (CPU).

Key facts from the repo code:
  * the model runs on an ERP (equirectangular) patch, not on the original image;
  * its depth is EUCLIDEAN range (the demo converts GT z-depth -> range for NYU/KITTI);
  * output is mapped back to camera space with erp_patch_to_cam_fast.
So we return z-depth in the ORIGINAL image grid (euclid_to_z) so it is comparable
to other models.

Changes from original sandbox version:
  - valid mask passed as HxWx1 float32 (cam_to_erp_patch_fast requires 3-D)
  - dep_e already comes out as HxW from cam_to_erp_patch_fast, expanded to HxWx1
    before feeding into resize_for_input (which expects [..., 1] depth)
  - erp_h cast to float before erp_patch_to_cam_fast
  - fwd_sz=(H,W) order consistent with resize_for_input signature
"""
import json, sys, os, numpy as np, torch, torchvision.transforms.functional as TF


def euclid_to_z(rng: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Convert Euclidean range map to z-depth using pinhole projection."""
    h, w = rng.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return rng / np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)


class UniDACRunner:
    def __init__(
        self,
        repo: str = "UniDAC",
        ckpt: str = "UniDAC/checkpoints/unidac.pt",
        config: str = "configs/test/dac_dinov3l+dpt_indoor_test_nyu.json",
        device: str = "cuda",
    ):
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from unidac.models.unidac import UniDAC
        from unidac.utils.erp_geometry import cam_to_erp_patch_fast, erp_patch_to_cam_fast
        from unidac.dataloaders.dataset import resize_for_input
        self._c2e     = cam_to_erp_patch_fast
        self._e2c     = erp_patch_to_cam_fast
        self._resize  = resize_for_input
        cfg = json.load(open(os.path.join(repo, config)))
        self.cano = cfg["data"]["cano_sz"]           # e.g. [1400, 1400]
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = UniDAC.build(cfg)
        self.model.load_pretrained(ckpt)
        self.model = self.model.to(self.dev).eval()

    def __call__(self, rgb_u8: np.ndarray, K: tuple = None):
        if K is not None:
            fx, fy, cx, cy = K
        else:
            H, W = rgb_u8.shape[:2]
            fx = fy = 518.85
            cx, cy = W / 2.0, H / 2.0
        return self.predict_z(rgb_u8, fx, fy, cx, cy)

    @torch.no_grad()
    def predict_z(
        self,
        rgb_u8: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        crop_wFoV: int = 180,
        fwd_sz: tuple = (512, 704),   # (height, width) — model input size
        name: str = "perspective",
    ):
        """Return (z_depth, active_mask) in the original image grid.

        z_depth   : float32 HxW  – z-depth in metres (0 where inactive)
        active_mask: bool   HxW  – True where the ERP patch covers the pixel

        IMPORTANT: ``name`` is the value placed in cam_params['dataset'] and is
        used by erp_patch_to_cam_fast to select the camera-model branch:
          - 'kitti360', 'scannetpp', 'zipnerf'  → fisheye lookup-table path (needs grid)
          - anything else with cx/cy/fx/fy keys  → standard pinhole path (no grid needed)
        For ordinary perspective cameras, leave ``name`` as 'perspective' (the default).
        """
        H, W = rgb_u8.shape[:2]
        cam = dict(dataset=name, fx=fx, fy=fy, cx=cx, cy=cy)

        # Inputs to cam_to_erp_patch_fast
        img   = rgb_u8.astype(np.float32) / 255.0          # HxWx3
        dep   = np.zeros((H, W, 1), np.float32)            # HxWx1
        valid = np.ones( (H, W, 1), np.float32)            # HxWx1 ← FIXED (was 2-D)

        cw = int(self.cano[0] * crop_wFoV / 180)
        ch = int(cw * fwd_sz[0] / fwd_sz[1])

        img_e, dep_e, _, erp_mask, lat, lon = self._c2e(
            img, dep, valid,
            0, np.float32(0),
            ch, cw,
            self.cano[0], self.cano[0] * 2,
            cam, np.float32(0),
            scale_fac=None,
        )
        # dep_e is HxW from cam_to_erp_patch_fast; expand to HxWx1 for resize_for_input
        if dep_e.ndim == 2:
            dep_e = dep_e[:, :, np.newaxis]

        lat_r = torch.tensor([float(lat.min()), float(lat.max())])
        lon_r = torch.tensor([float(lon.min()), float(lon.max())])

        # resize_for_input(image, depth, output_shape, intrinsic,
        #                  canonical_shape, to_canonical_ratio, ...)
        img_r, dep_r, pad, psf, attn, lat_g, lon_g = self._resize(
            (img_e * 255.0).astype(np.uint8),
            dep_e,
            fwd_sz,           # (H_out, W_out)
            None,             # intrinsic (unused when mask provided)
            [img_e.shape[0], img_e.shape[1]],  # canonical_shape
            1.0,              # to_canonical_ratio
            padding_rgb=[0, 0, 0],
            mask=erp_mask,
            lat_grid=lat,
            long_grid=lon,
        )

        # Normalize and batch
        x = TF.normalize(
            TF.to_tensor(img_r),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).unsqueeze(0).to(self.dev)
        attn_t = TF.to_tensor((attn > 0).astype(np.float32)).unsqueeze(0).to(self.dev)

        preds, _, _ = self.model(
            x,
            lat_r.unsqueeze(0).to(self.dev),
            lon_r.unsqueeze(0).to(self.dev),
            attn_mask=attn_t,
            lat_grid=torch.tensor(lat_g).unsqueeze(0).to(self.dev),
        )
        preds = preds * psf

        erp_h = float(self.cano[0]) * float(psf)   # ← cast to float

        # Map ERP prediction back to camera space
        _, d_cam, _, active = self._e2c(
            x[0].cpu(), preds[0].cpu(), attn_t[0].cpu(),
            0.0, np.float32(0),
            H, W,
            erp_h, erp_h * 2,
            cam,
        )

        rng    = d_cam[0, 0].numpy().astype(np.float32)
        z      = euclid_to_z(rng, fx, fy, cx, cy)
        active_mask = (active[0].numpy() > 0) & (z > 0)
        return z, active_mask    # (z-depth HxW, bool HxW)