"""Dataset loaders for Phase-1 evaluation.

Each loader yields sample dicts:
  {
    "id"    : str             – unique scene/frame identifier
    "rgb"   : np.uint8 HxWx3 – RGB image
    "gt"    : np.float32 HxW  – z-depth ground truth in metres
    "K"     : (fx, fy, cx, cy) – camera intrinsics in pixels
    "valid" : np.bool HxW     – True where GT is reliable
  }

Usage
-----
  from loaders.nyu_loader      import load_nyu
  from loaders.scannetpp_loader import load_scannetpp
  from loaders.shape25d_loader  import load_shape25d

  for sample in load_nyu("/data/nyu"):
      ...
"""
from .nyu_loader      import load_nyu, load_nyu_list
from .scannetpp_loader import load_scannetpp, load_scannetpp_list
from .shape25d_loader  import load_shape25d, load_shape25d_list

__all__ = [
    "load_nyu", "load_nyu_list",
    "load_scannetpp", "load_scannetpp_list",
    "load_shape25d", "load_shape25d_list",
]
