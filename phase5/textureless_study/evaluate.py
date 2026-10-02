"""Phase-1 harness. models: {name: fn(rgb_u8, K)->z-depth HxW}. samples: list of dicts
{id, rgb: HxWx3 u8, gt: HxW z-depth (m), K: (fx,fy,cx,cy), valid: optional bool HxW}.
Regions: all / lowtex / textured. Metrics: raw + median-scale-aligned. CIs bootstrap over scenes."""
import json, numpy as np
from texture_mask import texture_map, percentile_tau, low_texture_mask
from metrics import masked_metrics, summarize

def fit_tau(val_samples, pct=20):
    return percentile_tau([texture_map(s["rgb"]) for s in val_samples], pct)

def run(models, samples, tau, out_json=None):
    per = {m: {r: [] for r in ("all", "lowtex", "textured")} for m in models}
    for s in samples:
        T = texture_map(s["rgb"]); lt = low_texture_mask(T, tau)
        valid = s.get("valid", s["gt"] > 1e-3)
        regions = {"all": np.ones_like(lt), "lowtex": lt, "textured": ~lt}
        for name, fn in models.items():
            pred = fn(s["rgb"], s["K"])
            if isinstance(pred, tuple):          # (depth, active_mask) as UniDACRunner returns
                pred, act = pred; vv = valid & act
            else:
                vv = valid
            for r, reg in regions.items():
                per[name][r].append(masked_metrics(pred, s["gt"], reg, vv))
    table = {}
    for name in models:
        table[name] = {}
        for r in ("all", "lowtex", "textured"):
            table[name][r] = {f"{k}_{m}": summarize(per[name][r], (k, m))
                              for k in ("raw", "aligned") for m in ("absrel", "rmse", "d1")}
    if out_json: json.dump(table, open(out_json, "w"), indent=1)
    return table

def fmt(table, key="raw_absrel"):
    print(f"{'model':22s}" + "".join(f"{r:>26s}" for r in ("all", "lowtex", "textured")))
    for n, t in table.items():
        row = ""
        for r in ("all", "lowtex", "textured"):
            v = t[r][key]; row += f"{'' if v is None else f'{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]':>26s}"
        print(f"{n:22s}{row}")