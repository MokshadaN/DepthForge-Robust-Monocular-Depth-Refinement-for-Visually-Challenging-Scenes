# Final workflow of TextureAware-UniDAC

This is the end-to-end flow as the project now stands, with the fixes from my reviews. Items marked **(mine)** are my additions from those reviews, not part of your original plan.

## 1. The question

Does UniDAC, which predicts relative depth plus a spatially varying scale, stay accurate when RGB appearance is a weak depth cue? If it doesn't, can a texture- and geometry-aware refinement fix it without damaging depth boundaries? The workflow is built so that either answer is a valid result.

## 2. Per-image inference pipeline (the system)

```text
RGB + intrinsics K
   │
   ├─► UniDAC (frozen) ─► metric depth D
   │
   └─► texture map T ─► low-texture mask M
                              │
        D + K ─► back-project ─► per-component plane fit (RANSAC)
                              │
        low texture AND high plane confidence ─► refine
        otherwise ─► keep D
                              │
                     boundary protection
                              │
                        final depth D*
```

**Step by step**

1. **Run UniDAC.** The checkpoint comes from the official repo. It needs camera parameters; for images with unknown intrinsics, the repo's in-the-wild path uses AnyCalib. Output is metric depth D.
2. **Texture map.** T = α·|∇I| + β·Var(I) + γ·H(I) over a k×k window. Mask M = [T < τ], then morphological cleanup and connected components. Fix α, β, γ, τ on a validation split only.
3. **Back-project** each component's pixels: X=(u−cₓ)D/fₓ, Y=(v−c_y)D/f_y, Z=D.
4. **Plane fit** per component with RANSAC (aX+bY+cZ+d=0). Plane confidence is the inlier ratio plus the residual.
5. **Gate.** Refine only if the region is low-texture **and** the plane confidence is high. Non-planar regions keep UniDAC's depth.
6. **Refine.** For a pixel with ray (rₓ, r_y) = ((u−cₓ)/fₓ, (v−c_y)/f_y), the plane-implied depth is Z = −d / (a·rₓ + b·r_y + c). Replace or blend D with this value. This works for perspective cameras; fisheye and 360° need unit ray directions per pixel instead.
7. **Boundary protection.** Don't refine across depth discontinuities. Blend weights fall to zero within a band around the boundary. On TCDB, use the known object masks to define the boundary. In real camouflage scenes RGB edges are unreliable, so say plainly how protection works there.

TAR is heuristic post-processing, and I can't confirm in advance that it helps. It can't correct a consistent scale offset across a wall. It only removes local bumps.

## 3. Evaluation workflow, in three phases

**Phase 1 — Benchmark (what happens?)**
- Run UniDAC and 4–5 baselines (for example DACU, UniDepthV2, Metric3Dv2, Depth Anything V2 Metric) on NYU, KITTI-360, ScanNet++, Shape2.5D and Texture-less Deformable Surfaces.
- Run two camera settings separately: GT intrinsics and predicted intrinsics.
- Report AbsRel, RMSE and δ₁ over all pixels, the low-texture mask and the textured mask. Add scale-aligned error (multiply by median(D)/median(D̂) per mask) next to raw error **(mine)**.
- Output: failure-mode analysis. This phase is descriptive, because Shape2.5D has no textured control.

**Phase 2 — Controlled stress test (why?)**
- Build TCDB: fixed geometry, fixed camera, GT depth, with only the background and material varied across five similarity bins. Define similarity as a measured distance (for example in color or feature space between the object and its surrounding ring), not nominal percentages **(mine)**. Keep shading and shadows consistent.
- Compute error versus similarity, the object/background/boundary split, and ΔD = |D_obj − D_bg|.
- CAMO and COD10K are qualitative only, since I found no depth ground truth in them.
- Output: error-vs-similarity curves with per-scene bootstrap CIs.

**Phase 3 — Refinement (can we improve it?)**
- Build the ablation ladder: UniDAC, +mask, +plane, +boundary, full TAR.
- Apply the same refinement to the baselines as a fairness control, so you can tell whether a gain comes from UniDAC or just from smoothing **(mine)**.
- Evaluate on interior, boundary and planar-residual metrics.
- Optional demo: object distance using a segmentation mask, with GT masks for the formal evaluation so detector failure isn't counted as depth failure **(mine)**.

## 4. Decision gates

| After | If | Then |
|---|---|---|
| Phase 1 | No failure clusters on low-texture regions | Report as a benchmark. Phase 2 still tests the cause. |
| Phase 2 | Error rises with similarity and is localized at planar low-texture regions | Proceed to TAR. |
| Phase 2 | Errors are mostly a global scale offset | TAR is unlikely to help. Look at scale correction or fusion instead. |
| Phase 3 | TAR beats the fair baseline with CIs not overlapping | Report as the method. Otherwise report the negative result. |

## 5. Outputs

- Mask-wise metric tables, GT versus predicted intrinsics.
- Error-vs-similarity plots.
- Qualitative panels (RGB, GT, each model, error maps, texture mask).
- Ablation table and runtime/memory table.

## 6. Still unsettled

- COD-DA2's contents and the Texture-less Deformable Surfaces sample count are unverified.
- Novelty is unchecked, so run a literature search before claiming a new benchmark.
- Whether UniDAC is actually better on textureless data is unknown until Phase 1–2 run.

**Confidence:** high on the pipeline logic and formulas, since they follow from the plan and standard geometry. Moderate on the design choices, which are my judgment and untested. I can write the pipeline code scaffold (texture mask, plane fit, metrics) if you want to start Phase 1.