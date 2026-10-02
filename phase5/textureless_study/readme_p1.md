# Phase 1 scaffold (TextureAware-UniDAC)

Files: texture_mask.py (low-texture mask), metrics.py (raw + scale-aligned, scene-level bootstrap CIs),
evaluate.py (harness), unidac_runner.py (UniDAC adapter, UNTESTED), smoke_test.py (synthetic check, passes).

Tested here: texture mask, metrics, harness (synthetic data only).
NOT tested: unidac_runner.py (needs HF checkpoint + GPU; huggingface.co is not reachable from the sandbox).

Setup: clone https://github.com/girish1511/UniDAC, follow its install, download checkpoints/unidac.pt, run `bash demo.sh`
first. Then: UniDACRunner(repo, ckpt).predict_z(rgb_u8, fx, fy, cx, cy) -> (z-depth, valid mask).

Conventions to keep identical across models: z-depth (not Euclidean range), original image grid, same valid-GT mask,
same intrinsics (run GT-K and predicted-K as separate settings), tau fitted on a validation split only.
Per-dataset loaders (NYU, Shape2.5D, ...) still need writing: each returns {id, rgb, gt, K, valid}.