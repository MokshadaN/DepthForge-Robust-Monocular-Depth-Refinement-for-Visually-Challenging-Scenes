import streamlit as st
import os

import re, os as _os

st.set_page_config(page_title="Architecture - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
import sys
sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from nav_utils import inject_theme, render_sidebar

inject_theme()
render_sidebar()

BASE = os.path.dirname(os.path.dirname(__file__))

st.markdown(
    "<div style='font-family:Outfit,sans-serif;font-size:2rem;font-weight:900;"
    "background:linear-gradient(135deg,#f0f6ff,#93c5fd,#a78bfa);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "margin-bottom:0.3rem;'>🏗️ Architecture & Innovations</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "Phase 9 introduces a Multi-Head Dilated Residual Architecture with automatic soft domain routing. "
    "Explore the pipeline components, their purpose, and the four key engineering contributions.</div>",
    unsafe_allow_html=True
)

# ── SVG Architecture Diagram ──
st.markdown("#### System Architecture Diagram")

svg_path = os.path.join(BASE, "assets", "depthforge_phase9_inference_pipeline.svg")
if not os.path.exists(svg_path):
    svg_path = os.path.join(BASE, "assets", "architecture_diagram.svg")

if os.path.exists(svg_path):
    st.image(
        svg_path,
        caption="DepthForge Phase 9 — Multi-Head Inference Pipeline Architecture",
        use_container_width=True,
    )
else:
    st.warning("SVG diagram not found at assets/depthforge_phase9_inference_pipeline.svg")

with st.expander("📄 View Text Schematic (ASCII Flowchart)"):
    _pipeline = r"""+=========================================================================+
|            DEPTHFORGE PHASE 9  -  INFERENCE PIPELINE                   |
+=========================================================================+
|  INPUTS                                                                 |
|   RGB Image (3ch) + MiDaS Base Depth (1ch) + Sobel Gradient (1ch)     |
|                     5-Channel Input Tensor                              |
+=========================================================================+
|  SHARED MULTI-SCALE DILATED TRUNK  (228K params total)                 |
|   Stem: Conv(5->64) + BatchNorm + ReLU + Drop2d(0.10)                  |
|   +----------+----------+----------+                                    |
|   |  d=1     |  d=2     |  d=4     |  Parallel dilated convolutions    |
|   | 32 ch    | 32 ch    | 32 ch    |  Fine / Mid / Wide context        |
|   +----------+----------+----------+                                    |
|              Concat -> 96 channels                                      |
+=========================================================================+
|  SOFT DOMAIN ROUTER                                                     |
|   GAP(96ch) -> Linear(96->32) -> ReLU -> Linear(32->3) -> Softmax      |
|   Output: w_camo, w_glass, w_tex  (sum=1.0)  -- 94.4% accuracy        |
+====================+=================+=================================+
|  CAMOUFLAGE HEAD   |   GLASS HEAD    |   TEXTURELESS HEAD               |
|  Expert residual   | Expert residual |   Expert residual                |
+====================+=================+=================================+
|  SOFT ROUTING BLEND                                                     |
|   delta = w_camo*d_camo + w_glass*d_glass + w_tex*d_tex                |
+=========================================================================+
|  TANH GATED RESIDUAL ADDITION                                           |
|   D_refined = clamp(D_base + 0.30 * tanh(delta),  0.0, 1.0)           |
|   OUTPUT: Refined depth map in [0, 1]                                  |
+=========================================================================+"""
    st.code(_pipeline, language=None)

# ── Component deep-dives ──
st.markdown("#### Component Deep-Dives")
components = [
    ("5-Channel Input Tensor", "#3b82f6", "Inputs",
     "X_in = [D_base, I_R, I_G, I_B, E_Sobel] in R^(5 x H x W)",
     "Stacking base depth + RGB + Sobel gradient gives the adapter both photometric appearance "
     "and structural edge cues. The Sobel prior is the key signal that enables boundary recovery -- "
     "it amplifies optical discontinuities that depth estimators tend to smooth over.",
     "phase9/kaggle_phase9_finetune.py", "MultiHeadDilatedAdapter.__init__"),
    ("Shared Multi-Scale Dilated Trunk", "#06b6d4", "Feature Extraction",
     "Parallel Conv2d with d in {1,2,4} -> concat 96ch",
     "d=1 captures fine edge transitions (pixel-level boundary cues). d=2 captures mid-range "
     "surface texture patterns. d=4 captures wider contextual surface geometry — all without "
     "downsampling, preserving full spatial resolution for dense depth output.",
     "phase9/kaggle_phase9_finetune.py", "DilatedBranch"),
    ("Soft Domain Router", "#a855f7", "Classification",
     "GAP + Linear(96→32→3) + Softmax → w ∈ ℝ³",
     "Global Average Pooling compresses the 96-channel feature map to a domain summary vector. "
     "Two linear layers learn domain-discriminative statistics. The softmax output is differentiable, "
     "enabling end-to-end training with cross-entropy domain loss (weight 0.1). Result: 94.4% accuracy "
     "with zero inference-time domain labels.",
     "phase9/kaggle_phase9_finetune.py", "DomainRouter"),
    ("Three Expert Heads", "#22c55e", "Specialization",
     "Head_camo, Head_glass, Head_tex → Δ_k each",
     "Each head is a dedicated Conv block that learns domain-specific residual corrections. "
     "The camouflage head learns to amplify boundary discontinuities. The glass head learned to amplify "
     "subtle specular highlights, pulling the glass surface plane forward. The textureless head "
     "learns a near-identity delta — correctly detecting that MiDaS already performs well there.",
     "phase9/kaggle_phase9_finetune.py", "ExpertHead"),
    ("Tanh Gated Residual", "#f97316", "Output",
     "D_refined = clamp(D_base + 0.30·tanh(Δ), 0, 1)",
     "The tanh activation bounds the maximum residual correction to ±0.30 in normalized depth space. "
     "This prevents runaway corrections while still allowing meaningful boundary recovery. "
     "alpha=0.30 was selected empirically — large enough to recover glass/camouflage boundaries, "
     "small enough to avoid corrupting textureless scenes.",
     "phase9/kaggle_phase9_finetune.py", "forward()"),
]

for name, color, category, formula, explanation, source_file, function in components:
    with st.expander(f"{name}  ·  {category}"):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Formula / Interface:**")
            st.code(formula, language="python")
            st.markdown(f"**Explanation:** {explanation}")
        with c2:
            st.markdown(
                f"<div style='background:rgba(13,31,60,0.8);border:1px solid {color}44;"
                f"border-top:3px solid {color};border-radius:10px;padding:0.9rem;'>"
                f"<div style='font-size:0.65rem;color:{color};font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:0.5rem;'>Source Reference</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#94a3b8;margin-bottom:4px;'>{source_file}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{color};'>{function}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── Four contributions ──
st.markdown("---")
st.markdown("#### Four Key Contributions")
contrib = [
    ("1. Automatic Soft Domain Routing", "#a855f7",
     "Three expert heads are blended according to learned scene-domain weights instead of requiring "
     "the user to supply a domain label at inference time. The router is trained end-to-end with a "
     "cross-entropy loss at weight 0.1.",
     "94.4% routing accuracy on 450 held-out images. 425/450 correctly classified."),
    ("2. Multi-Scale Dilated Residual Refinement", "#06b6d4",
     "Parallel dilations d∈{1,2,4} capture fine local edges and wider spatial context while preserving "
     "the depth map's spatial resolution (no downsampling). Outputs concatenated to 96-channel feature map.",
     "Enables simultaneous recovery of pixel-level silhouette boundaries and regional surface geometry."),
    ("3. Cross-Model Supervision", "#22c55e",
     "The adapter uses DPT-Large outputs as pseudo-targets rather than learning to reproduce its own "
     "MiDaS-Small baseline. This breaks the circular self-distillation loop discovered in Phase 7.",
     "DPT-Large is a 345M-parameter model. Phase 9 distills its geometric fidelity into 228K parameters."),
    ("4. Diagnostic & Evaluation Improvements", "#f59e0b",
     "Phases 7-8 exposed and corrected: (1) self-distillation loop, (2) BSI boundary noise metric, "
     "(3) class imbalance (95% camouflage), (4) grid-crop data leakage. Phase 9 inherits the corrected protocol.",
     "3,000 authentic images (1K/domain), zero synthetic fallbacks, stratified scene-level splits."),
]
c1, c2 = st.columns(2)
for i, (title, color, desc, result) in enumerate(contrib):
    with (c1 if i % 2 == 0 else c2):
        st.markdown(
            f"<div class='df-card' style='border-left:3px solid {color};'>"
            f"<div style='font-family:Outfit,sans-serif;font-size:1rem;font-weight:700;color:#f0f6ff;margin-bottom:0.5rem;'>{title}</div>"
            f"<div style='font-size:0.83rem;color:#94a3b8;line-height:1.65;margin-bottom:0.7rem;'>{desc}</div>"
            f"<div style='font-size:0.78rem;color:{color};font-family:JetBrains Mono,monospace;'>→ {result}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
