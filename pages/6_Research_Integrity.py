import streamlit as st
import pandas as pd
import os

import re, os as _os

st.set_page_config(page_title="Research Integrity - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
import sys
sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from nav_utils import inject_theme, render_sidebar

inject_theme()
render_sidebar()


st.markdown(
    "<div style='font-family:Outfit,sans-serif;font-size:2rem;font-weight:900;"
    "background:linear-gradient(135deg,#f0f6ff,#93c5fd,#a78bfa);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "margin-bottom:0.3rem;'>🔐 Research Integrity & Next Steps</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "Dataset provenance, evaluation protocol, limitations, pseudo-GT caveats, and recommended next experiments.</div>",
    unsafe_allow_html=True
)

# ── Dataset provenance ──
st.markdown("#### Dataset Provenance")
ds = {
    "Domain": ["Camouflage", "Glass/Transparent", "Textureless/Indoor"],
    "Source Dataset": ["CAMO-FS (Thanhdat3010/CAMO-FS)", "GSD — Glass Surface Detection (garrying/GSD)", "NYU Depth v2 (tanganke/nyuv2)"],
    "Ingestion": ["Direct JPEG stream from Hugging Face", "Parquet streaming with binary JPEG decoding", "Split-aware Parquet (795 train + 205 val)"],
    "Images Ingested": [1000, 1000, 1000],
    "Synthetic Fallbacks": ["None", "None", "None"],
}
st.dataframe(pd.DataFrame(ds), use_container_width=True, hide_index=True)

# ── Split protocol ──
st.markdown("---")
st.markdown("#### Stratified Zero-Leakage Split Protocol")
split = {
    "Split": ["Train", "Validation", "Test"],
    "Camouflage": [700, 150, 150],
    "Glass": [700, 150, 150],
    "Textureless": [700, 150, 150],
    "Total": [2100, 450, 450],
    "Purpose": [
        "Model parameter optimization",
        "Checkpoint selection (Val SILog)",
        "Unseen final quantitative evaluation"
    ]
}
st.dataframe(pd.DataFrame(split), use_container_width=True, hide_index=True)

# ── Evidence type glossary ──
st.markdown("---")
st.markdown("#### Evidence Type Glossary")
st.markdown(
    "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;'>",
    unsafe_allow_html=True
)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "<div class='df-card' style='border-top:3px solid #22c55e;'>"
        "<div style='font-size:0.65rem;color:#22c55e;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:0.4rem;'>QUANTITATIVE BENCHMARK</div>"
        "<div style='font-size:0.84rem;color:#94a3b8;line-height:1.6;'>Metrics (AbsRel, SILog, d1) computed on the 450 held-out test images. "
        "These are the primary evidence for DepthForge's improvement claims.</div>"
        "</div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        "<div class='df-card' style='border-top:3px solid #f59e0b;'>"
        "<div style='font-size:0.65rem;color:#f59e0b;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:0.4rem;'>TEACHER AGREEMENT</div>"
        "<div style='font-size:0.84rem;color:#94a3b8;line-height:1.6;'>Similarity to DPT-Large pseudo-targets. This measures how well Phase 9 replicates "
        "the teacher's depth map — NOT physical metric accuracy against real sensor depth.</div>"
        "</div>",
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        "<div class='df-card' style='border-top:3px solid #3b82f6;'>"
        "<div style='font-size:0.65rem;color:#3b82f6;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:0.4rem;'>QUALITATIVE OBSERVATION</div>"
        "<div style='font-size:0.84rem;color:#94a3b8;line-height:1.6;'>Visual inspection of boundary recovery, depth bleeding reduction, and surface continuity "
        "in the qualitative panels. Subjective but informative for demonstrating failure correction.</div>"
        "</div>",
        unsafe_allow_html=True
    )

# ── Known limitations ──
st.markdown("---")
st.markdown("#### Known Limitations")
limitations = [
    ("Pseudo-GT Dependency", "#f97316",
     "DPT-Large outputs are pseudo-targets, not LiDAR/ToF sensor-derived ground truth. The reported metrics measure "
     "alignment with a strong teacher model, not physical depth accuracy. Language like 'metric-accurate depth' should be avoided."),
    ("SILog Trade-off", "#ef4444",
     "While AbsRel improves, SILog worsens on camouflage (+6.9%) and glass (+12.7%). Sharp boundary corrections "
     "create high-frequency log-ratio divergence against the teacher's smooth boundaries. This is an accepted "
     "trade-off in edge-guided refinement."),
    ("Textureless Near-Identity", "#8b5cf6",
     "Only +0.2% AbsRel improvement on textureless scenes. This demonstrates stability (MiDaS already performs "
     "well there) but means the textureless expert head learned a near-identity mapping — limited specialization."),
    ("Per-Image Alignment", "#3b82f6",
     "The metric code applies per-image scale alignment before computing AbsRel. The exact alignment procedure "
     "must be applied consistently between baseline and refined predictions for fair comparison."),
    ("Training Not Converged", "#f59e0b",
     "The loss curve had not fully plateaued at epoch 60. Training for 80–100 epochs with cosine warm restarts "
     "may yield further AbsRel gains (Phase 10+ recommendation)."),
]
for title, color, desc in limitations:
    st.markdown(
        f"<div class='df-card' style='border-left:3px solid {color};margin-bottom:0.5rem;'>"
        f"<div style='font-family:Outfit,sans-serif;font-size:0.95rem;font-weight:700;color:#f0f6ff;margin-bottom:0.35rem;'>{title}</div>"
        f"<div style='font-size:0.83rem;color:#94a3b8;line-height:1.6;'>{desc}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── Next steps ──
st.markdown("---")
st.markdown("#### Recommended Next Steps (Phase 10+)")
next_steps = [
    ("Uncertainty-Weighted Routing", "#a855f7", False,
     "Use entropy of the router output w as uncertainty metric to adaptively control refinement magnitude alpha."),
    ("Extended Epoch Schedule", "#3b82f6", False,
     "Train for 80–100 epochs with cosine warm restarts. Loss curve had not plateaued at epoch 60."),
    ("Physical Sensor Validation", "#22c55e", False,
     "Benchmark against real LiDAR / Time-of-Flight ground truth rather than DPT pseudo-labels."),
    ("ONNX / TensorRT Export", "#06b6d4", False,
     "Export phase9_multihead_best.pt to ONNX for real-time robotic edge deployment."),
    ("Extended Domain Coverage", "#f59e0b", False,
     "Add night-time, rain, and fog domains to test robustness to additional optical failure modes."),
]
for title, color, done, desc in next_steps:
    icon = "✅" if done else "◻"
    st.markdown(
        f"<div style='display:flex;gap:0.6rem;align-items:flex-start;padding:0.6rem;background:rgba(13,31,60,0.5);"
        f"border-radius:8px;margin-bottom:0.4rem;border:1px solid {color}22;'>"
        f"<span style='color:{color};font-size:1rem;margin-top:2px;'>{icon}</span>"
        f"<div><div style='font-size:0.88rem;font-weight:700;color:#f0f6ff;margin-bottom:2px;'>{title}</div>"
        f"<div style='font-size:0.8rem;color:#64748b;line-height:1.5;'>{desc}</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='padding:1rem;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.2);"
    "border-radius:12px;font-size:0.78rem;color:#64748b;line-height:1.7;'>"
    "<strong style='color:#60a5fa;'>Reproducibility Links:</strong> "
    "Phase 9 checkpoint: <code>phase9/phase9_adapter_best.zip</code> · "
    "Training script: <code>phase9/kaggle_phase9_finetune.py</code> · "
    "Results: <code>phase9/extended_visualizations/</code> · "
    "Metrics: <code>phase9/extended_visualizations/domain_metric_summary.csv</code><br>"
    "<strong style='color:#60a5fa;'>Compute:</strong> NVIDIA Tesla T4 (15.6 GB VRAM) · PyTorch 2.11.0+cu128 · Kaggle Cloud · October 14, 2026"
    "</div>",
    unsafe_allow_html=True
)
