import streamlit as st
import os

import re, os as _os

st.set_page_config(page_title="Visual Lab - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
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
    "margin-bottom:0.3rem;'>🔬 Visual Lab</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "Browse the qualitative evidence for all three domains. Each panel shows: "
    "<strong style='color:#60a5fa;'>RGB Input</strong> → "
    "<strong style='color:#a78bfa;'>Sobel Edge Prior</strong> → "
    "<strong style='color:#f59e0b;'>MiDaS Baseline</strong> → "
    "<strong style='color:#06b6d4;'>DPT-Large Teacher</strong> → "
    "<strong style='color:#22c55e;'>Phase 9 Refined</strong> → "
    "<strong style='color:#ef4444;'>Error Map</strong></div>",
    unsafe_allow_html=True
)

domain_info = {
    "Camouflage": {
        "icon": "🦎", "color": "#22c55e",
        "panel_path": "phase9/results_extracted/panels/panel_phase9_camouflage.png",
        "grid_path": "phase9/extended_visualizations/11_qualitative_camouflage.png",
        "desc": "Animals and objects whose texture mimics their surroundings. The adapter recovers sharp silhouette boundaries where MiDaS produces an undifferentiated flat slope.",
        "metric": "AbsRel: 0.6441 → 0.6197 (+3.8%)"
    },
    "Glass": {
        "icon": "🪟", "color": "#06b6d4",
        "panel_path": "phase9/results_extracted/panels/panel_phase9_glass.png",
        "grid_path": "phase9/extended_visualizations/11_qualitative_glass.png",
        "desc": "Glass surfaces transmit background rays, causing the baseline to look through glass to the background wall. The glass expert head pulls the predicted surface forward.",
        "metric": "AbsRel: 0.5264 → 0.5018 (+4.7%)"
    },
    "Textureless": {
        "icon": "🟫", "color": "#a855f7",
        "panel_path": "phase9/results_extracted/panels/panel_phase9_textureless.png",
        "grid_path": "phase9/extended_visualizations/11_qualitative_textureless.png",
        "desc": "Smooth monochromatic surfaces lack photometric gradients, causing depth warping. The adapter maintains smooth planar gradients without hallucinating texture.",
        "metric": "AbsRel: 0.5838 → 0.5827 (+0.2% — stable identity)"
    },
}

domain_sel = st.selectbox(
    "Select domain to inspect:",
    list(domain_info.keys()),
    format_func=lambda d: f"{domain_info[d]['icon']} {d}"
)
info = domain_info[domain_sel]

st.markdown(
    f"<div class='df-card' style='border-left:3px solid {info['color']};margin-bottom:1rem;'>"
    f"<div style='font-size:0.65rem;color:{info['color']};letter-spacing:.12em;font-weight:700;text-transform:uppercase;margin-bottom:0.4rem;'>DOMAIN PROFILE</div>"
    f"<div style='font-family:Outfit,sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;'>{info['icon']} {domain_sel}</div>"
    f"<div style='font-size:0.85rem;color:#94a3b8;line-height:1.65;margin-bottom:0.6rem;'>{info['desc']}</div>"
    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.8rem;color:{info['color']};font-weight:600;'>{info['metric']}</span>"
    f"</div>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Full Comparison Panel", "Qualitative Grid (multi-sample)"])

with tab1:
    panel_path = os.path.join(BASE, info["panel_path"])
    if os.path.exists(panel_path):
        st.markdown(
            "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem;'>"
            "6-column panel: RGB → Sobel → Base MiDaS → DPT Teacher → Phase 9 Refined → Error Map</div>",
            unsafe_allow_html=True
        )
        st.image(panel_path, use_container_width=True)
    else:
        st.warning(f"Panel image not found at: {panel_path}")

with tab2:
    grid_path = os.path.join(BASE, info["grid_path"])
    if os.path.exists(grid_path):
        st.markdown(
            "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem;'>"
            "Multi-sample qualitative grid across the test set for this domain</div>",
            unsafe_allow_html=True
        )
        st.image(grid_path, use_container_width=True)
    else:
        st.warning(f"Grid image not found at: {grid_path}")

st.markdown("---")
st.markdown("#### Column Legend")
col_desc = [
    ("RGB Input", "Original test image from the held-out set", "#f0f6ff"),
    ("Sobel Edge Prior", "Gradient magnitude — structural cue fed to adapter", "#a78bfa"),
    ("Base MiDaS", "Unmodified MiDaS-Small output — shows the failure", "#f59e0b"),
    ("DPT-Large Teacher", "Pseudo-ground-truth target used during training", "#06b6d4"),
    ("Phase 9 Refined", "Adapter output — the corrected depth map", "#22c55e"),
    ("Error Map", "Absolute difference between refined and teacher", "#ef4444"),
]
cols = st.columns(6)
for col, (name, desc, color) in zip(cols, col_desc):
    with col:
        st.markdown(
            f"<div style='border-top:3px solid {color};border-radius:8px;padding:0.8rem 0.5rem;"
            f"background:rgba(13,31,60,0.6);text-align:center;'>"
            f"<div style='font-size:0.72rem;font-weight:700;color:{color};margin-bottom:4px;'>{name}</div>"
            f"<div style='font-size:0.68rem;color:#64748b;line-height:1.4;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown(
    "<div style='margin-top:1rem;padding:0.8rem;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);"
    "border-radius:10px;font-size:0.78rem;color:#94a3b8;line-height:1.6;'>"
    "<strong style='color:#fbbf24;'>Note:</strong> DPT-Large outputs are pseudo-targets, not sensor-derived ground truth. "
    "The error maps measure deviation from the teacher, not physical metric depth accuracy.</div>",
    unsafe_allow_html=True
)
