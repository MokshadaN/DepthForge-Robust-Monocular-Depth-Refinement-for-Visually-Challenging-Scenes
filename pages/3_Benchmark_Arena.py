import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

import re, os as _os

st.set_page_config(page_title="Benchmark Arena - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
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
    "margin-bottom:0.3rem;'>📊 Benchmark Arena</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "Quantitative results on the 450-image held-out test set (150 per domain). "
    "Select a metric to see how the Phase 9 refined adapter compares to the MiDaS-Small baseline. "
    "Trade-offs are shown explicitly — not just wins.</div>",
    unsafe_allow_html=True
)

# ── Load CSVs ──
csv_path = os.path.join(BASE, "phase9/extended_visualizations/domain_metric_summary.csv")
df_raw = pd.read_csv(csv_path)

# ── Metric selector ──
col_sel, col_info = st.columns([2, 3])
with col_sel:
    metric = st.selectbox(
        "Select Metric",
        ["AbsRel", "SqRel", "RMSE", "SILog", "d1", "d2", "d3"],
        index=0,
        help="AbsRel=lower is better. d1/d2/d3=higher is better."
    )
with col_info:
    metric_desc = {
        "AbsRel": "Mean Absolute Relative Error (lower is better). Primary metric for DepthForge evaluation.",
        "SqRel": "Mean Squared Relative Error (lower is better). Penalises large outlier errors more heavily.",
        "RMSE": "Root Mean Squared Error (lower is better). Scale-dependent absolute error measure.",
        "SILog": "Scale-Invariant Logarithmic Error (lower is better). Robust to global scale ambiguity.",
        "d1": "Delta-1 accuracy: % pixels with max(pred/gt, gt/pred) < 1.25 (higher is better).",
        "d2": "Delta-2 accuracy: threshold 1.25² (higher is better).",
        "d3": "Delta-3 accuracy: threshold 1.25³ (higher is better).",
    }
    lower_better = metric in ["AbsRel", "SqRel", "RMSE", "SILog"]
    direction = "↓ lower is better" if lower_better else "↑ higher is better"
    st.info(f"**{metric}** — {metric_desc[metric]}  ({direction})")

# ── Per-domain bar chart ──
st.markdown("#### Per-Domain Comparison")
domains = ["camouflage", "glass", "textureless"]
baseline_vals = [df_raw[(df_raw.domain==d) & (df_raw.model=="baseline")][metric].values[0] for d in domains]
refined_vals  = [df_raw[(df_raw.domain==d) & (df_raw.model=="refined")][metric].values[0] for d in domains]

fig = go.Figure()
fig.add_bar(
    name="MiDaS Baseline",
    x=["Camouflage", "Glass", "Textureless"],
    y=baseline_vals,
    marker_color=["#3b82f6","#06b6d4","#8b5cf6"],
    marker_opacity=0.5,
    text=[f"{v:.4f}" for v in baseline_vals],
    textposition="outside",
)
fig.add_bar(
    name="Phase 9 Refined",
    x=["Camouflage", "Glass", "Textureless"],
    y=refined_vals,
    marker_color=["#22c55e","#22d3ee","#c084fc"],
    text=[f"{v:.4f}" for v in refined_vals],
    textposition="outside",
)
fig.update_layout(
    barmode="group",
    plot_bgcolor="#050c18",
    paper_bgcolor="#050c18",
    font_color="#94a3b8",
    font_family="Inter",
    legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", font_color="#94a3b8"),
    xaxis=dict(gridcolor="#0f2347", linecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#0f2347", linecolor="#1e3a5f",
               title=f"{metric} ({'lower better' if lower_better else 'higher better'})"),
    margin=dict(t=20, b=20, l=20, r=20),
    height=380,
)
st.plotly_chart(fig, use_container_width=True)

# ── Gain summary ──
st.markdown("#### Relative Gain by Domain")
g1, g2, g3 = st.columns(3)
domain_colors = {"camouflage": "#22c55e", "glass": "#06b6d4", "textureless": "#a855f7"}
domain_icons = {"camouflage": "🦎", "glass": "🪟", "textureless": "🟫"}
for col, (dom, base, ref) in zip([g1,g2,g3], zip(domains, baseline_vals, refined_vals)):
    gain = (base - ref) / base * 100 if lower_better else (ref - base) / base * 100
    color = domain_colors[dom]
    is_pos = gain > 0
    tag_class = "gain-positive" if is_pos else "gain-negative"
    sign = "+" if is_pos else ""
    with col:
        st.markdown(
            f"<div class='metric-tile' style='border-top:3px solid {color};'>"
            f"<div class='metric-label'>{domain_icons[dom]} {dom.capitalize()}</div>"
            f"<div class='metric-value' style='font-size:1.6rem;'>{sign}{gain:.2f}%</div>"
            f"<div class='metric-sub'>{base:.4f} → {ref:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── Metric heatmap ──
st.markdown("---")
st.markdown("#### All-Metric Summary Heatmap")
img_path = os.path.join(BASE, "phase9/extended_visualizations/13_metric_summary_heatmap.png")
if os.path.exists(img_path):
    st.image(img_path, use_container_width=True)
else:
    st.info("Heatmap image not found.")

# ── Full data table ──
st.markdown("---")
st.markdown("#### Full Metric Table (450-image Test Set)")
df_display = df_raw.copy()
df_display.columns = [c.upper() for c in df_display.columns]
df_display["DOMAIN"] = df_display["DOMAIN"].str.capitalize()
df_display["MODEL"] = df_display["MODEL"].str.capitalize()
st.dataframe(
    df_display.style.format({c: "{:.4f}" for c in ["ABSREL","SQREL","RMSE","SILOG","D1","D2","D3"]}),
    use_container_width=True, hide_index=True
)

# ── Phase 8 vs 9 ──
st.markdown("---")
st.markdown("#### Phase 8 vs Phase 9 Comparison")
local_img = os.path.join(BASE, "phase9/local_analysis/06_phase8_vs_phase9.png")
if os.path.exists(local_img):
    st.image(local_img, use_container_width=True,
             caption="Phase 8 (single-head, 83.4K params) vs Phase 9 (multi-head, 228.4K params)")
else:
    comparison = {
        "Feature": ["Architecture", "Parameters", "Domain Handling", "Dataset Scale",
                    "Domain Balance", "Glass AbsRel Gain", "Camouflage AbsRel Gain"],
        "Phase 8": ["Single shared head", "83,400 (~0.33 MB)", "Monolithic (single mode)",
                    "522 images", "Imbalanced (84% camo)", "General improvement", "General improvement"],
        "Phase 9": ["Shared trunk + 3 Expert Heads", "228,374 (~0.91 MB)", "Dynamic Soft Routing (94.4%)",
                    "3,000 images", "Balanced (1K/1K/1K)", "+4.7% AbsRel reduction", "+3.8% AbsRel reduction"],
    }
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

st.markdown(
    "<div style='margin-top:1rem;padding:0.8rem;background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);"
    "border-radius:10px;font-size:0.78rem;color:#94a3b8;line-height:1.6;'>"
    "<strong style='color:#f87171;'>Caveat:</strong> Phase 8 and Phase 9 use different dataset sizes (522 vs 3,000 images) "
    "and different evaluation protocols. Their absolute metric scores cannot be directly compared in a continuous chart.</div>",
    unsafe_allow_html=True
)
