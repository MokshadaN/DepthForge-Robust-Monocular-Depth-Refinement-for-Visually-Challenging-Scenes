import streamlit as st
import json, pandas as pd

import re, os as _os

st.set_page_config(page_title="Phase Explorer - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
import sys
sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from nav_utils import inject_theme, render_sidebar

inject_theme()
render_sidebar()

with open("dashboard_data/phase_metadata.json", encoding="utf-8") as f:
    phases = json.load(f)["phases"]

st.markdown(
    "<div style='font-family:Outfit,sans-serif;font-size:2rem;font-weight:900;"
    "background:linear-gradient(135deg,#f0f6ff,#93c5fd,#a78bfa);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "margin-bottom:0.3rem;'>🗺️ Phase Explorer</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "DepthForge evolved through 9 experimental phases — from a baseline road-scene detector to a multi-head "
    "domain-expert depth refinement system. Each phase had a concrete goal, produced measurable evidence, "
    "and directly motivated the next step. Click any phase to expand its full story.</div>",
    unsafe_allow_html=True
)

tag_colors = {
    "FOUNDATION": "#3b82f6", "FAILURE DISCOVERY": "#ef4444", "BENCHMARK": "#f59e0b",
    "EVALUATION": "#8b5cf6", "ARCHITECTURE": "#06b6d4", "AUDIT": "#f97316",
    "VERIFIED": "#22c55e", "BREAKTHROUGH": "#a855f7"
}

for ph in phases:
    color = ph["color"]
    tag_c = tag_colors.get(ph["tag"], color)
    with st.expander(
        f"Phase {ph['id']:02d}  ·  {ph['title']}  ·  {ph['tag']}",
        expanded=(ph["id"] == 9)
    ):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(
                f"<div style='border-left:3px solid {color};padding-left:0.8rem;margin-bottom:0.8rem;'>"
                f"<div style='font-family:Outfit,sans-serif;font-size:1.2rem;font-weight:800;color:#f0f6ff;'>{ph['title']}</div>"
                f"<div style='font-size:0.82rem;color:#94a3b8;font-style:italic;'>{ph['subtitle']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"**🎯 Goal:** {ph['goal']}")
            st.markdown(f"**⚙️ Technique:** {ph['technique']}")
            st.markdown("**🔍 Key Findings:**")
            for finding in ph["findings"]:
                st.success(finding)
            if ph["problems"]:
                st.markdown("**⚠️ Issues Identified:**")
                for p in ph["problems"]:
                    st.warning(p)
        with c2:
            st.markdown(
                f"<div style='background:rgba(13,31,60,0.8);border:1px solid {color}44;"
                f"border-top:3px solid {color};border-radius:12px;padding:1rem;'>"
                f"<div style='font-size:0.65rem;color:{color};letter-spacing:.12em;font-weight:700;"
                f"text-transform:uppercase;margin-bottom:0.6rem;'>📋 Evidence</div>"
                + "".join(f"<div style='font-size:0.82rem;color:#94a3b8;padding:3px 0;'>✓ {e}</div>" for e in ph["evidence"])
                + "</div>",
                unsafe_allow_html=True
            )
            st.info(f"**🏆 Outcome:** {ph['outcome']}")

st.markdown("---")
st.markdown("#### Phase-at-a-Glance")
df_rows = []
for p in phases:
    df_rows.append({
        "Phase": f"P{p['id']:02d}",
        "Title": p["title"],
        "Tag": p["tag"],
        "Key Finding": p["findings"][0][:90] + ("..." if len(p["findings"][0]) > 90 else ""),
        "Outcome": p["outcome"][:80] + ("..." if len(p["outcome"]) > 80 else ""),
    })
st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)
