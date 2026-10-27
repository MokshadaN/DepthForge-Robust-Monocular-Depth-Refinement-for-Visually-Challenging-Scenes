"""
DepthForge Dashboard - Shared Navigation & Styling Utility
Eliminates navigation duplication, ensures clean UI, and provides unified dark CV theme.
"""

import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;600;700;800;900&display=swap');

:root {
    --bg-primary: #050c18;
    --bg-secondary: #0a1628;
    --bg-card: #0d1f3c;
    --bg-card2: #0f2347;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-orange: #f97316;
    --accent-pink: #ec4899;
    --text-primary: #f0f6ff;
    --text-secondary: #94a3b8;
    --text-muted: #4b6080;
    --border: rgba(59, 130, 246, 0.18);
    --border-bright: rgba(59, 130, 246, 0.40);
    --glow-blue: 0 0 24px rgba(59, 130, 246, 0.35);
}

/* App Background & Typography */
html, body, [class*="css"], .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Clean Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c18 0%, #07152a 100%) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

/* HIDE STREAMLIT AUTO-GENERATED DUPLICATE NAVIGATION COMPLETELY */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarNavItems"] {
    display: none !important;
}

/* Custom Navigation Links */
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 0.35rem 0.65rem !important;
    color: #94a3b8 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin-bottom: 2px !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(59, 130, 246, 0.12) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
    color: #ffffff !important;
    transform: translateX(4px) !important;
}

[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.22), rgba(168, 85, 247, 0.18)) !important;
    border-color: rgba(96, 165, 250, 0.5) !important;
    color: #93c5fd !important;
    font-weight: 600 !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* Cards & Metrics */
.df-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .25s, box-shadow .25s;
}

.df-card:hover {
    border-color: var(--border-bright);
    box-shadow: var(--glow-blue);
}

.metric-tile {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all .3s;
}

.metric-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}

.metric-tile.blue::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.metric-tile.purple::before { background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink)); }
.metric-tile.green::before { background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan)); }
.metric-tile.orange::before { background: linear-gradient(90deg, var(--accent-orange), var(--accent-pink)); }

.metric-tile:hover {
    transform: translateY(-3px);
    box-shadow: var(--glow-blue);
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.3rem 0;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.metric-sub {
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Interactive & Forms */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all .25s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, .4) !important;
}

div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue), #1d4ed8) !important;
    color: white !important;
}

hr {
    border-color: var(--border) !important;
}
</style>
"""


def inject_theme():
    """Injects unified CSS, typography, and overrides Streamlit's default navigation."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar():
    """Renders clean, categorized, non-cluttered navigation with active status."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.8rem 0 0.8rem 0;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:.2em;font-weight:700;">DEPTHFORGE LAB</div>
                <div style="font-family:'Outfit',sans-serif;font-size:1.45rem;font-weight:900;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-top:2px;line-height:1.15;">Research Portal</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">Robust Monocular Depth Refinement</div>
            </div>
            <hr style="margin: 0 0 0.8rem 0;">
            """,
            unsafe_allow_html=True
        )

        # ── Group 1: Interactive Lab ──────────────────────────────
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#60a5fa;letter-spacing:.15em;font-weight:700;margin:0.2rem 0 0.4rem 0.3rem;text-transform:uppercase;\">Interactive Lab</div>",
            unsafe_allow_html=True
        )
        st.page_link("pages/0_Live_Demo.py", label="🚀 Live Demonstration", use_container_width=True)
        st.page_link("pages/2_Visual_Lab.py", label="🔬 Visual Evidence Lab", use_container_width=True)

        # ── Group 2: Research Insights ────────────────────────────
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#a855f7;letter-spacing:.15em;font-weight:700;margin:0.9rem 0 0.4rem 0.3rem;text-transform:uppercase;\">Research Insights</div>",
            unsafe_allow_html=True
        )
        st.page_link("dashboard.py", label="🏠 Executive Overview", use_container_width=True)
        st.page_link("pages/1_Phase_Explorer.py", label="🗺️ 9-Phase Journey", use_container_width=True)
        st.page_link("pages/3_Benchmark_Arena.py", label="📊 Benchmark Arena", use_container_width=True)

        # ── Group 3: Engineering & Integrity ──────────────────────
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#06b6d4;letter-spacing:.15em;font-weight:700;margin:0.9rem 0 0.4rem 0.3rem;text-transform:uppercase;\">Engineering & Rigor</div>",
            unsafe_allow_html=True
        )
        st.page_link("pages/4_Architecture.py", label="🏗️ Model Architecture", use_container_width=True)
        st.page_link("pages/5_Training_Diagnostics.py", label="📈 Training & Diagnostics", use_container_width=True)
        st.page_link("pages/6_Research_Integrity.py", label="🔐 Research Integrity", use_container_width=True)

        # ── Status Card ───────────────────────────────────────────
        st.markdown(
            """
            <hr style="margin: 1.2rem 0 0.8rem 0;">
            <div style="padding:0.75rem 0.85rem;background:rgba(15,23,42,0.7);border:1px solid rgba(59,130,246,0.25);border-radius:12px;">
                <div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;font-weight:700;color:#22c55e;">
                    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e;"></span>
                    Phase 9 Checkpoint Active
                </div>
                <div style="font-size:0.68rem;color:#94a3b8;margin-top:4px;font-family:'JetBrains Mono',monospace;">
                    228K params · 94.4% routing
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
