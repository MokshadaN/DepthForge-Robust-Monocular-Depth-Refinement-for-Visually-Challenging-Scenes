"""
DepthForge Research Dashboard — Main Entry Point
Phase 9: Multi-Head Domain-Expert Depth Refinement
"""

import streamlit as st

st.set_page_config(
    page_title="DepthForge · Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from nav_utils import inject_theme, render_sidebar

inject_theme()
render_sidebar()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="hero-section">
    <div class="hero-tag">📡 DEPTHFORGE · PHASE 09 · MULTI-HEAD DOMAIN-EXPERT REFINEMENT</div>
    <div class="hero-title">When Vision Fails,<br>Refine Depth.</div>
    <div class="hero-subtitle">
        A lightweight multi-expert residual adapter targeting three visually adversarial domains —
        biological camouflage, transparent glass, and textureless surfaces — with autonomous soft domain
        routing and zero inference-time domain labels.
    </div>
    <div style="margin-top:1.5rem;display:flex;gap:0.8rem;flex-wrap:wrap;">
        <span class="badge badge-camo">🦎 Camouflage</span>
        <span class="badge badge-glass">🪟 Transparent Glass</span>
        <span class="badge badge-texture">🟫 Textureless</span>
        <span class="badge badge-phase">Phase 9 · Oct 2026</span>
    </div>
</div>
''', unsafe_allow_html=True)

st.page_link("pages/0_Live_Demo.py", label="🚀 Launch Live Demonstration (Upload Image & Compare Before / After Depth)", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Headline Metrics ────────────────────────────────────────────────────────────
st.markdown('''<div class="section-header"><div><div class="section-number">01 — OVERVIEW</div><div class="section-title">Headline Results</div></div></div>''', unsafe_allow_html=True)

cols = st.columns(4)
metrics = [
    ("blue","Test Images","450","150 per domain · held-out test set"),
    ("purple","Adapter Parameters","228.4K","~857 KB zipped · <5 ms GPU inference"),
    ("green","Mean AbsRel Gain","+2.9%","0.5848 → 0.5680 across 450 images"),
    ("orange","Routing Accuracy","94.4%","425/450 correctly classified · no labels"),
]
for col,(color,label,val,sub) in zip(cols,metrics):
    with col:
        st.markdown(f'''<div class="metric-tile {color}"><div class="metric-label">{label}</div><div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Three Domains ───────────────────────────────────────────────────────────────
st.markdown('''<div class="section-header"><div><div class="section-number">02 — DOMAINS</div><div class="section-title">Three Failure Modes, Three Expert Solutions</div></div></div>''', unsafe_allow_html=True)
d1,d2,d3 = st.columns(3)
domain_cards = [
    ("🦎","Biological Camouflage","#22c55e","Animals and objects whose texture mimics surroundings. MiDaS silhouette-flattens camouflaged subjects into the background plane.","0.6441","0.6197","+3.8%"),
    ("🪟","Transparent Glass","#06b6d4","Windows and glassware transmit background rays, tricking estimators into predicting distance behind the glass (200–400% overestimate).","0.5264","0.5018","+4.7%"),
    ("🟫","Textureless Surfaces","#a855f7","Monochromatic walls lack photometric gradients, causing depth warping, false curvature, and spherical bulging artifacts.","0.5838","0.5827","+0.2%"),
]
for col,(icon,title,color,desc,base,ref,gain) in zip([d1,d2,d3],domain_cards):
    with col:
        st.markdown(f'''
        <div class="df-card" style="border-top:3px solid {color};">
            <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
            <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;margin-bottom:1rem;">{desc}</div>
            <div style="display:flex;gap:0.5rem;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.65rem;color:#4b6080;text-transform:uppercase;letter-spacing:.1em;">Baseline → Refined</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:#94a3b8;margin-top:2px;">{base} → <span style="color:{color};font-weight:700;">{ref}</span></div>
                </div>
                <span class="gain-tag gain-positive">{gain} AbsRel</span>
            </div>
        </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase Timeline ──────────────────────────────────────────────────────────────
st.markdown('''<div class="section-header"><div><div class="section-number">03 — RESEARCH JOURNEY</div><div class="section-title">9-Phase Experimental Journey</div></div></div>''', unsafe_allow_html=True)
phases_preview = [
    (1,"Baseline","Road-scene detection + UniDAC depth","#3b82f6"),
    (2,"Failures","Discover 3 optical failure domains","#ef4444"),
    (3,"Benchmark","Assemble multi-domain dataset","#f59e0b"),
    (4,"Bake-Off","Compare 5 depth backbones","#8b5cf6"),
    (5,"First Adapter","Single-scale residual adapter","#10b981"),
    (6,"Multi-Scale","Dilated parallel branches d=1,2,4","#06b6d4"),
    (7,"Audit","Expose 4 critical experimental flaws","#f97316"),
    (8,"Verified","Corrected pipeline, clean results","#22c55e"),
    (9,"Multi-Head","Specialist experts + soft routing","#a855f7"),
]
cols_t = st.columns(9)
for col,(ph,name,desc,color) in zip(cols_t,phases_preview):
    with col:
        st.markdown(f'''
        <div style="background:rgba(13,31,60,0.8);border:1px solid {color}33;border-top:3px solid {color};border-radius:10px;padding:0.7rem 0.5rem;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:{color};font-weight:700;letter-spacing:.1em;">P{ph:02d}</div>
            <div style="font-family:'Outfit',sans-serif;font-size:0.78rem;font-weight:700;color:#f0f6ff;margin:3px 0;">{name}</div>
            <div style="font-size:0.62rem;color:#4b6080;line-height:1.4;">{desc}</div>
        </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Key Findings ────────────────────────────────────────────────────────────────
st.markdown('''<div class="section-header"><div><div class="section-number">04 — KEY FINDINGS</div><div class="section-title">Main Research Findings</div></div></div>''', unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    st.markdown('''<div class="df-card" style="border-left:3px solid #a855f7;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#a855f7;letter-spacing:.15em;margin-bottom:0.5rem;">ARCHITECTURAL INSIGHT</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;">Multi-Head Specialization Works</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.65;">Splitting features into domain expert heads outperforms a monolithic adapter. The glass expert learned to amplify subtle specular highlights; the camouflage expert recovered discrete silhouette boundaries.</div>
    </div>''', unsafe_allow_html=True)
with c2:
    st.markdown('''<div class="df-card" style="border-left:3px solid #06b6d4;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:.15em;margin-bottom:0.5rem;">ROUTING RESULT</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.6rem;">Autonomous Routing is Practical</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.65;">Differentiable soft classification achieves <strong style="color:#22d3ee;">94.4% routing accuracy</strong> without manual domain flags. At 228K total parameters, the shared trunk autonomously extracts domain-discriminative optical statistics.</div>
    </div>''', unsafe_allow_html=True)

st.markdown('''<div class="df-card" style="border-left:3px solid #f59e0b;background:linear-gradient(135deg,rgba(245,158,11,0.05),rgba(13,31,60,0.8));">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#f59e0b;letter-spacing:.15em;margin-bottom:0.5rem;">⚠️ HONEST TRADE-OFFS</div>
    <div style="font-size:0.85rem;color:#94a3b8;line-height:1.65;">While <strong style="color:#fde68a;">AbsRel improves across all domains</strong>, SILog slightly worsens on camouflage and glass. The adapter introduces sharp depth discontinuities that diverge from DPT-Large's smoother pseudo-GT in log-ratio space — even though visual quality is substantially superior. The Benchmark Arena page exposes these trade-offs explicitly.</div>
</div>''', unsafe_allow_html=True)

st.markdown('''
<div style="text-align:center;padding:1.5rem;border-top:1px solid rgba(59,130,246,0.15);margin-top:2rem;">
    <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DepthForge</div>
    <div style="font-size:0.75rem;color:#4b6080;margin-top:4px;">Executed October 2026 · NVIDIA Tesla T4 · PyTorch 2.11 · Kaggle Cloud<br><span style="color:#334155;">All metrics computed on 450 held-out test images. DPT-Large outputs are pseudo-targets, not sensor-derived ground truth.</span></div>
</div>''', unsafe_allow_html=True)
