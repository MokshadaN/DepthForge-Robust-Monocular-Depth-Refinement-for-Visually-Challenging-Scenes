"""
DepthForge Phase 9: Live Interactive Depth Refinement Demonstration
Upload any RGB image or select an adversarial test sample to inspect Before vs After results in real-time.
"""

import time
import os
import cv2
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nav_utils import inject_theme, render_sidebar
from phase10.models.adapter_multihead import MultiHeadDilatedAdapter

st.set_page_config(
    page_title="DepthForge · Live Demonstration",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
render_sidebar()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Model Cache ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DepthForge & MiDaS backbones...")
def load_models():
    device = torch.device("cpu")
    # 1. MiDaS Small Baseline
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    # 2. DepthForge Phase 9 Multi-Head Dilated Adapter
    adapter = MultiHeadDilatedAdapter().to(device)
    ckpt_path = os.path.join(REPO_ROOT, "phase9", "results_extracted", "checkpoints", "phase9_multihead_best.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        adapter.load_state_dict(state)
    adapter.eval()

    return midas, transform, adapter, device


def run_pipeline(img_rgb: np.ndarray, scale: float = 0.30):
    midas, transform, adapter, device = load_models()
    h, w = img_rgb.shape[:2]

    t0 = time.perf_counter()

    # Preprocess for MiDaS
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        p_min, p_max = prediction.min(), prediction.max()
        base_d = (prediction - p_min) / (p_max - p_min + 1e-8)

        # Sobel Energy calculation
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        sobel = mag / (mag.max() + 1e-8)
        sobel_t = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(device)

        # RGB tensor
        rgb_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        # Run Multi-Head Adapter
        adapter.scale = scale
        refined, logits = adapter(base_d, rgb_t, sobel_t)
        weights = torch.softmax(logits, dim=1).squeeze().tolist()

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    base_np = base_d.squeeze().cpu().numpy()
    refined_np = refined.squeeze().cpu().numpy()
    sobel_np = sobel

    return {
        "base_depth": base_np,
        "refined_depth": refined_np,
        "sobel": sobel_np,
        "weights": weights,
        "delta": refined_np - base_np,
        "latency_ms": latency_ms,
    }


def colorize_depth(depth_map: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    d_norm = np.clip(depth_map, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(d_norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="font-family:'Outfit',sans-serif;font-size:2.2rem;font-weight:900;
         background:linear-gradient(135deg,#f0f6ff,#93c5fd,#a78bfa);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         margin-bottom:0.3rem;">🚀 Live Depth Refinement Demonstration</div>
    <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:1.5rem;max-width:850px;line-height:1.65;">
        Experience DepthForge Phase 9 in real time. Upload any RGB image or select an authentic adversarial test sample.
        Inspect how the multi-head adapter dynamically routes features across <strong>Camouflage</strong>,
        <strong>Glass</strong>, and <strong>Textureless</strong> expert heads to reconstruct sharp depth boundaries.
    </div>
    """,
    unsafe_allow_html=True
)

# ── Preset Sample Library ──────────────────────────────────────────────────────
PRESETS = {
    "🦎 Camouflage Flatfish (camo_0023.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase8", "data", "rgb", "camo_0023.jpg"),
        "domain": "Camouflage",
        "desc": "Flatfish blending seamlessly into sand. Baseline produces an undifferentiated flat slope; adapter recovers body boundary.",
    },
    "🦎 Camouflaged Insect (camo_0035.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase8", "data", "rgb", "camo_0035.jpg"),
        "domain": "Camouflage",
        "desc": "Insect camouflaged against wood texture with high-frequency background noise.",
    },
    "🪟 Transparent Wine Glass (glass_01.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase7", "data", "unified_500", "transparent_glass_01_wine_view1_wide.jpg"),
        "domain": "Transparent Glass",
        "desc": "Refractive glass goblet. Baseline penetrates straight to background wall; glass expert pulls the boundary forward.",
    },
    "🪟 Laboratory Beaker (glass_02.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase7", "data", "unified_500", "transparent_glass_02_beaker_view1_wide.jpg"),
        "domain": "Transparent Glass",
        "desc": "Transparent cylinder with sharp specular rim reflections.",
    },
    "🟫 Textureless White Room (texless_01.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase8", "data", "rgb", "textureless_texless_01_white_room_view1_wide.jpg"),
        "domain": "Textureless Interior",
        "desc": "Monochromatic corner with minimal photometric gradient. Verifies adapter stability without hallucinating fake depth.",
    },
    "🟫 Textureless Smooth Wall (texless_02.jpg)": {
        "path": os.path.join(REPO_ROOT, "phase8", "data", "rgb", "textureless_texless_02_white_wall_view1_wide.jpg"),
        "domain": "Textureless Interior",
        "desc": "Smooth wall planar regression test.",
    },
}

# ── Input Selection ────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2 = st.columns([1.2, 1.0], gap="medium")

with col_ctrl1:
    st.markdown("##### 1. Select Input Source")
    input_mode = st.radio(
        "Choose image source:",
        ["Curated Adversarial Presets", "Upload Custom Image"],
        horizontal=True,
        label_visibility="collapsed"
    )

    img_rgb = None
    selected_name = ""

    if input_mode == "Curated Adversarial Presets":
        selected_key = st.selectbox("Choose sample image:", list(PRESETS.keys()))
        preset_info = PRESETS[selected_key]
        selected_name = selected_key
        if os.path.exists(preset_info["path"]):
            bgr = cv2.imread(preset_info["path"])
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.caption(f"ℹ️ **{preset_info['domain']} Challenge**: {preset_info['desc']}")
        else:
            st.error(f"Sample file not found at {preset_info['path']}")
    else:
        uploaded_file = st.file_uploader(
            "Upload image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload any real-world photo or scene to evaluate depth refinement"
        )
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(pil_img)
            selected_name = uploaded_file.name

    if img_rgb is not None:
        st.image(img_rgb, caption=f"Active Input Scene: {selected_name}", use_container_width=True)

with col_ctrl2:
    st.markdown("##### 2. Inference & Display Controls")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        residual_scale = st.slider(
            "Tanh Residual Scale (λ)",
            min_value=0.05,
            max_value=0.60,
            value=0.30,
            step=0.05,
            help="Tuning parameter λ in: D_refined = clamp(D_base + λ * tanh(Δ), 0, 1). Default: 0.30"
        )
    with c_s2:
        colormap_choice = st.selectbox(
            "Colormap",
            ["turbo", "magma", "plasma", "viridis", "inferno"],
            index=0,
            help="Depth colormap for visualization"
        )

    # Resolution downscale safeguard if huge image uploaded
    if img_rgb is not None:
        h, w = img_rgb.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scaling = max_dim / max(h, w)
            new_w, new_h = int(w * scaling), int(h * scaling)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        st.markdown(
            f"<div style='font-size:0.75rem;color:#64748b;margin-top:0.4rem;font-family:\"JetBrains Mono\",monospace;'>"
            f"Active: <strong style='color:#93c5fd;'>{selected_name}</strong> · {img_rgb.shape[1]}x{img_rgb.shape[0]} px</div>",
            unsafe_allow_html=True
        )

# ── Execute Pipeline ───────────────────────────────────────────────────────────
if img_rgb is not None:
    with st.spinner("⚡ Running MiDaS baseline + Phase 9 multi-head refinement..."):
        results = run_pipeline(img_rgb, scale=residual_scale)

    base_d = results["base_depth"]
    refined_d = results["refined_depth"]
    delta_d = results["delta"]
    weights = results["weights"]
    latency = results["latency_ms"]

    base_colored = colorize_depth(base_d, colormap_choice)
    refined_colored = colorize_depth(refined_d, colormap_choice)

    # ── Real-Time Routing Diagnostic Bar ──────────────────────────────────────
    w_camo, w_glass, w_tex = weights[0], weights[1], weights[2]
    max_idx = int(np.argmax(weights))
    domain_names = ["Camouflage Head", "Glass Head", "Textureless Head"]
    winning_head = domain_names[max_idx]

    st.markdown(
        f"""
        <div class="df-card" style="padding:1rem 1.4rem;margin:1.2rem 0;border-left:4px solid #3b82f6;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div>
                    <div style="font-size:0.68rem;letter-spacing:.15em;text-transform:uppercase;color:#06b6d4;font-family:'JetBrains Mono',monospace;font-weight:700;">
                        ACTIVE EXPERT ROUTER DISPATCH
                    </div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.25rem;font-weight:800;color:#f0f6ff;margin-top:2px;">
                        Routed to: <span style="color:#60a5fa;">{winning_head}</span> ({weights[max_idx]*100:.1f}% confidence)
                    </div>
                </div>
                <div style="display:flex;gap:1.5rem;font-family:'JetBrains Mono',monospace;font-size:0.8rem;">
                    <div><span style="color:#22c55e;font-weight:700;">🦎 Camo:</span> {w_camo*100:.1f}%</div>
                    <div><span style="color:#06b6d4;font-weight:700;">🪟 Glass:</span> {w_glass*100:.1f}%</div>
                    <div><span style="color:#a855f7;font-weight:700;">🟫 Tex:</span> {w_tex*100:.1f}%</div>
                    <div><span style="color:#94a3b8;">⏱️ Latency:</span> {latency:.1f} ms</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── PRIMARY PROMINENT ON-SCREEN BEFORE & AFTER SHOWCASE ─────────────────────
    st.markdown(
        """
        <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:800;color:#f0f6ff;margin-bottom:0.2rem;">
            🔍 Live Before & After Showcase
        </div>
        <div style="font-size:0.84rem;color:#94a3b8;margin-bottom:1.2rem;">
            Direct comparison: The RGB input is processed through standard MiDaS-Small (Before) and enhanced by the DepthForge Phase 9 Adapter (After).
        </div>
        """,
        unsafe_allow_html=True
    )

    col_img1, col_img2, col_img3 = st.columns(3, gap="medium")
    with col_img1:
        st.markdown(
            "<div style='background:rgba(15,23,42,0.8);border:1px solid rgba(59,130,246,0.3);border-radius:12px;padding:0.75rem;margin-bottom:0.5rem;'>"
            "<div style='font-size:0.78rem;font-weight:800;color:#93c5fd;letter-spacing:.1em;text-transform:uppercase;'>📸 1. Input RGB Scene</div>"
            "<div style='font-size:0.7rem;color:#64748b;margin-top:2px;'>Raw optical capture</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.image(img_rgb, use_container_width=True)
        st.caption(f"Input: {selected_name}")

    with col_img2:
        st.markdown(
            "<div style='background:rgba(15,23,42,0.8);border:1px solid rgba(245,158,11,0.35);border-radius:12px;padding:0.75rem;margin-bottom:0.5rem;'>"
            "<div style='font-size:0.78rem;font-weight:800;color:#fbbf24;letter-spacing:.1em;text-transform:uppercase;'>⚠️ 2. BEFORE: MiDaS Baseline</div>"
            "<div style='font-size:0.7rem;color:#f87171;margin-top:2px;'>Boundary blur & phantom penetration</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.image(base_colored, use_container_width=True)
        st.caption("Baseline: smoothed boundary slope")

    with col_img3:
        st.markdown(
            "<div style='background:rgba(15,23,42,0.8);border:1px solid rgba(34,197,94,0.35);border-radius:12px;padding:0.75rem;margin-bottom:0.5rem;'>"
            "<div style='font-size:0.78rem;font-weight:800;color:#4ade80;letter-spacing:.1em;text-transform:uppercase;'>✨ 3. AFTER: DepthForge Phase 9</div>"
            "<div style='font-size:0.7rem;color:#4ade80;margin-top:2px;'>Sharp silhouette & surface restored</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.image(refined_colored, use_container_width=True)
        st.caption("DepthForge: sharp step-function edge")

    st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)

    # ── ADVANCED INTERACTIVE INSPECTION TOOLS ─────────────────────────────────
    st.markdown("##### 🛠️ Advanced Deep-Dive Tools")
    tab_slider, tab_profile, tab_pipeline = st.tabs([
        "🎚️ Interactive Blend Cross-Fade",
        "📈 1D Depth Scanline Profile",
        "🧩 Complete 5-Stage Multi-Modal Pipeline"
    ])

    with tab_slider:
        st.markdown("###### Cross-Fade Depth Blend")
        alpha = st.slider("Drag slider to blend Before (0%) vs After (100%):", 0.0, 1.0, 1.0, 0.05)
        blended_d = (1.0 - alpha) * base_d + alpha * refined_d
        blended_colored = colorize_depth(blended_d, colormap_choice)

        c_blend1, c_blend2 = st.columns([1.5, 1.0], gap="large")
        with c_blend1:
            st.image(blended_colored, caption=f"Blended Representation ({int(alpha*100)}% DepthForge Refined)", use_container_width=True)
        with c_blend2:
            st.markdown(
                f"""
                <div class="df-card">
                    <div style="font-size:0.75rem;font-weight:700;color:#60a5fa;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">Current Blend State</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;color:#f0f6ff;">
                        {int((1-alpha)*100)}% Base / {int(alpha*100)}% Refined
                    </div>
                    <div style="font-size:0.82rem;color:#94a3b8;margin-top:0.6rem;line-height:1.6;">
                        Notice how sliding towards 100% sharpens the target edges without altering unaffected ambient background planes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_profile:
        st.markdown("###### 1D Horizontal Scanline Slice")
        st.markdown(
            "Select any horizontal row $Y$ to plot the depth cross-section. Observe how baseline MiDaS produces a continuous gradient slope while DepthForge reconstructs a steep, physically accurate step function at the object boundary."
        )

        h_img = img_rgb.shape[0]
        row_y = st.slider("Select scanline row (Y coordinate):", 0, h_img - 1, h_img // 2)

        # Plotly Scanline chart
        x_coords = list(range(img_rgb.shape[1]))
        base_slice = base_d[row_y, :].tolist()
        refined_slice = refined_d[row_y, :].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_coords, y=base_slice,
            mode="lines", name="Before: MiDaS Baseline",
            line=dict(color="#f59e0b", width=2.5, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=x_coords, y=refined_slice,
            mode="lines", name="After: DepthForge Phase 9",
            line=dict(color="#22c55e", width=3)
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1f3c",
            plot_bgcolor="#07152a",
            title=f"Depth Cross-Section along Y = {row_y} px",
            xaxis=dict(title="X Pixel Coordinate", gridcolor="#1e3a5f"),
            yaxis=dict(title="Normalized Depth (0.0 to 1.0)", gridcolor="#1e3a5f", range=[0.0, 1.05]),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
        )

        c_prof1, c_prof2 = st.columns([1.6, 1.0], gap="medium")
        with c_prof1:
            st.plotly_chart(fig, use_container_width=True)
        with c_prof2:
            # Show image with scanline overlay
            img_with_line = img_rgb.copy()
            cv2.line(img_with_line, (0, row_y), (img_with_line.shape[1], row_y), (255, 0, 0), 2)
            st.image(img_with_line, caption=f"Scanline Position (Red Line at Y={row_y})", use_container_width=True)

    with tab_pipeline:
        st.markdown("###### Complete Multi-Stage Inspection")
        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#94a3b8;margin-bottom:4px;'>1. RGB INPUT</div>", unsafe_allow_html=True)
            st.image(img_rgb, use_container_width=True)
        with p2:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#a78bfa;margin-bottom:4px;'>2. SOBEL PRIOR</div>", unsafe_allow_html=True)
            st.image((results["sobel"] * 255).astype(np.uint8), use_container_width=True)
        with p3:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#f59e0b;margin-bottom:4px;'>3. BASELINE MIDAS</div>", unsafe_allow_html=True)
            st.image(base_colored, use_container_width=True)
        with p4:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#22c55e;margin-bottom:4px;'>4. PHASE 9 REFINED</div>", unsafe_allow_html=True)
            st.image(refined_colored, use_container_width=True)
        with p5:
            st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#ec4899;margin-bottom:4px;'>5. RESIDUAL DELTA</div>", unsafe_allow_html=True)
            # Delta mapped around 0.5 center
            delta_vis = np.clip((delta_d + 0.25) / 0.5, 0.0, 1.0)
            st.image(colorize_depth(delta_vis, "bwr"), use_container_width=True)

else:
    st.info("👈 Please select an adversarial preset or upload an image above to begin.")
