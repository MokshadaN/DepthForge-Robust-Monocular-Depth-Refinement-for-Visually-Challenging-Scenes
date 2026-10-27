import streamlit as st
import os

import re, os as _os
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
    "margin-bottom:0.3rem;'>📈 Training & Diagnostics</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:0.88rem;color:#94a3b8;margin-bottom:1.5rem;max-width:800px;line-height:1.65;'>"
    "Training dynamics, dataset composition, routing reliability, error distributions, and "
    "best/worst case analysis for the Phase 9 multi-head adapter (60 epochs, Kaggle T4).</div>",
    unsafe_allow_html=True
)

# ── Training convergence table ──
st.markdown("#### Training Convergence Summary")
import pandas as pd

st.set_page_config(page_title="Training Diagnostics - DepthForge", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
epochs = [1, 10, 20, 30, 40, 50, 55, 60]
train_loss = [0.44829, 0.40680, 0.39463, 0.38327, 0.37610, 0.37412, 0.37326, 0.37545]
val_silog  = [0.35473, 0.35405, 0.35330, 0.35226, 0.35028, 0.34988, 0.34947, 0.34956]
val_absrel = [0.5584, 0.5525, 0.5519, 0.5522, 0.5493, 0.5487, 0.5483, 0.5492]
lr_vals    = [1.50e-3, 1.40e-3, 1.13e-3, 7.53e-4, 3.79e-4, 1.09e-4, 3.05e-5, 5.00e-6]
notes      = ["Initial convergence", "Trunk stabilized", "Routing >85%",
              "Cosine plateau", "Grad loss settling", "Fine-tuning experts",
              "🏆 Best Checkpoint", "Final converged"]
df_conv = pd.DataFrame({
    "Epoch": epochs, "Train Loss": train_loss, "Val SILog": val_silog,
    "Val AbsRel": val_absrel, "LR": [f"{lr:.2e}" for lr in lr_vals], "Status": notes
})
st.dataframe(df_conv, use_container_width=True, hide_index=True)

# ── Visualization panels ──
st.markdown("---")
st.markdown("#### Diagnostic Panels")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Training Dynamics", "Dataset Balance", "Error Distributions",
    "Routing Analysis", "Scatter Plot", "Best & Worst"
])

panels = {
    "Training Dynamics": [
        ("phase9/extended_visualizations/01_training_dynamics.png",
         "Training loss, validation SILog, and learning rate over 60 epochs"),
        ("phase9/results_extracted/results/phase9_training_curves.png",
         "Training curves from Kaggle run"),
    ],
    "Dataset Balance": [
        ("phase9/extended_visualizations/02_dataset_split_balance.png",
         "Train/Val/Test counts per domain — all balanced at 700/150/150"),
    ],
    "Error Distributions": [
        ("phase9/extended_visualizations/05_per_image_error_boxplots.png",
         "Per-image AbsRel error boxplots: baseline vs refined per domain"),
        ("phase9/extended_visualizations/07_improvement_histogram.png",
         "Distribution of per-image improvement (positive=better)"),
        ("phase9/extended_visualizations/04_relative_error_reduction.png",
         "Relative error reduction by domain"),
    ],
    "Routing Analysis": [
        ("phase9/extended_visualizations/08_routing_confusion_matrix.png",
         "Routing confusion matrix — 94.4% overall accuracy"),
        ("phase9/extended_visualizations/09_routing_confidence_by_correctness.png",
         "Routing confidence: correctly vs incorrectly classified scenes"),
        ("phase9/local_analysis/04_routing_analysis.png",
         "Routing analysis overview"),
    ],
    "Scatter Plot": [
        ("phase9/extended_visualizations/06_paired_absrel_scatter.png",
         "Per-image paired scatter: baseline AbsRel vs refined AbsRel (points above diagonal = improvement)"),
        ("phase9/extended_visualizations/10_pixelwise_teacher_agreement.png",
         "Pixelwise teacher agreement: Phase 9 refined vs DPT-Large pseudo-target"),
    ],
    "Best & Worst": [
        ("phase9/extended_visualizations/14_best_and_worst_images.png",
         "Best improvements and worst regressions on the test set"),
        ("phase9/extended_visualizations/12_center_row_profiles.png",
         "Center-row depth profile comparison: baseline vs refined"),
    ],
}

for tab, (name, imgs) in zip([tab1,tab2,tab3,tab4,tab5,tab6], panels.items()):
    with tab:
        for img_rel, caption in imgs:
            img_path = os.path.join(BASE, img_rel)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption=caption)
            else:
                st.warning(f"Image not found: {img_rel}")

# ── Routing accuracy ──
st.markdown("---")
st.markdown("#### Routing Accuracy Summary")
routing_data = {
    "Domain": ["Camouflage", "Glass", "Textureless", "Total"],
    "Correctly Classified": [136, 140, 149, 425],
    "Total Test Samples": [150, 150, 150, 450],
    "Accuracy": ["90.7%", "93.3%", "99.3%", "94.4%"],
    "Primary Misclassification": [
        "Confused with Glass (8.0%)",
        "Confused with Textureless (6.7%)",
        "Confused with Glass (0.7%)",
        "—"
    ]
}
st.dataframe(pd.DataFrame(routing_data), use_container_width=True, hide_index=True)
