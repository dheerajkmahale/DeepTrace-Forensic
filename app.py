"""Streamlit prototype user interface for the deepfake detector.

Provides an interactive web interface for uploading videos, inspecting
preprocessing and CNN-LSTM model predictions, and visualizing results.

Note:
This interface is a demonstration prototype running on the synthetic demo model.
Real Celeb-DF evaluation has not yet been performed.
"""

import os
import shutil
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import Config
from predict import predict_video

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector | CNN-LSTM Prototype",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, academic/research-grade aesthetic
st.markdown(
    """
    <style>
    /* Main container and font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    /* Prototype Notice Banner */
    .prototype-banner {
        background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
        border: 1px solid #bfdbfe;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 24px;
    }
    .prototype-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1e40af;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .prototype-text {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.45;
        margin: 0;
    }

    /* Result Card Styles */
    .result-card {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }
    .result-card-fake {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .result-card-real {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
    }
    .verdict-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }
    .metric-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 4px 6px;
    }

    /* Probability bar container */
    .prob-bar-container {
        margin-top: 14px;
        margin-bottom: 14px;
        background-color: #e2e8f0;
        border-radius: 8px;
        height: 28px;
        overflow: hidden;
        display: flex;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
    }
    .prob-bar-real {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: width 0.6s ease;
    }
    .prob-bar-fake {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        color: white;
        font-weight: 600;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: width 0.6s ease;
    }

    /* Model metadata pill */
    .meta-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }

    /* Footer disclaimer */
    .footer-disclaimer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.82rem;
        color: #94a3b8;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "outputs/demo/best_model.keras"
UPLOAD_DIR = "data/demo/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Sidebar: System Controls & Model Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    model_exists = os.path.exists(DEFAULT_MODEL_PATH)
    if model_exists:
        st.success(f"✓ Prototype Model Loaded\n`{DEFAULT_MODEL_PATH}`", icon="✅")
    else:
        st.error(
            f"Model not found at `{DEFAULT_MODEL_PATH}`.\n\n"
            "Please run `python demo.py` in your terminal to build the prototype model.",
            icon="⚠️",
        )

    st.markdown("---")
    st.markdown("#### 🔬 Analysis Settings")

    threshold = st.slider(
        "Decision Threshold (P(fake) cutoff)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Videos with P(fake) ≥ threshold are classified as FAKE.",
    )

    no_face_detect = st.checkbox(
        "Center Crop (No Face Detection)",
        value=True,
        help=(
            "Performs center square crop instead of Haar face cascade. "
            "Recommended when evaluating synthetic demonstration videos."
        ),
    )

    st.markdown("---")
    st.markdown("#### 📦 Demo Presets")
    preset_choice = st.radio(
        "Quick Test Samples:",
        options=[
            "None (Upload my own video)",
            "Synthetic Fake Sample (synth_fake_00.mp4)",
            "Synthetic Real Sample (synth_real_15.mp4)",
        ],
        index=0,
    )

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Deepfake Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Spatio-Temporal CNN-LSTM Video Deepfake Detection Pipeline</div>',
    unsafe_allow_html=True,
)

# Prominent Prototype Status Banner
st.markdown(
    """
    <div class="prototype-banner">
        <div class="prototype-title">⚠️ PROTOTYPE — Demonstration Model Trained on Synthetic Data</div>
        <p class="prototype-text">
            This web interface demonstrates the end-to-end execution of the video preprocessing, temporal sequence extraction,
            and CNN-LSTM inference pipeline.
            <b>Results shown by this prototype must not be interpreted as validated real-world deepfake detection performance.</b>
            Academic evaluation on the official Celeb-DF v2 benchmark requires academic access approval.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model Overview Metrics
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Model Architecture", value="CNN + LSTM", delta="Hybrid")
with col2:
    st.metric(label="Sequence Length", value="10 Frames", delta="Temporal")
with col3:
    st.metric(label="Clips Analysed", value="3 Segments", delta="Uniform")
with col4:
    st.metric(label="Active Model", value="demo/best_model", delta="Prototype")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Video Input Section (Upload or Preset)
# -----------------------------------------------------------------------------
target_video_path: Optional[str] = None
video_display_name: str = ""

st.markdown("### 📤 Video Input")

left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    if preset_choice == "Synthetic Fake Sample (synth_fake_00.mp4)":
        preset_path = "data/demo/raw/fake/synth_fake_00.mp4"
        if os.path.exists(preset_path):
            target_video_path = preset_path
            video_display_name = "synth_fake_00.mp4 (Synthetic FAKE Sample)"
            st.info(f"Loaded preset video: `{video_display_name}`")
        else:
            st.warning("Preset video not found. Please run `python demo.py` first.")

    elif preset_choice == "Synthetic Real Sample (synth_real_15.mp4)":
        preset_path = "data/demo/raw/real/synth_real_15.mp4"
        if os.path.exists(preset_path):
            target_video_path = preset_path
            video_display_name = "synth_real_15.mp4 (Synthetic REAL Sample)"
            st.info(f"Loaded preset video: `{video_display_name}`")
        else:
            st.warning("Preset video not found. Please run `python demo.py` first.")

    else:
        uploaded_file = st.file_uploader(
            "Choose a video file to evaluate (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video to analyze with the CNN-LSTM deepfake detector.",
        )

        if uploaded_file is not None:
            # Save uploaded video safely to data/demo/uploads/
            target_video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(target_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_display_name = uploaded_file.name
            st.success(f"Video uploaded successfully: `{video_display_name}`")

    # Analyze Button
    analyze_clicked = st.button(
        "🔍 Analyze Video",
        type="primary",
        disabled=(target_video_path is None or not model_exists),
        use_container_width=True,
    )

with right_col:
    st.markdown("#### Video Preview")
    if target_video_path and os.path.exists(target_video_path):
        try:
            st.video(target_video_path)
        except Exception:
            st.caption("Video preview format not supported by browser; analysis will still proceed.")
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #cbd5e1; border-radius: 8px; padding: 40px; text-align: center; color: #94a3b8;">
                No video selected. Upload a video file or pick a preset demo sample to preview.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Video Analysis Execution
# -----------------------------------------------------------------------------
if analyze_clicked and target_video_path:
    st.markdown("---")
    st.markdown("### 📊 Detection Analysis")

    # Progress bar and status feedback
    status_box = st.status("Initializing deepfake detection pipeline...", expanded=True)

    with status_box:
        st.write("1. 📥 Loading input video...")
        time.sleep(0.2)

        st.write("2. 🎞️ Extracting temporal segments (clips_per_video = 3)...")
        time.sleep(0.2)

        if no_face_detect:
            st.write("3. ✂️ Performing center-crop normalization (synthetic mode enabled)...")
        else:
            st.write("3. 👤 Detecting and cropping facial regions using OpenCV Haar cascade...")
        time.sleep(0.2)

        st.write("4. 🧠 Generating uint8 temporal sequences of shape (10, 128, 128, 3)...")
        time.sleep(0.2)

        st.write(f"5. ⚡ Running CNN-LSTM model inference with `{DEFAULT_MODEL_PATH}`...")

        try:
            result = predict_video(
                video_path=target_video_path,
                model_path=DEFAULT_MODEL_PATH,
                threshold=threshold,
                seq_len=10,
                img_size=128,
                clips_per_video=3,
                face_margin=0.25,
                no_face_detect=no_face_detect,
            )
            st.write("6. 📈 Aggregating clip-level probabilities into video-level score...")
            time.sleep(0.2)
            status_box.update(label="Analysis Complete!", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="Analysis Failed", state="error", expanded=True)
            st.error(f"Error during video processing: {e}")
            st.stop()

    # -------------------------------------------------------------------------
    # Results Display Card
    # -------------------------------------------------------------------------
    p_fake = result.fake_probability
    p_real = result.real_probability
    verdict = result.verdict
    num_clips = result.clips_analyzed

    if verdict == "FAKE":
        card_class = "result-card-fake"
        verdict_icon = "⚠️"
        verdict_text = "FAKE DETECTED"
    else:
        card_class = "result-card-real"
        verdict_icon = "✓"
        verdict_text = "REAL VIDEO"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="verdict-title">{verdict_icon} {verdict_text}</div>
            <div>
                <span class="metric-chip">Probability of Fake: <b>{p_fake * 100:.1f}%</b></span>
                <span class="metric-chip">Probability of Real: <b>{p_real * 100:.1f}%</b></span>
                <span class="metric-chip">Clips Analysed: <b>{num_clips}</b></span>
                <span class="metric-chip">Decision Cutoff: <b>{threshold:.2f}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Clean Probability Visualization Bar
    # -------------------------------------------------------------------------
    st.markdown("#### Probability Distribution")
    real_pct = max(0.0, min(100.0, p_real * 100.0))
    fake_pct = max(0.0, min(100.0, p_fake * 100.0))

    st.markdown(
        f"""
        <div class="prob-bar-container">
            <div class="prob-bar-real" style="width: {real_pct:.1f}%;">
                REAL {real_pct:.1f}%
            </div>
            <div class="prob-bar-fake" style="width: {fake_pct:.1f}%;">
                FAKE {fake_pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Individual clip breakdown
    if result.clip_probabilities:
        with st.expander("🔍 View Per-Clip Breakdown", expanded=False):
            clip_df = pd.DataFrame(
                {
                    "Clip Index": [f"Segment {i+1}" for i in range(len(result.clip_probabilities))],
                    "P(Fake)": [f"{p:.4f}" for p in result.clip_probabilities],
                    "P(Real)": [f"{1.0 - p:.4f}" for p in result.clip_probabilities],
                    "Clip Verdict": ["FAKE" if p >= threshold else "REAL" for p in result.clip_probabilities],
                }
            )
            st.dataframe(clip_df, use_container_width=True)

# -----------------------------------------------------------------------------
# Technical Details Expander
# -----------------------------------------------------------------------------
with st.expander("🛠️ Technical Details & System Architecture", expanded=False):
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown(
            """
            - **Model Architecture**: Hybrid Spatio-Temporal CNN + LSTM
            - **Spatial Backbone**: Lightweight 4-Block CNN (`32`, `64`, `128`, `128` filters)
            - **Normalization**: `BatchNormalization` with calibrated `momentum=0.9`
            - **Temporal Unit**: LSTM (`128` units) with sequence length = `10` frames
            """
        )
    with tcol2:
        st.markdown(
            """
            - **Input Resolution**: `128 x 128` RGB pixels (`uint8` tensor)
            - **Temporal Segments**: `3` equal-duration clips per video
            - **Face Extractor**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
            - **Aggregation Strategy**: Mean pooling of clip probabilities across video
            """
        )

# -----------------------------------------------------------------------------
# Bottom Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-disclaimer">
        Prototype only. The current demonstration model is trained/evaluated on synthetic demonstration data.
        Real Celeb-DF validation has not yet been performed.
    </div>
    """,
    unsafe_allow_html=True,
)
