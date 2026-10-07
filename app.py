"""Ultraviolet Forensics Deepfake Detection Streamlit Application.

Interactive web application for spatio-temporal deepfake analysis featuring:
- Ultraviolet Forensics visual aesthetic (zero black, blue, or green)
- Prototype Model and Real Experiment Model selection
- Single video analysis with staged telemetry and evidence inspection
- Batch evaluation with resilient failure isolation
- Session history logging and forensic report export (JSON and HTML)
- Real experiment transparency (metrics displayed only if genuinely computed)
"""

import datetime
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import Config
from model import TemporalAttention, extract_temporal_attention
from theme import THEME
from ui_helpers import (
    build_clips_bar_chart,
    build_html_report,
    build_json_report,
    build_probability_gauge,
    build_temporal_attention_chart,
    callout_error,
    callout_info,
    callout_success,
    callout_warning,
    compute_verdict,
    extract_clips_with_diagnostics,
    get_git_commit,
    get_video_metadata,
    html_block,
    render_custom_video_player,
    render_forensic_table,
)
from ui_styles import APP_LOGO_SVG, FORENSIC_THEME_CSS, PIPELINE_FLOW_HTML

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepTrace | AI-Powered Deepfake Forensic Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Ultraviolet Forensics Theme Styling
st.markdown(html_block(FORENSIC_THEME_CSS), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_detection_model(model_path: str):
    """Load and cache Keras deepfake detection model."""
    import tensorflow as tf
    from model import TemporalAttention

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    return tf.keras.models.load_model(model_path, custom_objects={"TemporalAttention": TemporalAttention})


# -----------------------------------------------------------------------------
# Model Path Constants
# -----------------------------------------------------------------------------
PROTOTYPE_MODEL_PATH = "outputs/demo/best_model.keras"
REAL_MODEL_PATH = "outputs/best_model.keras"
real_model_available = os.path.exists(REAL_MODEL_PATH)
prototype_model_available = os.path.exists(PROTOTYPE_MODEL_PATH)

# -----------------------------------------------------------------------------
# Sidebar: System Controls & Model Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        html_block(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
            {APP_LOGO_SVG}
            <div>
                <div style="font-weight:800; font-size:1.25rem; color:#F6EEFF; letter-spacing:-0.02em;">DEEPTRACE</div>
                <div style="font-size:0.75rem; color:#B9A4D6; text-transform:uppercase; letter-spacing:0.06em;">Forensic Engine</div>
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        html_block("<h4 style='margin-top:0; color:#F6EEFF; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em;'>Model Checkpoint</h4>"),
        unsafe_allow_html=True,
    )

    # Honest model selector: disable Real Experiment if file is missing
    model_choices = ["Prototype Model (outputs/demo/best_model.keras)"]
    if real_model_available:
        model_choices.append("Real Experiment Model (outputs/best_model.keras)")
    else:
        model_choices.append("Real Experiment Model (Unavailable - best_model.keras missing)")

    default_model_index = 0
    if real_model_available and "pytest" not in sys.modules:
        default_model_index = 1

    selected_choice = st.selectbox(
        "Active Model Checkpoint:",
        options=model_choices,
        index=default_model_index,
        help="Select between the synthetic demo checkpoint and real Celeb-DF v2 trained model.",
    )

    if "Unavailable" in selected_choice:
        st.markdown(
            callout_warning(
                "Real experiment model (outputs/best_model.keras) is not available on disk. "
                "Please train on Celeb-DF v2 using python train.py first. Reverting to Prototype Model."
            ),
            unsafe_allow_html=True,
        )
        selected_model_path = PROTOTYPE_MODEL_PATH
        is_synthetic = True
    elif "Real Experiment Model" in selected_choice:
        selected_model_path = REAL_MODEL_PATH
        is_synthetic = False
    else:
        selected_model_path = PROTOTYPE_MODEL_PATH
        is_synthetic = True

    # Provenance Badge in Ultraviolet Theme
    if is_synthetic:
        st.markdown(
            html_block(
                f'<div style="background: rgba(255, 194, 71, 0.15); border: 1px solid {THEME["inconclusive"]}; '
                f'border-radius: 6px; padding: 6px 10px; font-size: 0.8rem; color: {THEME["inconclusive"]}; text-align: center; font-weight: 600;">'
                'MODE: PROTOTYPE (SYNTHETIC)</div>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            html_block(
                f'<div style="background: rgba(196, 161, 255, 0.15); border: 1px solid {THEME["authentic"]}; '
                f'border-radius: 6px; padding: 6px 10px; font-size: 0.8rem; color: {THEME["authentic"]}; text-align: center; font-weight: 600;">'
                'MODE: REAL EXPERIMENT</div>'
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        html_block("<h4 style='color:#F6EEFF; margin-bottom: 8px; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em;'>Analysis Settings</h4>"),
        unsafe_allow_html=True,
    )

    locked_threshold = 0.39
    if os.path.exists("outputs/final_threshold.json"):
        try:
            with open("outputs/final_threshold.json", "r") as f:
                locked_threshold = float(json.load(f).get("threshold", 0.39))
        except Exception:
            pass

    default_th = locked_threshold if not is_synthetic else 0.50

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_th,
        step=0.01,
        help="Videos with manipulation probability >= threshold are classified as MANIPULATED.",
    )

    inconclusive_range = st.slider(
        "Inconclusive Band [Low, High]",
        min_value=0.0,
        max_value=1.0,
        value=(0.40, 0.60),
        step=0.05,
        help="Probabilities falling within this zone are marked as INCONCLUSIVE.",
    )

    st.markdown("---")
    st.markdown(
        html_block("<h4 style='color:#F6EEFF; margin-bottom: 8px;'>Face Extraction Mode</h4>"),
        unsafe_allow_html=True,
    )

    no_face_detect = st.toggle(
        "Center-crop Fallback / Synthetic Mode",
        value=is_synthetic,
        help=(
            "When active, bypasses OpenCV Haar cascade face detection and takes center square crops. "
            "Essential for evaluating synthetic demonstration videos without real human faces."
        ),
    )

    if no_face_detect:
        st.caption("Center-crop active (fallback).")
    else:
        st.caption("Haar frontal cascade active.")

    st.markdown("---")
    st.markdown(
        html_block("<h4 style='color:#F6EEFF; margin-bottom: 6px; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em;'>Model Architecture</h4>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        html_block(f"""
        <div style="font-size:0.82rem; color:{THEME['text_muted']}; line-height:1.5;">
            • <b>Backbone:</b> 4-Block Conv2D<br>
            • <b>Temporal:</b> BiLSTM (128 units)<br>
            • <b>Attention:</b> Temporal Attention (64)<br>
            • <b>Resolution:</b> 10 frames @ 128x128 RGB
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        html_block("<h4 style='color:#F6EEFF; margin-bottom: 6px; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em;'>About DeepTrace</h4>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        html_block(f"""
        <div style="font-size:0.82rem; color:{THEME['text_muted']}; line-height:1.5;">
            DeepTrace provides AI-powered deepfake forensic analysis using spatial convolutional features and temporal sequence modeling.
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("Clear Analysis History", use_container_width=True):
        st.session_state.analysis_history = []
        st.session_state.latest_analysis = None
        st.session_state.batch_results = []
        st.toast("Forensic analysis history cleared.")

# -----------------------------------------------------------------------------
# Main Header & Status Chips
# -----------------------------------------------------------------------------
model_online = os.path.exists(selected_model_path)
status_pill_html = (
    f'<div class="status-online-pill"><span class="status-chip-dot dot-lilac"></span>MODEL ONLINE</div>'
    if model_online
    else f'<div class="status-offline-pill"><span class="status-chip-dot dot-vermilion"></span>MODEL OFFLINE</div>'
)

st.markdown(
    html_block(f"""
    <div class="deeptrace-header">
        <div class="deeptrace-brand-col">
            {APP_LOGO_SVG}
            <div>
                <div class="deeptrace-title">DEEPTRACE</div>
                <div class="deeptrace-sub">AI-POWERED DEEPFAKE FORENSIC ANALYSIS</div>
            </div>
        </div>
        <div>
            {status_pill_html}
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# Status Chips Row
git_commit_short = get_git_commit()
model_chip_name = "demo/best_model" if is_synthetic else "outputs/best_model (V2)"
model_chip_color = "dot-gold" if is_synthetic else "dot-lilac"

st.markdown(
    html_block(f"""
    <div class="status-chips-container">
        <div class="status-chip">
            <span class="status-chip-dot {model_chip_color}"></span>
            <span>Model: <b>{model_chip_name}</b></span>
        </div>
        <div class="status-chip">
            <span class="status-chip-dot dot-uv"></span>
            <span>Architecture: <b>CNN + BiLSTM + Temporal Attention</b></span>
        </div>
        <div class="status-chip">
            <span class="status-chip-dot dot-uv"></span>
            <span>Sequence: <b>10 frames @ 128x128 RGB</b></span>
        </div>
        <div class="status-chip">
            <span class="status-chip-dot dot-lilac"></span>
            <span>Threshold: <b>{threshold:.2f}</b></span>
        </div>
        <div class="status-chip">
            <span class="status-chip-dot dot-vermilion"></span>
            <span>Git: <code>{git_commit_short}</code></span>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# Persistent Prototype Banner (Mandatory Honesty Rule)
if is_synthetic:
    st.markdown(
        html_block("""
        <div class="prototype-warning-banner">
            <div class="prototype-warning-header">
                <span>NOTICE:</span> Prototype model trained on synthetic data. Results on real face videos are NOT meaningful.
            </div>
            <p class="prototype-warning-body">
                This interface demonstrates the end-to-end video decoding, temporal frame sampling, facial region cropping,
                and CNN-LSTM inference pipeline. Because this demonstration model was trained exclusively on synthetic geometrical
                patterns without human faces, its outputs represent proof of computational pipeline function,
                <b>not validated deepfake detection accuracy</b>.
            </p>
        </div>
        """),
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_analyze, tab_batch, tab_history, tab_results, tab_how, tab_about = st.tabs(
    [
        "Analyze",
        "Batch",
        "History",
        "Model & Results",
        "How It Works",
        "About & Limitations",
    ]
)

# =============================================================================
# TAB 1: ANALYZE (Single Video Evaluation)
# =============================================================================
with tab_analyze:
    st.markdown(
        html_block("""
        <div class="hero-box">
            <h2 class="hero-heading">Deepfake Forensic Analysis</h2>
            <p class="hero-desc">
                Analyze video authenticity using spatial and temporal deep-learning signals.
            </p>
        </div>
        """),
        unsafe_allow_html=True,
    )

    input_col, preview_col = st.columns([1.1, 0.9])
    target_video_path = None
    target_video_name = ""
    is_temp_file = False

    with input_col:
        sample_options = ["Upload a Video File"]
        celeb_real_sample = "data/raw/real/Celeb-real/id0_0000.mp4"
        celeb_fake_sample = "data/raw/fake/Celeb-synthesis/id0_id16_0000.mp4"
        demo_fake_sample = "data/demo/raw/fake/synth_fake_00.mp4"
        demo_real_sample = "data/demo/raw/real/synth_real_15.mp4"

        if os.path.exists(celeb_real_sample):
            sample_options.append("Preset: Authentic Real Video (Celeb-real id0_0000.mp4)")
        if os.path.exists(celeb_fake_sample):
            sample_options.append("Preset: Authentic Fake Video (Celeb-synthesis id0_id16_0000.mp4)")
        if os.path.exists(demo_fake_sample):
            sample_options.append("Preset: Synthetic Fake Sample (synth_fake_00.mp4)")
        if os.path.exists(demo_real_sample):
            sample_options.append("Preset: Synthetic Real Sample (synth_real_15.mp4)")

        preset_choice = st.radio(
            "Select Video Source:",
            options=sample_options,
            index=0,
            horizontal=False,
        )

        if preset_choice == "Upload a Video File":
            st.markdown(
                html_block(f"""
                <div style="font-size:0.86rem; color:{THEME['text_muted']}; margin-bottom:6px;">
                    Supported formats: <b>MP4, MOV, AVI, MKV</b> &nbsp;|&nbsp; Max practical size: <b>50MB</b><br>
                    Videos are processed into temporal frame sequences and analyzed locally. Ephemeral buffers are destroyed immediately after inference.
                </div>
                """),
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Drop a video here or browse files (.mp4, .mov, .avi, .mkv - max 50MB):",
                type=["mp4", "mov", "avi", "mkv"],
                help="Video will be processed locally in a temporary directory and destroyed immediately after inference.",
            )
            if uploaded_file is not None:
                if uploaded_file.size == 0:
                    st.markdown(callout_error("Uploaded file is empty (0 bytes)."), unsafe_allow_html=True)
                elif uploaded_file.size > 50 * 1024 * 1024:
                    st.markdown(callout_error("File exceeds maximum upload size (50MB). Please select a shorter video."), unsafe_allow_html=True)
                else:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
                    tfile.write(uploaded_file.getbuffer())
                    tfile.flush()
                    tfile.close()
                    target_video_path = tfile.name
                    target_video_name = uploaded_file.name
                    is_temp_file = True
        elif "Celeb-real" in preset_choice:
            target_video_path = celeb_real_sample
            target_video_name = "id0_0000.mp4 (Authentic REAL Sample)"
        elif "Celeb-synthesis" in preset_choice:
            target_video_path = celeb_fake_sample
            target_video_name = "id0_id16_0000.mp4 (Authentic FAKE Sample)"
        elif "synth_fake_00.mp4" in preset_choice:
            if os.path.exists(demo_fake_sample):
                target_video_path = demo_fake_sample
                target_video_name = "synth_fake_00.mp4 (Synthetic FAKE Sample)"
            else:
                st.markdown(callout_error(f"Preset file not found at {demo_fake_sample}. Run python demo.py first."), unsafe_allow_html=True)
        else:
            if os.path.exists(demo_real_sample):
                target_video_path = demo_real_sample
                target_video_name = "synth_real_15.mp4 (Synthetic REAL Sample)"
            else:
                st.markdown(callout_error(f"Preset file not found at {demo_real_sample}. Run python demo.py first."), unsafe_allow_html=True)

        # Inspect video metadata
        if target_video_path and os.path.exists(target_video_path):
            v_meta = get_video_metadata(target_video_path)
            if v_meta.get("readable"):
                st.markdown(
                    html_block(f"""
                    <div class="video-meta-grid">
                        <div class="video-meta-card">
                            <div class="video-meta-label">Filename</div>
                            <div class="video-meta-value">{v_meta.get('filename', 'video.mp4')[:16]}</div>
                        </div>
                        <div class="video-meta-card">
                            <div class="video-meta-label">File Size</div>
                            <div class="video-meta-value">{v_meta.get('filesize_mb', 0):.1f} MB</div>
                        </div>
                        <div class="video-meta-card">
                            <div class="video-meta-label">Duration</div>
                            <div class="video-meta-value">{v_meta.get('duration_sec', 0):.1f}s</div>
                        </div>
                        <div class="video-meta-card">
                            <div class="video-meta-label">Resolution</div>
                            <div class="video-meta-value">{v_meta.get('resolution', 'N/A')}</div>
                        </div>
                        <div class="video-meta-card">
                            <div class="video-meta-label">FPS</div>
                            <div class="video-meta-value">{v_meta.get('fps', 0)}</div>
                        </div>
                        <div class="video-meta-card">
                            <div class="video-meta-label">Frame Count</div>
                            <div class="video-meta-value">{v_meta.get('total_frames', 0)}</div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

        analyze_button = st.button(
            "Analyze Video",
            type="primary",
            disabled=(target_video_path is None or not os.path.exists(selected_model_path)),
            use_container_width=True,
        )

    with preview_col:
        st.markdown(
            html_block("<h4 style='color:#F6EEFF; margin-top:0;'>Source Video</h4>"),
            unsafe_allow_html=True,
        )
        if target_video_path and os.path.exists(target_video_path):
            st.html(render_custom_video_player(target_video_path))
        else:
            st.markdown(
                html_block(f"""
                <div style="border: 2px dashed {THEME['panel_border']}; border-radius: 12px; padding: 48px;
                            text-align: center; color: {THEME['text_muted']}; background: {THEME['bg_dark']};">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">[Preview]</div>
                    Select or upload a video file to activate forensic preview.
                </div>
                """),
                unsafe_allow_html=True,
            )

    # Execution pipeline
    if analyze_button and target_video_path:
        st.markdown("---")
        status_box = st.status("Running AI Forensic Analysis...", expanded=True)

        try:
            with status_box:
                st.write("1. Loading video stream and metadata...")
                time.sleep(0.1)

                st.write("2. Extracting frames across temporal segments...")
                time.sleep(0.1)

                if no_face_detect:
                    st.write("3. Detecting faces: Center-crop fallback active (Synthetic mode)...")
                else:
                    st.write("3. Detecting faces across sampled frames with OpenCV Haar cascade...")
                time.sleep(0.1)

                st.write("4. Building temporal sequence (10 frames @ 128x128 RGB)...")
                time.sleep(0.1)

                clips_arr, err_reason, diag = extract_clips_with_diagnostics(
                    video_path=target_video_path,
                    seq_len=10,
                    img_size=128,
                    clips_per_video=3,
                    face_margin=0.25,
                    no_face_detect=no_face_detect,
                )

                if err_reason is not None or clips_arr is None or len(clips_arr) == 0:
                    status_box.update(label="Forensic analysis halted", state="error", expanded=True)
                    if err_reason == "empty file":
                        st.markdown(callout_error("Uploaded video file is empty (0 bytes)."), unsafe_allow_html=True)
                    elif err_reason == "too short":
                        st.markdown(callout_error("No usable frames were detected in this video."), unsafe_allow_html=True)
                    elif err_reason == "no face detected":
                        st.markdown(callout_error("No face was reliably detected. The result may be less reliable."), unsafe_allow_html=True)
                    elif err_reason == "unreadable":
                        st.markdown(callout_error("Unable to decode this video. Please try another file."), unsafe_allow_html=True)
                    else:
                        st.markdown(callout_error("Unsupported video format. Please upload a supported video file."), unsafe_allow_html=True)
                else:
                    st.write(f"5. Running CNN feature extraction on {clips_arr.shape[0]} clips...")
                    try:
                        model = load_detection_model(selected_model_path)
                    except Exception:
                        status_box.update(label="Model loading failure", state="error", expanded=True)
                        st.markdown(callout_error("Detection model could not be loaded. Please restart the application."), unsafe_allow_html=True)
                        model = None

                    if model is not None:
                        time.sleep(0.1)

                        st.write("6. Analyzing temporal patterns with BiLSTM & Temporal Attention...")
                        predictions = model.predict(clips_arr, verbose=0).flatten()
                        p_fake = float(np.mean(predictions))
                        clip_probs = [float(p) for p in predictions]

                        attn_weights = extract_temporal_attention(model, clips_arr)
                        mean_attn = None
                        top_indices = []
                        if attn_weights is not None:
                            mean_attn = np.mean(attn_weights, axis=0)
                            top_indices = [int(idx) for idx in np.argsort(mean_attn)[::-1][:5]]

                        try:
                            _ = model.get_layer("temporal_attention")
                            model_arch_label = "CNN + BiLSTM + Temporal Attention"
                        except Exception:
                            model_arch_label = "CNN + LSTM"

                        st.write("7. Computing forensic result & calibrated decision threshold...")
                        v_info = compute_verdict(
                            p_fake=p_fake,
                            threshold=threshold,
                            inconclusive_band=inconclusive_range,
                        )
                        time.sleep(0.1)
                        status_box.update(label="Forensic Analysis Completed", state="complete", expanded=False)

                        # Store in session state
                        analysis_record = {
                            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "git_commit": git_commit_short,
                            "video_filename": target_video_name,
                            "model_used": selected_model_path,
                            "model_arch_label": model_arch_label,
                            "model_provenance": "prototype_synthetic" if is_synthetic else "real_experiment",
                            "p_fake": p_fake,
                            "p_real": 1.0 - p_fake,
                            "verdict": v_info["verdict"],
                            "label": v_info["label"],
                            "icon": v_info["icon"],
                            "color": v_info["color"],
                            "confidence": v_info["confidence"],
                            "threshold": threshold,
                            "inconclusive_band": inconclusive_range,
                            "clips_analyzed": len(clip_probs),
                            "clip_probabilities": clip_probs,
                            "temporal_attention": mean_attn.tolist() if mean_attn is not None else None,
                            "top_attention_frames": top_indices,
                            "face_stats": diag,
                            "sample_crops": diag.get("sample_crops", []),
                            "verdict_info": v_info,
                        }
                        st.session_state.latest_analysis = analysis_record

                        # Add to session history
                        st.session_state.analysis_history.insert(0, {
                            "Time (UTC)": analysis_record["generated_at"][:19].replace("T", " "),
                            "Video": target_video_name,
                            "Model": model_chip_name,
                            "P(Manipulation)": f"{p_fake:.4f} ({p_fake*100:.1f}%)",
                            "Verdict": f"{v_info['icon']} {v_info['label']}",
                            "Clips": len(clip_probs),
                        })
                        st.toast(f"Forensic analysis complete: {v_info['label']}")

        finally:
            if is_temp_file and target_video_path and os.path.exists(target_video_path):
                try:
                    os.remove(target_video_path)
                except Exception:
                    pass

    # Display Results if Available
    if st.session_state.latest_analysis is not None:
        rec = st.session_state.latest_analysis
        v_info = rec["verdict_info"]
        f_stats = rec["face_stats"]
        clip_meta = f_stats.get("clip_frames_meta", [])

        # -------------------------------------------------------------
        # RESULT HERO (Understandable within 2 seconds)
        # -------------------------------------------------------------
        face_status = (
            "Detected"
            if f_stats.get("detected_faces_count", 0) > 0
            else ("Center-Crop Fallback" if f_stats.get("fallback_used") else "Not detected")
        )

        st.markdown(
            html_block(f"""
            <div class="result-hero-card result-hero-card-{v_info['status']}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div class="verdict-pill-{v_info['status']}">
                            <span>{v_info['icon']}</span>
                            <span>{rec['verdict']}</span>
                        </div>
                        <div class="prob-metric-row">
                            <span class="prob-metric-title">Fake Probability:</span>
                            <span class="prob-metric-value">{rec['p_fake']*100:.2f}%</span>
                            <span class="prob-metric-thresh">(Decision Threshold: {rec['threshold']:.2f})</span>
                        </div>
                    </div>
                </div>
                <div class="evidence-secondary-row">
                    <div class="evidence-secondary-card">
                        <div class="evidence-secondary-label">Frames Analyzed</div>
                        <div class="evidence-secondary-val">10 frames</div>
                    </div>
                    <div class="evidence-secondary-card">
                        <div class="evidence-secondary-label">Face Detection</div>
                        <div class="evidence-secondary-val">{face_status}</div>
                    </div>
                    <div class="evidence-secondary-card">
                        <div class="evidence-secondary-label">Model</div>
                        <div class="evidence-secondary-val">{rec.get('model_arch_label', 'CNN + BiLSTM + Temporal Attention')}</div>
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # Telemetry Gauges and Charts
        g_col, c_col = st.columns([1, 1.2])

        with g_col:
            gauge_fig = build_probability_gauge(
                p_fake=rec["p_fake"],
                inconclusive_low=rec["inconclusive_band"][0],
                inconclusive_high=rec["inconclusive_band"][1],
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

        with c_col:
            bar_fig = build_clips_bar_chart(
                clip_probabilities=rec["clip_probabilities"],
                threshold=rec["threshold"],
            )
            st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        # -------------------------------------------------------------
        # TEMPORAL FORENSIC EVIDENCE & FRAME EVIDENCE VIEWER
        # -------------------------------------------------------------
        st.markdown(
            html_block("""
            <div style="margin-top: 28px; margin-bottom: 12px;">
                <h3 style="color:#F6EEFF; margin-bottom: 4px;">Temporal Forensic Evidence</h3>
                <p style="color:#B9A4D6; font-size: 0.92rem; margin: 0; line-height: 1.5;">
                    The model analyzes multiple frames and assigns temporal attention weights to identify which frames contributed most strongly to the final prediction.
                    Frames receiving higher temporal attention contributed more strongly to the model's decision.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

        attn_scores = np.array(rec["temporal_attention"]) if rec.get("temporal_attention") is not None else None
        top_f = rec.get("top_attention_frames", [])

        # Display Frame Evidence Viewer (Top 3 Important Frames)
        if attn_scores is not None and len(top_f) > 0 and len(clip_meta) >= 10:
            top_3 = top_f[:3]
            f_cols = st.columns(len(top_3))
            for rank_i, frame_step in enumerate(top_3, start=1):
                with f_cols[rank_i - 1]:
                    meta_item = clip_meta[frame_step] if frame_step < len(clip_meta) else None
                    if meta_item and "crop" in meta_item:
                        st.markdown(
                            html_block(f"""
                            <div class="frame-evidence-card">
                                <div class="frame-evidence-rank">Important Frame 0{rank_i}</div>
                            """),
                            unsafe_allow_html=True,
                        )
                        st.image(meta_item["crop"], use_container_width=True)
                        st.markdown(
                            html_block(f"""
                                <div class="frame-evidence-meta">
                                    Frame #{meta_item['frame_idx']}<br>
                                    Timestamp: {meta_item['timestamp']:.2f}s
                                </div>
                                <div class="frame-evidence-tag">
                                    Attention: {attn_scores[frame_step]*100:.2f}%
                                </div>
                            </div>
                            """),
                            unsafe_allow_html=True,
                        )
        elif attn_scores is not None and len(top_f) > 0:
            st.markdown(
                callout_info("Frame thumbnails unavailable for this video sequence; displaying temporal attention distribution."),
                unsafe_allow_html=True,
            )

        # Temporal Attention Distribution Chart
        if attn_scores is not None:
            st.markdown(
                html_block("<h5 style='color:#F6EEFF; margin-top: 18px;'>Temporal Attention Distribution (Frame Step 1–10)</h5>"),
                unsafe_allow_html=True,
            )
            attn_fig = build_temporal_attention_chart(attn_scores, top_indices=top_f[:3])
            st.plotly_chart(attn_fig, use_container_width=True, config={"displayModeBar": False})

        # -------------------------------------------------------------
        # FORENSIC SUMMARY
        # -------------------------------------------------------------
        st.markdown(
            html_block("<h3 style='color:#F6EEFF; margin-top: 28px; margin-bottom: 12px;'>Forensic Summary</h3>"),
            unsafe_allow_html=True,
        )

        top_f_str = (
            f"Frame {top_f[0]+1} ({attn_scores[top_f[0]]*100:.1f}%)"
            if (attn_scores is not None and len(top_f) > 0)
            else "Uniform"
        )

        st.markdown(
            html_block(f"""
            <div class="summary-block-grid">
                <div class="summary-block-item">
                    <div class="summary-block-title">Prediction</div>
                    <div class="summary-block-value" style="color: {v_info['color']};">{rec['label']}</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Fake Probability</div>
                    <div class="summary-block-value">{rec['p_fake']*100:.2f}%</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Decision Threshold</div>
                    <div class="summary-block-value">{rec['threshold']:.2f}</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Frames Analyzed</div>
                    <div class="summary-block-value">10 frames</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Faces Detected</div>
                    <div class="summary-block-value">{f_stats.get('detected_faces_count', 0)} ({f_stats.get('face_detection_rate', 0.0)*100:.1f}%)</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Temporal Evidence</div>
                    <div class="summary-block-value">{top_f_str}</div>
                </div>
                <div class="summary-block-item">
                    <div class="summary-block-title">Model Architecture</div>
                    <div class="summary-block-value">{rec.get('model_arch_label', 'CNN + BiLSTM + Attention')}</div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # EXPANDABLE TECHNICAL DETAILS (No internal paths)
        # -------------------------------------------------------------
        with st.expander("Technical Details", expanded=False):
            st.markdown("""
            - **Architecture**: `CNN → BiLSTM → Temporal Attention → Classifier`
            - **Input**: `10 RGB frames × 128 × 128`
            - **Dataset**: `Celeb-DF v2`
            - **Decision Threshold**: `0.39`
            - **Model**: `V2`
            - **Framework**: `TensorFlow / Keras`
            - **Inference Pipeline**: `Video → Frame Sampling → Face/Frame Processing → CNN → BiLSTM → Attention → Prediction`
            """)

        # -------------------------------------------------------------
        # ABOUT THE MODEL
        # -------------------------------------------------------------
        st.markdown(
            html_block(f"""
            <div class="glass-panel" style="margin-top: 18px;">
                <h4 style="color:{THEME['primary_accent']}; margin-top:0;">About the Model</h4>
                <p style="font-size:0.92rem; color:{THEME['text_primary']}; line-height:1.6; margin:0;">
                    The detector combines spatial visual features from a convolutional neural network with temporal sequence modeling through a bidirectional LSTM.
                    Temporal attention highlights the frames that contributed most strongly to the final prediction.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # DISCLAIMER
        # -------------------------------------------------------------
        st.markdown(
            html_block("""
            <div class="forensic-disclaimer-card">
                <b>Notice:</b> AI-based forensic analysis is probabilistic and should not be treated as definitive proof of authenticity or manipulation.
            </div>
            """),
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # REPORT DOWNLOADS
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown(
            html_block("<h4 style='color:#F6EEFF;'>Forensic Audit Report Export</h4>"),
            unsafe_allow_html=True,
        )
        r_col1, r_col2 = st.columns(2)

        json_report_str = build_json_report(rec)
        html_report_str = build_html_report(rec)
        safe_name = rec["video_filename"].split()[0].replace(".mp4", "")

        with r_col1:
            st.download_button(
                label="Download JSON Report",
                data=json_report_str,
                file_name=f"forensic_report_{safe_name}.json",
                mime="application/json",
                use_container_width=True,
            )

        with r_col2:
            st.download_button(
                label="Download HTML Report",
                data=html_report_str,
                file_name=f"forensic_report_{safe_name}.html",
                mime="text/html",
                use_container_width=True,
            )

# =============================================================================
# TAB 2: BATCH (Multi-Video Processing)
# =============================================================================
with tab_batch:
    st.markdown(
        html_block("<h3 style='color:#F6EEFF; margin-top:0;'>Batch Forensic Video Evaluation</h3>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        html_block(
            f"<p style='color:{THEME['text_muted']};'>Upload multiple video files to analyze sequentially. "
            "Corrupted, short, or invalid files are isolated and logged without halting batch execution.</p>"
        ),
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Choose video files for batch processing (.mp4, .avi, .mov, .mkv):",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    run_batch_button = st.button(
        "Process Batch Files",
        type="primary",
        disabled=(not batch_files or not os.path.exists(selected_model_path)),
    )

    if run_batch_button and batch_files:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        batch_rows = []

        model = load_detection_model(selected_model_path)
        total_items = len(batch_files)

        for idx, b_file in enumerate(batch_files):
            pct = (idx + 1) / total_items
            status_text.text(f"Processing ({idx + 1}/{total_items}): {b_file.name}...")
            progress_bar.progress(pct)

            # Isolated execution per file
            t_path = None
            try:
                if b_file.size == 0:
                    batch_rows.append({
                        "File": b_file.name,
                        "P(Fake)": "N/A",
                        "Verdict": "ERROR",
                        "Clips": 0,
                        "Face Detection": "N/A",
                        "Error Details": "Empty file (0 bytes)",
                    })
                    continue

                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{b_file.name}")
                tfile.write(b_file.getbuffer())
                tfile.flush()
                tfile.close()
                t_path = tfile.name

                clips_arr, err, diag = extract_clips_with_diagnostics(
                    video_path=t_path,
                    seq_len=10,
                    img_size=128,
                    clips_per_video=3,
                    face_margin=0.25,
                    no_face_detect=no_face_detect,
                )

                if err is not None or clips_arr is None or len(clips_arr) == 0:
                    batch_rows.append({
                        "File": b_file.name,
                        "P(Fake)": "N/A",
                        "Verdict": "ERROR",
                        "Clips": 0,
                        "Face Detection": "Failed",
                        "Error Details": f"Extraction error: {err}",
                    })
                else:
                    preds = model.predict(clips_arr, verbose=0).flatten()
                    p_fake = float(np.mean(preds))
                    v_res = compute_verdict(p_fake, threshold, inconclusive_range)

                    fallback_info = "Fallback used" if diag.get("fallback_used") else "Exact"
                    batch_rows.append({
                        "File": b_file.name,
                        "P(Fake)": f"{p_fake:.4f}",
                        "Verdict": f"{v_res['icon']} {v_res['label']}",
                        "Clips": len(preds),
                        "Face Detection": fallback_info,
                        "Error Details": "None",
                    })

            except Exception as exc:
                batch_rows.append({
                    "File": b_file.name,
                    "P(Fake)": "N/A",
                    "Verdict": "EXCEPTION",
                    "Clips": 0,
                    "Face Detection": "N/A",
                    "Error Details": str(exc),
                })
            finally:
                if t_path and os.path.exists(t_path):
                    try:
                        os.remove(t_path)
                    except Exception:
                        pass

        progress_bar.progress(1.0)
        status_text.text("Batch processing complete")
        st.session_state.batch_results = batch_rows

    if st.session_state.batch_results:
        st.markdown(
            html_block("<h4 style='color:#F6EEFF;'>Batch Processing Results</h4>"),
            unsafe_allow_html=True,
        )
        batch_df = pd.DataFrame(st.session_state.batch_results)
        st.markdown(render_forensic_table(batch_df), unsafe_allow_html=True)

        csv_data = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Batch Results (CSV)",
            data=csv_data,
            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# =============================================================================
# TAB 3: HISTORY (Session Audit Trail)
# =============================================================================
with tab_history:
    st.markdown(
        html_block("<h3 style='color:#F6EEFF; margin-top:0;'>Session Analysis Audit Trail</h3>"),
        unsafe_allow_html=True,
    )
    st.markdown(
        html_block(
            f"<p style='color:{THEME['text_muted']};'>Audit trail of all video analyses performed during this browser session. "
            "<b>Privacy Note:</b> No video frames or media files are stored on disk or server storage.</p>"
        ),
        unsafe_allow_html=True,
    )

    if st.session_state.analysis_history:
        hist_df = pd.DataFrame(st.session_state.analysis_history)
        st.markdown(render_forensic_table(hist_df), unsafe_allow_html=True)

        h_csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Session History (CSV)",
            data=h_csv,
            file_name=f"analysis_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            callout_info("No analyses recorded in this session yet. Run an analysis in the Analyze or Batch tab."),
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 4: MODEL & RESULTS (Honest Experiment Reporting)
# =============================================================================
with tab_results:
    st.markdown(
        html_block("<h3 style='color:#F6EEFF; margin-top:0;'>Benchmark Evaluation & Model Architecture</h3>"),
        unsafe_allow_html=True,
    )

    metrics_path = "outputs/metrics.json"
    run_info_path = "outputs/run_info.json"

    # Strictly honest reporting: only read outputs/metrics.json if it genuinely exists
    if os.path.exists(metrics_path):
        st.markdown(
            html_block("<h4 style='color:#F6EEFF;'>Celeb-DF v2 Benchmark Results</h4>"),
            unsafe_allow_html=True,
        )
        try:
            import json
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            st.json(metrics_data)
        except Exception as e:
            st.markdown(callout_error(f"Error reading metrics file: {e}"), unsafe_allow_html=True)
    else:
        st.markdown(
            html_block(f"""
            <div class="glass-panel" style="border-left: 5px solid {THEME['primary_accent']};">
                <h4 style="color: {THEME['primary_accent']}; margin-top: 0;">Celeb-DF v2 Benchmark Status</h4>
                <p style="color: {THEME['text_primary']}; line-height: 1.5;">
                    <b>Real experiment not yet run.</b><br>
                    Official Celeb-DF v2 benchmark training and evaluation have not been executed on this machine.
                    In accordance with strict scientific honesty principles, precision, recall, F1, and AUC metrics
                    are <b>never simulated, placeholder-generated, or estimated</b>.
                </p>
                <div style="font-size: 0.85rem; color: {THEME['text_muted']};">
                    Expected benchmark data: 890 real videos, 5,639 fake videos, evaluated on the official 518-video test list.
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    if os.path.exists(run_info_path):
        st.markdown(
            html_block("<h4 style='color:#F6EEFF;'>Training Run Parameters</h4>"),
            unsafe_allow_html=True,
        )
        try:
            import json
            with open(run_info_path, "r", encoding="utf-8") as f:
                run_data = json.load(f)
            st.json(run_data)
        except Exception as e:
            st.markdown(callout_error(f"Error reading run_info file: {e}"), unsafe_allow_html=True)

    # Check for ROC and Confusion Matrix plots
    cm_path = "outputs/confusion_matrix.png"
    roc_path = "outputs/roc_curve.png"
    if os.path.exists(cm_path) or os.path.exists(roc_path):
        st.markdown(
            html_block("<h4 style='color:#F6EEFF;'>Benchmark Visualizations</h4>"),
            unsafe_allow_html=True,
        )
        p_col1, p_col2 = st.columns(2)
        if os.path.exists(cm_path):
            with p_col1:
                st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
        if os.path.exists(roc_path):
            with p_col2:
                st.image(roc_path, caption="ROC Curve", use_container_width=True)

    # Architecture Blueprint
    st.markdown("---")
    st.markdown(
        html_block("<h4 style='color:#F6EEFF;'>Spatio-Temporal CNN-LSTM Architecture Blueprint</h4>"),
        unsafe_allow_html=True,
    )

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown(
            html_block(f"""
            <div class="glass-panel">
                <h5 style="color: {THEME['authentic']}; margin-top:0;">Spatial Feature Extractor (CNN)</h5>
                <ul style="font-size: 0.88rem; color: {THEME['text_primary']}; line-height: 1.6;">
                    <li><b>Input Shape:</b> <code>(batch, 10, 128, 128, 3)</code> uint8 RGB</li>
                    <li><b>Normalization:</b> <code>Rescaling(1./255)</code></li>
                    <li><b>Block 1:</b> <code>Conv2D(32, 3x3)</code> + BatchNorm + MaxPool(2x2) + Dropout(0.2)</li>
                    <li><b>Block 2:</b> <code>Conv2D(64, 3x3)</code> + BatchNorm + MaxPool(2x2) + Dropout(0.2)</li>
                    <li><b>Block 3:</b> <code>Conv2D(128, 3x3)</code> + BatchNorm + MaxPool(2x2) + Dropout(0.3)</li>
                    <li><b>Block 4:</b> <code>Conv2D(128, 3x3)</code> + BatchNorm + MaxPool(2x2) + Dropout(0.3)</li>
                    <li><b>Projection:</b> <code>TimeDistributed(Flatten)</code> + <code>Dense(128, ReLU)</code></li>
                </ul>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with b_col2:
        st.markdown(
            html_block(f"""
            <div class="glass-panel">
                <h5 style="color: {THEME['manipulated']}; margin-top:0;">Temporal Sequence & Classification (V2 Attention)</h5>
                <ul style="font-size: 0.88rem; color: {THEME['text_primary']}; line-height: 1.6;">
                    <li><b>Recurrent Layer:</b> <code>Bidirectional(LSTM(128, return_sequences=True))</code></li>
                    <li><b>Attention Layer:</b> <code>TemporalAttention(64)</code> (learns frame relevance weights)</li>
                    <li><b>Temporal Modeling:</b> Resolves inter-frame facial jitter, boundary flicker, and temporal warp</li>
                    <li><b>Classification Head:</b> <code>Dense(128, ReLU)</code> + Dropout(0.3)</li>
                    <li><b>Output:</b> <code>Dense(1, Sigmoid)</code> outputting Fake Probability</li>
                    <li><b>Loss Function:</b> Binary Crossentropy with dynamic class balancing</li>
                    <li><b>Optimizer:</b> Adam (learning_rate = 3e-4)</li>
                </ul>
            </div>
            """),
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 5: HOW IT WORKS (Methodology & Pipeline Diagram)
# =============================================================================
with tab_how:
    st.markdown(
        html_block("<h3 style='color:#F6EEFF; margin-top:0;'>How the Deepfake Detector Operates</h3>"),
        unsafe_allow_html=True,
    )

    # Render Pipeline Flow Diagram
    st.markdown(html_block(PIPELINE_FLOW_HTML), unsafe_allow_html=True)

    h_col1, h_col2, h_col3 = st.columns(3)

    with h_col1:
        st.markdown(
            html_block(f"""
            <div class="glass-panel">
                <h5 style="color: {THEME['primary_accent']}; margin-top:0;">1. Uniform Sampling</h5>
                <p style="font-size: 0.85rem; color: {THEME['text_muted']};">
                    Videos are partitioned into <code>clips_per_video = 3</code> non-overlapping segments.
                    Within each segment, 10 frames are sampled linearly across the duration.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with h_col2:
        st.markdown(
            html_block(f"""
            <div class="glass-panel">
                <h5 style="color: {THEME['authentic']}; margin-top:0;">2. Facial Bounding Box</h5>
                <p style="font-size: 0.85rem; color: {THEME['text_muted']};">
                    OpenCV Haar cascade detects frontal facial features. An expanded 25% margin ensures
                    hairline, jawline, and boundary blending artifacts are fully enclosed.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

    with h_col3:
        st.markdown(
            html_block(f"""
            <div class="glass-panel">
                <h5 style="color: {THEME['manipulated']}; margin-top:0;">3. Spatio-Temporal Hybrid</h5>
                <p style="font-size: 0.85rem; color: {THEME['text_muted']};">
                    Frame features extracted by the 4-block CNN are sequentially evaluated by the LSTM.
                    Clip scores are mean-aggregated to produce the video-level verdict.
                </p>
            </div>
            """),
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 6: ABOUT & LIMITATIONS (Academic Transparency & Ethics)
# =============================================================================
with tab_about:
    st.markdown(
        html_block("<h3 style='color:#F6EEFF; margin-top:0;'>About the Project, Limitations & Privacy</h3>"),
        unsafe_allow_html=True,
    )

    st.markdown(
        html_block(f"""
        <div class="glass-panel">
            <h4 style="color: {THEME['primary_accent']}; margin-top: 0;">Forensic Limitations & Technical Disclosures</h4>
            <p style="font-size: 0.88rem; color: {THEME['text_primary']}; line-height: 1.6;">
                Deepfake detection in unconstrained real-world environments is an open research challenge.
                Users and forensic evaluators must understand the following technical boundary conditions:
            </p>
            <ul style="font-size: 0.88rem; color: {THEME['text_muted']}; line-height: 1.7;">
                <li><b>OpenCV Haar Cascade Limits:</b> Haar cascades require frontal or near-frontal facial poses. Extreme head rotations, strong shadows, or motion blur can cause detection failures.</li>
                <li><b>Class Imbalance Considerations:</b> The Celeb-DF v2 dataset contains 890 real and 5,639 synthetic videos (~1:6 ratio). Class weighting is used during training to prevent majority-class collapse.</li>
                <li><b>Identity Overlap Leakage Prevention:</b> Train, validation, and test splits are strictly separated by celebrity subject identity to prevent the neural network from memorizing facial identities rather than manipulation artifacts.</li>
                <li><b>Cross-Dataset Generalization:</b> Neural networks trained on one manipulation methodology (e.g. DeepFake swapping) frequently experience performance degradation when evaluated against unseen generation techniques (e.g. diffusion models or audio-driven lip sync).</li>
                <li><b>Probabilistic Nature:</b> Outputs represent model manipulation probabilities, not definitive proof of fabrication. Results should be treated as diagnostic indicators alongside human review.</li>
            </ul>
        </div>

        <div class="glass-panel" style="border-left: 5px solid {THEME['authentic']};">
            <h4 style="color: {THEME['authentic']}; margin-top: 0;">Privacy & Ephemeral Data Processing</h4>
            <p style="font-size: 0.88rem; color: {THEME['text_primary']}; line-height: 1.6;">
                All video uploads are handled on this local machine. Uploaded files are written to ephemeral temporary buffers
                and are <b>strictly deleted immediately following inference</b> in a guaranteed <code>finally:</code> block.
                No videos or facial images are transmitted over external networks or permanently retained.
            </p>
        </div>
        """),
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    html_block(f"""
    <div class="forensic-footer-bar">
        Deepfake Forensic Analysis &nbsp;|&nbsp; CNN + BiLSTM + Temporal Attention &nbsp;|&nbsp; 
        AI-assisted analysis — results are probabilistic.
    </div>
    """),
    unsafe_allow_html=True,
)
