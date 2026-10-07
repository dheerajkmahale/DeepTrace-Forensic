"""Helper utilities, analysis logic, report generators, and visualizers for Streamlit UI.

Provides pure functions for:
- Verdict determination with custom decision thresholds and inconclusive bands
- JSON and HTML forensic report generation (Ultraviolet Forensics palette)
- Plotly gauge and per-clip bar chart visualizers (zero black, blue, or green)
- Diagnostic video extraction with face detection metrics and sample crop capture
"""

import base64
import datetime
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import plotly.graph_objects as go

from preprocess import center_crop_and_resize, crop_face_with_margin, get_face_cascade
from theme import THEME


def html_block(html_str: str) -> str:
    """Format an HTML string safely for Streamlit markdown rendering.

    Guarantees:
    - Strips leading and trailing whitespace from every line (no line starts with 4+ spaces).
    - Discards empty or whitespace-only lines (no blank lines).
    - Prevents CommonMark from mistakenly interpreting indented HTML as preformatted code blocks.
    """
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "\n".join(lines)


def render_custom_video_player(video_path: str) -> str:
    """Render an accessible HTML5 video player conforming to the Ultraviolet Forensics palette.

    Replaces native browser controls (which introduce near-black pixels in Blink shadow DOM)
    with a fully-themed accessible plum/violet control bar.

    Features:
    - <video> element WITHOUT native controls (object-fit: contain, background #3A1A63).
    - Fully-plum/violet control bar (#2A1248 background, #5B2E91 border).
    - Keyboard-accessible Play/Pause button (#8B5CF6 accent) with aria-label.
    - Keyboard-accessible timeline scrubber range input with aria-label.
    - Monospace timestamp display (#F6EEFF lilac-white, aria-label).
    - File size guard: files > 15MB display a plum-framed first-frame poster with notice.
    """
    if not os.path.exists(video_path):
        return html_block(f"""
        <div style="border: 2px dashed {THEME['panel_border']}; border-radius: 12px; padding: 48px;
                    text-align: center; color: {THEME['text_muted']}; background: {THEME['bg_dark']};">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">[No Video]</div>
            Selected video file not found on disk.
        </div>
        """)

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    if file_size_mb > 15.0:
        # Generate plum-framed first-frame poster
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            poster_b64 = base64.b64encode(buf).decode('utf-8')
            poster_src = f"data:image/jpeg;base64,{poster_b64}"
            return html_block(f"""
            <div class="video-preview-wrapper" style="border: 1px solid {THEME['panel_border']}; border-radius: 12px; overflow: hidden; background: {THEME['panel_dark']};">
                <div class="custom-video-screen" style="position: relative; width: 100%; aspect-ratio: 16/9; background: {THEME['panel_dark']}; display: flex; align-items: center; justify-content: center;">
                    <img src="{poster_src}" alt="Video First Frame Poster" style="width: 100%; height: 100%; object-fit: contain; background: {THEME['panel_dark']};" />
                    <div class="scanner-overlay"></div>
                </div>
                <div style="padding: 10px 14px; background: {THEME['bg_dark']}; border-top: 1px solid {THEME['panel_border']}; color: {THEME['text_muted']}; font-size: 0.82rem; text-align: center;">
                    Video file ({file_size_mb:.1f} MB) exceeds inline preview limit (15 MB). Plum-framed first-frame poster shown; full video processed by pipeline.
                </div>
            </div>
            """)

    # Read and base64 encode
    with open(video_path, "rb") as f:
        v_b64 = base64.b64encode(f.read()).decode("utf-8")

    player_html = f"""
    <div class="video-preview-wrapper" style="border: 1px solid {THEME['panel_border']}; border-radius: 12px; overflow: hidden; background: {THEME['panel_dark']};">
        <div class="custom-video-screen" style="position: relative; width: 100%; aspect-ratio: 16/9; background: {THEME['panel_dark']}; display: flex; align-items: center; justify-content: center; overflow: hidden;">
            <video id="forensic-custom-video"
                   src="data:video/mp4;base64,{v_b64}"
                   preload="metadata"
                   playsinline
                   ontimeupdate="var s=document.getElementById('forensic-seeker'); var t=document.getElementById('forensic-time'); if(s && this.duration) {{ s.value=(this.currentTime/this.duration)*100; }} if(t) {{ var m=Math.floor(this.currentTime/60); var sec=Math.floor(this.currentTime%60); t.innerText=(m<10?'0':'')+m+':'+(sec<10?'0':'')+sec; }}"
                   onended="var b=document.getElementById('forensic-play-btn'); if(b) b.innerText='Play';"
                   style="width: 100%; height: 100%; object-fit: contain; background: {THEME['panel_dark']};"></video>
            <div class="scanner-overlay"></div>
        </div>
        <div class="custom-video-controls" style="display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: {THEME['bg_dark']}; border-top: 1px solid {THEME['panel_border']};">
            <button id="forensic-play-btn" type="button" aria-label="Play or pause forensic video preview" tabindex="0"
                    onclick="var v=document.getElementById('forensic-custom-video'); if(v){{if(v.paused){{v.play(); this.innerText='Pause';}}else{{v.pause(); this.innerText='Play';}}}}"
                    style="background: {THEME['primary_accent']}; color: #FFFFFF; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 700; font-size: 0.95rem; cursor: pointer;">
                Play
            </button>
            <input id="forensic-seeker" type="range" min="0" max="100" value="0" aria-label="Timeline scrubber" tabindex="0"
                   oninput="var v=document.getElementById('forensic-custom-video'); if(v && v.duration){{v.currentTime=(this.value/100)*v.duration;}}"
                   style="flex: 1; accent-color: {THEME['primary_accent']}; background: {THEME['panel_dark']}; cursor: pointer;">
            <span id="forensic-time" aria-label="Playback timestamp" style="color: {THEME['text_primary']}; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 600;">
                00:00
            </span>
        </div>
    </div>
    """
    return html_block(player_html)


def render_forensic_table(df: Any) -> str:
    """Render a pandas DataFrame as a fully-themed Ultraviolet Forensics HTML table.

    Replaces Streamlit's Glide Data Grid canvas (which renders hardcoded dark-blue canvas cells)
    with a clean, styled HTML5 table adhering strictly to zero black, blue, or green.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return html_block(f"""
        <div style="padding: 20px; text-align: center; color: {THEME['text_muted']}; background: {THEME['panel_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px;">
            No records to display.
        </div>
        """)

    headers = "".join([f"<th style='padding: 10px 14px; background: {THEME['bg_dark']}; color: {THEME['text_muted']}; border-bottom: 1px solid {THEME['panel_border']}; text-align: left; font-size: 0.85rem;'>{col}</th>" for col in df.columns])

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = str(row[col])
            # Color-code verdict chips
            if "AUTHENTIC" in val or "REAL" in val:
                cell_content = f"<span style='color: {THEME['authentic']}; font-weight: 700;'>{val}</span>"
            elif "MANIPULATED" in val or "FAKE" in val:
                cell_content = f"<span style='color: {THEME['manipulated']}; font-weight: 700;'>{val}</span>"
            elif "INCONCLUSIVE" in val:
                cell_content = f"<span style='color: {THEME['inconclusive']}; font-weight: 700;'>{val}</span>"
            elif "P(" in str(col) or "Prob" in str(col):
                cell_content = f"<span style='font-family: monospace; color: {THEME['text_primary']};'>{val}</span>"
            else:
                cell_content = f"<span style='color: {THEME['text_primary']};'>{val}</span>"

            cells.append(f"<td style='padding: 10px 14px; border-bottom: 1px solid {THEME['panel_border']}; font-size: 0.88rem;'>{cell_content}</td>")

        rows_html.append(f"<tr style='background: {THEME['panel_dark']};'>{''.join(cells)}</tr>")

    table_html = f"""
    <div style="overflow-x: auto; border: 1px solid {THEME['panel_border']}; border-radius: 10px; margin: 12px 0 16px 0;">
        <table style="width: 100%; border-collapse: collapse; background: {THEME['panel_dark']};">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    return html_block(table_html)


def callout_info(msg: str) -> str:
    """Render a styled info callout within the Ultraviolet Forensics palette."""
    return html_block(f"""
        <div style="background: {THEME['panel_dark']}; border-left: 4px solid {THEME['primary_accent']}; padding: 12px 16px; border-radius: 6px; margin: 10px 0; color: {THEME['text_primary']};">
            <span style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 0.85rem; letter-spacing: 0.05em;">INFO</span> &nbsp; {msg}
        </div>
    """)


def callout_warning(msg: str) -> str:
    """Render a styled warning callout within the Ultraviolet Forensics palette."""
    return html_block(f"""
        <div style="background: {THEME['panel_dark']}; border-left: 4px solid {THEME['inconclusive']}; padding: 12px 16px; border-radius: 6px; margin: 10px 0; color: {THEME['text_primary']};">
            <span style="color: {THEME['inconclusive']}; font-weight: bold; font-size: 0.85rem; letter-spacing: 0.05em;">NOTICE</span> &nbsp; {msg}
        </div>
    """)


def callout_error(msg: str) -> str:
    """Render a styled error callout within the Ultraviolet Forensics palette."""
    return html_block(f"""
        <div style="background: {THEME['panel_dark']}; border-left: 4px solid {THEME['manipulated']}; padding: 12px 16px; border-radius: 6px; margin: 10px 0; color: {THEME['text_primary']};">
            <span style="color: {THEME['manipulated']}; font-weight: bold; font-size: 0.85rem; letter-spacing: 0.05em;">ERROR</span> &nbsp; {msg}
        </div>
    """)


def callout_success(msg: str) -> str:
    """Render a styled success callout within the Ultraviolet Forensics palette."""
    return html_block(f"""
        <div style="background: {THEME['panel_dark']}; border-left: 4px solid {THEME['authentic']}; padding: 12px 16px; border-radius: 6px; margin: 10px 0; color: {THEME['text_primary']};">
            <span style="color: {THEME['authentic']}; font-weight: bold; font-size: 0.85rem; letter-spacing: 0.05em;">VERIFIED</span> &nbsp; {msg}
        </div>
    """)


def get_git_commit(cwd: Optional[str] = None) -> str:
    """Retrieve current short git commit hash or 'unknown' if not in a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd or os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def compute_verdict(
    p_fake: float,
    threshold: float = 0.5,
    inconclusive_band: Tuple[float, float] = (0.40, 0.60),
) -> Dict[str, Any]:
    """Compute forensic verdict, confidence assessment, and visual attributes.

    Args:
        p_fake: Manipulation probability in [0.0, 1.0].
        threshold: Decision cutoff for binary classification.
        inconclusive_band: Tuple of (low, high) bounds defining the uncertain zone.

    Returns:
        Dict containing verdict label, status, icon, color, confidence description,
        and probability values.
    """
    p_fake = float(max(0.0, min(1.0, p_fake)))
    p_real = float(max(0.0, min(1.0, 1.0 - p_fake)))

    low_band, high_band = sorted(inconclusive_band)

    # Inconclusive band check takes precedence
    if low_band <= p_fake <= high_band:
        verdict = "INCONCLUSIVE"
        label = "INCONCLUSIVE"
        icon = "~"
        status = "inconclusive"
        color = THEME["inconclusive"]  # Gold (#FFC247)
        confidence = (
            f"Manipulation probability ({p_fake:.1%}) lies within the configured inconclusive "
            f"band [{low_band:.2f}, {high_band:.2f}]. Signals are ambiguous."
        )
    elif p_fake >= threshold:
        verdict = "FAKE"
        label = "MANIPULATED"
        icon = "!"
        status = "fake"
        color = THEME["manipulated"]  # Vermilion (#FF5A36)
        if p_fake >= 0.85:
            confidence = f"High confidence manipulation ({p_fake:.1%}) detected across spatio-temporal sequences."
        else:
            confidence = f"Moderate confidence manipulation ({p_fake:.1%}) exceeding threshold cutoff ({threshold:.2f})."
    else:
        verdict = "REAL"
        label = "AUTHENTIC"
        icon = "✓"
        status = "real"
        color = THEME["authentic"]  # Lilac (#C4A1FF)
        if p_fake <= 0.15:
            confidence = f"High confidence authentic sequence ({p_real:.1%} real) with strong temporal consistency."
        else:
            confidence = f"Moderate confidence authentic features ({p_real:.1%} real) below threshold cutoff ({threshold:.2f})."

    return {
        "verdict": verdict,
        "label": label,
        "icon": icon,
        "status": status,
        "color": color,
        "confidence": confidence,
        "p_fake": p_fake,
        "p_real": p_real,
        "threshold": threshold,
        "inconclusive_band": (low_band, high_band),
    }


def build_json_report(analysis_data: Dict[str, Any]) -> str:
    """Generate a clean, standardized JSON forensic report string without raw media."""
    raw_face_stats = analysis_data.get("face_stats", {})
    clean_face_stats = {
        "total_frames": int(raw_face_stats.get("total_frames", 0)),
        "frames_inspected": int(raw_face_stats.get("frames_inspected", 0)),
        "detected_faces_count": int(raw_face_stats.get("detected_faces_count", 0)),
        "face_detection_rate": round(float(raw_face_stats.get("face_detection_rate", 0.0)), 4),
        "fallback_used": bool(raw_face_stats.get("fallback_used", False)),
    }
    clean_data = {
        "report_version": "1.0",
        "generated_at": str(analysis_data.get("generated_at", datetime.datetime.now(datetime.timezone.utc).isoformat())),
        "git_commit": str(analysis_data.get("git_commit", "unknown")),
        "video_filename": str(analysis_data.get("video_filename", "unnamed_video.mp4")),
        "model_used": str(analysis_data.get("model_used", "outputs/demo/best_model.keras")),
        "model_provenance": str(analysis_data.get("model_provenance", "prototype_synthetic")),
        "manipulation_probability": round(float(analysis_data.get("p_fake", 0.0)), 4),
        "authentic_probability": round(float(analysis_data.get("p_real", 1.0)), 4),
        "verdict": str(analysis_data.get("verdict", "UNKNOWN")),
        "verdict_label": str(analysis_data.get("label", "UNKNOWN")),
        "decision_threshold": round(float(analysis_data.get("threshold", 0.5)), 2),
        "inconclusive_band": [
            round(float(analysis_data.get("inconclusive_band", (0.4, 0.6))[0]), 2),
            round(float(analysis_data.get("inconclusive_band", (0.4, 0.6))[1]), 2),
        ],
        "confidence_assessment": str(analysis_data.get("confidence", "")),
        "clips_analyzed": int(analysis_data.get("clips_analyzed", 0)),
        "clip_probabilities": [round(float(p), 4) for p in analysis_data.get("clip_probabilities", [])],
        "face_detection_stats": clean_face_stats,
        "legal_and_technical_disclaimer": (
            "Model probability reflects neural network estimation of manipulation artifacts, "
            "not definitive legal proof. Prototype model results on real face videos are NOT meaningful."
        ),
    }
    return json.dumps(clean_data, indent=2, default=str)


def build_html_report(analysis_data: Dict[str, Any]) -> str:
    """Generate a self-contained, themed HTML forensic report in Ultraviolet Forensics colors."""
    v_info = analysis_data.get("verdict_info", {})
    color = v_info.get("color", THEME["authentic"])
    icon = v_info.get("icon", "✓")
    label = v_info.get("label", "AUTHENTIC")
    p_fake = float(analysis_data.get("p_fake", 0.0))
    p_real = float(analysis_data.get("p_real", 1.0))
    is_proto = "demo" in analysis_data.get("model_used", "") or analysis_data.get("model_provenance") == "prototype_synthetic"

    proto_banner = ""
    if is_proto:
        proto_banner = f"""
        <div style="background: rgba(255, 194, 71, 0.15); border-left: 4px solid {THEME['inconclusive']}; padding: 12px 16px; margin: 16px 0; border-radius: 6px; color: {THEME['text_primary']};">
            <b style="color: {THEME['inconclusive']};">PROTOTYPE DEMONSTRATION MODEL</b><br>
            <span style="font-size: 0.88em; color: {THEME['text_muted']};">
                This analysis was generated with a prototype model trained on synthetic demo data. Results on real face videos are NOT meaningful.
            </span>
        </div>
        """

    clips_rows = ""
    for idx, p in enumerate(analysis_data.get("clip_probabilities", [])):
        clip_verdict = "MANIPULATED" if p >= analysis_data.get("threshold", 0.5) else "AUTHENTIC"
        clip_color = THEME["manipulated"] if clip_verdict == "MANIPULATED" else THEME["authentic"]
        clips_rows += f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid {THEME['panel_border']};">Segment {idx + 1}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid {THEME['panel_border']}; font-family: monospace;">{p:.4f} ({p*100:.1f}%)</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid {THEME['panel_border']}; color: {clip_color}; font-weight: bold;">{clip_verdict}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Forensic Video Analysis Report - {analysis_data.get('video_filename', 'Video')}</title>
    <style>
        body {{
            background-color: {THEME['bg_dark']};
            color: {THEME['text_primary']};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .report-box {{
            max-width: 800px;
            margin: 0 auto;
            background: {THEME['panel_dark']};
            border: 1px solid {THEME['panel_border']};
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(42, 18, 72, 0.6);
        }}
        h1 {{ margin: 0 0 4px 0; color: {THEME['text_primary']}; font-size: 1.8rem; }}
        .subtitle {{ color: {THEME['text_muted']}; font-size: 0.95rem; margin-bottom: 20px; }}
        .verdict-card {{
            border: 2px solid {color};
            background: rgba(58, 26, 99, 0.9);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }}
        .verdict-title {{ font-size: 2rem; font-weight: bold; color: {color}; }}
        .meta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin: 20px 0;
            font-size: 0.9rem;
        }}
        .meta-item {{ background: {THEME['bg_dark']}; padding: 10px 14px; border-radius: 6px; border: 1px solid {THEME['panel_border']}; }}
        .meta-label {{ color: {THEME['text_muted']}; font-size: 0.8rem; text-transform: uppercase; }}
        .meta-value {{ color: {THEME['text_primary']}; font-weight: 600; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
        th {{ text-align: left; padding: 8px 12px; background: {THEME['bg_dark']}; color: {THEME['text_muted']}; font-size: 0.8rem; }}
        .disclaimer {{
            font-size: 0.8rem;
            color: {THEME['text_muted']};
            margin-top: 30px;
            padding-top: 14px;
            border-top: 1px solid {THEME['panel_border']};
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="report-box">
        <h1>Forensic Video Analysis Report</h1>
        <div class="subtitle">Spatio-temporal manipulation diagnostic assessment</div>

        {proto_banner}

        <div class="verdict-card">
            <div class="verdict-title">{icon} {label}</div>
            <div style="margin-top: 8px; font-size: 1.1rem;">
                Manipulation Probability: <b>{p_fake*100:.1f}%</b> | Authentic Probability: <b>{p_real*100:.1f}%</b>
            </div>
            <div style="font-size: 0.88rem; color: {THEME['text_muted']}; margin-top: 6px;">
                {v_info.get('confidence', '')}
            </div>
        </div>

        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Target Video</div>
                <div class="meta-value">{analysis_data.get('video_filename', 'N/A')}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluation Date (UTC)</div>
                <div class="meta-value">{analysis_data.get('generated_at', 'N/A')}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Model Checkpoint</div>
                <div class="meta-value" style="font-family: monospace; font-size: 0.85em;">{analysis_data.get('model_used', 'N/A')}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Git Commit</div>
                <div class="meta-value" style="font-family: monospace;">{analysis_data.get('git_commit', 'unknown')}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Decision Threshold</div>
                <div class="meta-value">{analysis_data.get('threshold', 0.5):.2f} (Inconclusive band: {analysis_data.get('inconclusive_band', (0.4, 0.6))})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Clips Analyzed</div>
                <div class="meta-value">{analysis_data.get('clips_analyzed', 0)} segments</div>
            </div>
        </div>

        <h3 style="color: {THEME['primary_accent']}; font-size: 1.05rem; margin-top: 24px;">Segment Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Segment</th>
                    <th>P(Manipulation)</th>
                    <th>Segment Verdict</th>
                </tr>
            </thead>
            <tbody>
                {clips_rows}
            </tbody>
        </table>

        <div class="disclaimer">
            <b>Forensic Limitations & Technical Notice:</b><br>
            Outputs generated by this system represent statistical model probabilities of visual and temporal manipulation artifacts,
            not incontrovertible legal proof. Real face evaluation requires training on the verified Celeb-DF v2 benchmark.
        </div>
    </div>
</body>
</html>
"""
    return html


def build_probability_gauge(
    p_fake: float,
    inconclusive_low: float = 0.40,
    inconclusive_high: float = 0.60,
) -> go.Figure:
    """Create a sleek Plotly gauge indicator reflecting the Ultraviolet Forensics theme."""
    p_fake = float(max(0.0, min(1.0, p_fake)))
    low_band, high_band = sorted([inconclusive_low, inconclusive_high])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_fake * 100.0,
            number={"suffix": "%", "font": {"color": THEME["text_primary"], "size": 36, "family": "Space Grotesk, sans-serif"}},
            title={
                "text": "<b>P(MANIPULATION)</b><br><span style='font-size:0.75em;color:#B9A4D6'>Model confidence score</span>",
                "font": {"color": THEME["text_muted"], "size": 13, "family": "Space Grotesk, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": THEME["panel_border"],
                    "tickfont": {"color": THEME["text_muted"], "size": 10},
                },
                "bar": {"color": THEME["text_primary"], "thickness": 0.25},
                "bgcolor": THEME["panel_dark"],
                "borderwidth": 1,
                "bordercolor": THEME["panel_border"],
                "steps": [
                    {"range": [0, low_band * 100], "color": "rgba(196, 161, 255, 0.40)"},
                    {"range": [low_band * 100, high_band * 100], "color": "rgba(255, 194, 71, 0.40)"},
                    {"range": [high_band * 100, 100], "color": "rgba(255, 90, 54, 0.40)"},
                ],
                "threshold": {
                    "line": {"color": THEME["primary_accent"], "width": 3},
                    "thickness": 0.8,
                    "value": p_fake * 100.0,
                },
            },
        )
    )

    fig.update_layout(
        template=None,
        paper_bgcolor=THEME["panel_dark"],
        plot_bgcolor=THEME["panel_dark"],
        margin=dict(l=20, r=20, t=50, b=20),
        height=220,
        font=dict(color=THEME["text_primary"]),
        hoverlabel=dict(
            bgcolor=THEME["panel_dark"],
            bordercolor=THEME["panel_border"],
            font=dict(color=THEME["text_primary"]),
        ),
    )
    return fig


def build_clips_bar_chart(
    clip_probabilities: List[float],
    threshold: float = 0.5,
) -> go.Figure:
    """Create a Plotly bar chart displaying per-clip manipulation probabilities."""
    labels = [f"Segment {i+1}" for i in range(len(clip_probabilities))]
    scores = [p * 100.0 for p in clip_probabilities]
    colors = [THEME["manipulated"] if p >= threshold else THEME["authentic"] for p in clip_probabilities]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(color=colors, line=dict(color=THEME["panel_border"], width=1)),
            text=[f"{s:.1f}%" for s in scores],
            textposition="outside",
            textfont=dict(color=THEME["text_primary"], size=11),
            hoverinfo="x+text",
        )
    )

    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(clip_probabilities) - 0.5,
        y0=threshold * 100.0,
        y1=threshold * 100.0,
        line=dict(color=THEME["inconclusive"], width=2, dash="dash"),
    )

    fig.update_layout(
        template=None,
        title={
            "text": "<b>Per-Clip Manipulation Probability</b>",
            "font": {"color": THEME["text_primary"], "size": 13, "family": "Space Grotesk, sans-serif"},
        },
        paper_bgcolor=THEME["panel_dark"],
        plot_bgcolor=THEME["bg_dark"],
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
        hoverlabel=dict(
            bgcolor=THEME["panel_dark"],
            bordercolor=THEME["panel_border"],
            font=dict(color=THEME["text_primary"]),
        ),
        yaxis=dict(
            range=[0, 120],
            gridcolor=THEME["panel_border"],
            linecolor=THEME["panel_border"],
            tickfont=dict(color=THEME["text_muted"], size=10),
            title=dict(text="P(fake) %", font=dict(color=THEME["text_muted"], size=10)),
        ),
        xaxis=dict(
            gridcolor=THEME["panel_border"],
            linecolor=THEME["panel_border"],
            tickfont=dict(color=THEME["text_primary"], size=11),
        ),
    )
    return fig


def extract_clips_with_diagnostics(
    video_path: str,
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    no_face_detect: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, Any]]:
    """Extract temporal clips and diagnostic telemetry (face counts, fallback usage, sample crops).

    Returns:
        (clips_array, error_reason, diagnostics_dict)
    """
    if not os.path.exists(video_path):
        return None, "File does not exist", {}

    if os.path.getsize(video_path) == 0:
        return None, "empty file", {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "unreadable", {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len:
        cap.release()
        return None, "too short", {"total_frames": total_frames}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    face_cascade = None if no_face_detect else get_face_cascade()
    valid_clips: List[np.ndarray] = []
    any_frame_read = False
    detected_faces_count = 0
    total_frames_inspected = 0
    fallback_used = False
    sample_crops: List[np.ndarray] = []
    clip_frames_meta: List[Dict[str, Any]] = []

    for c in range(clips_per_video):
        seg_start = int(c * total_frames / clips_per_video)
        seg_end = int((c + 1) * total_frames / clips_per_video)
        if seg_end <= seg_start:
            continue

        frame_indices = np.linspace(seg_start, seg_end - 1, seq_len, dtype=int)
        raw_frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            any_frame_read = True
            raw_frames.append(frame)

        if len(raw_frames) != seq_len:
            continue

        total_frames_inspected += seq_len

        if no_face_detect:
            processed_frames = []
            for f in raw_frames:
                crop = center_crop_and_resize(f, img_size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb)
                if len(sample_crops) < 6:
                    sample_crops.append(rgb)
            valid_clips.append(np.array(processed_frames, dtype=np.uint8))
            fallback_used = True
            if c == 0 and not clip_frames_meta:
                for step_i in range(len(processed_frames)):
                    f_idx = int(frame_indices[step_i])
                    ts_val = float(f_idx / fps) if fps > 0 else 0.0
                    clip_frames_meta.append({
                        "step": step_i,
                        "frame_idx": f_idx,
                        "timestamp": ts_val,
                        "crop": processed_frames[step_i],
                    })
        else:
            bboxes = []
            for f in raw_frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                detected = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                if len(detected) > 0:
                    largest = max(detected, key=lambda b: b[2] * b[3])
                    bboxes.append(tuple(largest))
                    detected_faces_count += 1
                else:
                    bboxes.append(None)

            valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
            if not valid_indices:
                continue

            if len(valid_indices) < seq_len:
                fallback_used = True

            filled_bboxes = []
            last_valid = bboxes[valid_indices[0]]
            for b in bboxes:
                if b is not None:
                    last_valid = b
                filled_bboxes.append(last_valid)

            processed_frames = []
            for f, bbox in zip(raw_frames, filled_bboxes):
                crop = crop_face_with_margin(f, bbox, face_margin, img_size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb)
                if len(sample_crops) < 6:
                    sample_crops.append(rgb)

            valid_clips.append(np.array(processed_frames, dtype=np.uint8))
            if c == 0 and not clip_frames_meta:
                for step_i in range(len(processed_frames)):
                    f_idx = int(frame_indices[step_i])
                    ts_val = float(f_idx / fps) if fps > 0 else 0.0
                    clip_frames_meta.append({
                        "step": step_i,
                        "frame_idx": f_idx,
                        "timestamp": ts_val,
                        "crop": processed_frames[step_i],
                    })

    cap.release()

    diagnostics = {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "frames_inspected": total_frames_inspected,
        "detected_faces_count": detected_faces_count,
        "face_detection_rate": (detected_faces_count / max(1, total_frames_inspected)) if not no_face_detect else 0.0,
        "fallback_used": fallback_used,
        "sample_crops": sample_crops,
        "clip_frames_meta": clip_frames_meta,
    }

    if len(valid_clips) == 0:
        if not any_frame_read:
            return None, "unreadable", diagnostics
        if not no_face_detect and detected_faces_count == 0:
            return None, "no face detected", diagnostics
        return None, "too short", diagnostics

    return np.array(valid_clips, dtype=np.uint8), None, diagnostics


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Inspect video file metadata without performing heavy extraction."""
    if not os.path.exists(video_path):
        return {}
    size_bytes = os.path.getsize(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "filename": os.path.basename(video_path),
            "filesize_bytes": size_bytes,
            "filesize_mb": size_bytes / (1024 * 1024),
            "readable": False,
        }
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = total_frames / fps if fps > 0 else 0.0
    return {
        "filename": os.path.basename(video_path),
        "filesize_bytes": size_bytes,
        "filesize_mb": size_bytes / (1024 * 1024),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}" if width > 0 and height > 0 else "Unknown",
        "duration_sec": round(duration, 2),
        "readable": True,
    }


def build_temporal_attention_chart(
    attention_scores: np.ndarray,
    top_indices: Optional[List[int]] = None,
) -> go.Figure:
    """Create a Plotly bar chart displaying temporal attention coefficients across frames."""
    scores = np.asarray(attention_scores, dtype=float).flatten()
    seq_len = len(scores)
    labels = [f"Frame {i+1}" for i in range(seq_len)]

    top_set = set(top_indices) if top_indices is not None else set(np.argsort(scores)[::-1][:3])
    colors = [THEME["manipulated"] if i in top_set else THEME["primary_accent"] for i in range(seq_len)]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition="auto",
            textfont=dict(color=THEME["text_primary"]),
        )
    )

    fig.update_layout(
        template=None,
        paper_bgcolor=THEME["panel_dark"],
        plot_bgcolor=THEME["panel_dark"],
        margin=dict(l=20, r=20, t=30, b=30),
        height=220,
        font=dict(family="Space Grotesk, sans-serif", color=THEME["text_muted"], size=11),
        xaxis=dict(gridcolor="rgba(91, 46, 145, 0.2)", tickfont=dict(color=THEME["text_muted"], size=10)),
        yaxis=dict(gridcolor="rgba(91, 46, 145, 0.2)", title="Attention Weight", tickfont=dict(color=THEME["text_muted"], size=10)),
    )
    return fig

