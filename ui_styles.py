"""Ultraviolet Forensics styling and visual asset module for Deepfake Detector.

Provides custom CSS, theme variables, animations, and inline SVG assets
conforming strictly to the Ultraviolet Forensics specification:
- Background: #2A1248 (Deep Plum)
- Panels & Cards: #3A1A63 (Rich Plum)
- Borders & Outlines: #5B2E91 (Ultraviolet Border)
- Text: #F6EEFF (Crisp Lilac-White)
- Muted Text: #B9A4D6 (Lavender-Violet)
- Primary Accent / Neutral Controls: #8B5CF6 (Ultraviolet)
- Authentic: #C4A1FF (Lilac)
- Manipulated: #FF5A36 (Vermilion)
- Inconclusive: #FFC247 (Gold)
- Gradient: #8B5CF6 -> #FF5A36

HARD CONSTRAINT: Strictly zero black, blue, or green anywhere in the UI.
"""

from theme import THEME

FORENSIC_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {{
    --bg-dark: {THEME["bg_dark"]};
    --panel-dark: {THEME["panel_dark"]};
    --panel-border: {THEME["panel_border"]};
    --text-primary: {THEME["text_primary"]};
    --text-muted: {THEME["text_muted"]};
    --primary-accent: {THEME["primary_accent"]};
    --authentic-lilac: {THEME["authentic"]};
    --manipulated-vermilion: {THEME["manipulated"]};
    --inconclusive-gold: {THEME["inconclusive"]};
    --gradient-glitch: linear-gradient(135deg, {THEME["gradient_start"]} 0%, {THEME["gradient_end"]} 100%);
    --gradient-card: linear-gradient(180deg, rgba(58, 26, 99, 0.90) 0%, rgba(42, 18, 72, 0.95) 100%);
}}

/* Global Font and Base Overrides */
html, body, [class*="css"], .stApp {{
    background-color: var(--bg-dark);
}}

html, body, [class*="css"], .stApp, * {{
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}}

div[data-st-baseweb-layer-host="true"],
div[data-st-overlay-root="true"] {{
    background-color: transparent !important;
}}

code, kbd, samp, pre,
div[data-testid="stCodeBlock"],
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCodeBlock"] code,
div[data-testid="stCodeBlock"] span,
div[data-testid="stCode"],
.stCode {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: var(--panel-dark) !important;
    color: var(--authentic-lilac) !important;
    border: 1px solid var(--panel-border) !important;
}}

code {{
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid var(--panel-border) !important;
}}

/* Glitch Title Effect with Reduced Motion Safety */
.forensic-title-container {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 4px;
}}

.glitch-title {{
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    position: relative;
    display: inline-block;
    text-shadow: -1px -1px 0px rgba(139, 92, 246, 0.6), 1px 1px 0px rgba(255, 90, 54, 0.6);
}}

@keyframes titleGlitch {{
    0% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
    49% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
    50% {{
        text-shadow: 2px -1px {THEME["gradient_start"]}, -2px 1px {THEME["gradient_end"]};
    }}
    52% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
    90% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
    91% {{
        text-shadow: -2px 1px {THEME["gradient_start"]}, 2px -1px {THEME["gradient_end"]};
    }}
    93% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
    100% {{
        text-shadow: -1px 0 {THEME["gradient_start"]}, 1px 0 {THEME["gradient_end"]};
    }}
}}

@media (prefers-reduced-motion: no-preference) {{
    .glitch-title {{
        animation: titleGlitch 6s infinite ease-in-out;
    }}
}}

.forensic-subtitle {{
    font-size: 0.98rem;
    color: var(--text-muted);
    margin-top: -4px;
    margin-bottom: 20px;
    letter-spacing: 0.02em;
}}

/* Status Chips Container */
.status-chips-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}}

.status-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-primary);
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 6px rgba(42, 18, 72, 0.45);
}}

.status-chip-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}}

.dot-uv {{ background-color: var(--primary-accent); box-shadow: 0 0 6px var(--primary-accent); }}
.dot-lilac {{ background-color: var(--authentic-lilac); box-shadow: 0 0 6px var(--authentic-lilac); }}
.dot-vermilion {{ background-color: var(--manipulated-vermilion); box-shadow: 0 0 6px var(--manipulated-vermilion); }}
.dot-gold {{ background-color: var(--inconclusive-gold); box-shadow: 0 0 6px var(--inconclusive-gold); }}

/* Persistent Prototype Warning Banner */
.prototype-warning-banner {{
    background: linear-gradient(135deg, rgba(255, 194, 71, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
    border: 1px solid rgba(255, 194, 71, 0.4);
    border-left: 5px solid var(--inconclusive-gold);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 22px;
    backdrop-filter: blur(10px);
}}

.prototype-warning-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--inconclusive-gold);
    font-weight: 700;
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

.prototype-warning-body {{
    font-size: 0.86rem;
    color: var(--text-muted);
    line-height: 1.45;
    margin: 0;
}}

/* Glass-style Cards */
.glass-panel {{
    background: var(--gradient-card);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(42, 18, 72, 0.45);
}}

/* Verdict Display Banners (Verdict colors strictly reserved) */
.verdict-banner {{
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 18px 0 24px 0;
    backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(42, 18, 72, 0.55);
}}

.verdict-banner::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-glitch);
}}

.verdict-banner-fake {{
    background: linear-gradient(180deg, rgba(255, 90, 54, 0.16) 0%, rgba(58, 26, 99, 0.95) 100%);
    border: 2px solid var(--manipulated-vermilion);
}}

.verdict-banner-real {{
    background: linear-gradient(180deg, rgba(196, 161, 255, 0.16) 0%, rgba(58, 26, 99, 0.95) 100%);
    border: 2px solid var(--authentic-lilac);
}}

.verdict-banner-inconclusive {{
    background: linear-gradient(180deg, rgba(255, 194, 71, 0.16) 0%, rgba(58, 26, 99, 0.95) 100%);
    border: 2px solid var(--inconclusive-gold);
}}

.verdict-headline {{
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}}

.verdict-headline-fake {{ color: var(--manipulated-vermilion); }}
.verdict-headline-real {{ color: var(--authentic-lilac); }}
.verdict-headline-inconclusive {{ color: var(--inconclusive-gold); }}

.verdict-confidence-text {{
    font-size: 0.92rem;
    color: var(--text-muted);
    margin-top: 6px;
    font-style: italic;
}}

/* Video Scanning Preview Wrapper (No Black) */
.video-preview-wrapper {{
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--panel-border);
    background: var(--panel-dark);
}}

video,
div[data-testid="stVideo"] video {{
    background-color: var(--panel-dark) !important;
}}

video::-webkit-media-controls-panel,
video::-webkit-media-controls-enclosure {{
    background-color: var(--panel-dark) !important;
}}

.scanner-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    background: linear-gradient(
        to bottom,
        rgba(139, 92, 246, 0) 0%,
        rgba(139, 92, 246, 0.12) 50%,
        rgba(255, 90, 54, 0.25) 51%,
        rgba(139, 92, 246, 0) 55%
    );
    background-size: 100% 200%;
}}

@keyframes scanAnimation {{
    0% {{ background-position: 0% 0%; }}
    100% {{ background-position: 0% 200%; }}
}}

@media (prefers-reduced-motion: no-preference) {{
    .scanner-overlay {{
        animation: scanAnimation 4s linear infinite;
    }}
}}

/* Face Crop Strip */
.face-crop-strip-container {{
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 10px 4px;
    scrollbar-width: thin;
    scrollbar-color: var(--primary-accent) var(--panel-dark);
}}

.face-crop-thumbnail {{
    flex: 0 0 auto;
    width: 80px;
    height: 80px;
    border-radius: 8px;
    border: 1px solid var(--panel-border);
    object-fit: cover;
    box-shadow: 0 2px 8px rgba(42, 18, 72, 0.5);
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.face-crop-thumbnail:hover {{
    transform: scale(1.08);
    border-color: var(--primary-accent);
}}

/* Styled Streamlit Tabs (Neutral Controls use Ultraviolet) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: var(--panel-dark);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid var(--panel-border);
}}

.stTabs [data-baseweb="tab"] {{
    color: var(--text-muted);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
}}

.stTabs [aria-selected="true"] {{
    background: rgba(139, 92, 246, 0.18) !important;
    color: var(--primary-accent) !important;
    font-weight: 700 !important;
    border: 1px solid var(--primary-accent) !important;
}}

/* Primary & Neutral Button Styling (Ultraviolet Accent) */
div.stButton > button[kind="primary"] {{
    background: var(--primary-accent) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
    font-weight: 700;
    letter-spacing: 0.03em;
    border-radius: 10px;
    padding: 10px 20px;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.35);
    transition: all 0.25s ease;
}}

div.stButton > button[kind="primary"]:hover {{
    background: #9D71F7 !important;
    box-shadow: 0 0 24px rgba(139, 92, 246, 0.6);
    transform: translateY(-1px);
}}

div.stButton > button[kind="secondary"], div.stDownloadButton > button {{
    background: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px;
    transition: all 0.2s ease;
}}

div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button:hover {{
    border-color: var(--primary-accent) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.3);
}}

/* Sidebar Custom Styling (Plum panel color, not darker near-black) */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarNav"] {{
    background-color: var(--panel-dark) !important;
    border-right: 1px solid var(--panel-border) !important;
}}

/* Custom Scrollbars */
::-webkit-scrollbar {{
    width: 8px !important;
    height: 8px !important;
    background: var(--bg-dark) !important;
}}
::-webkit-scrollbar-track {{
    background: var(--bg-dark) !important;
}}
::-webkit-scrollbar-thumb {{
    background: var(--panel-border) !important;
    border-radius: 4px !important;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--primary-accent) !important;
}}
* {{
    scrollbar-color: var(--panel-border) var(--bg-dark) !important;
    scrollbar-width: thin !important;
}}

/* Dropdown and Selectbox Styling */
div[data-baseweb="select"] > div {{
    background-color: var(--panel-dark) !important;
    border-color: var(--panel-border) !important;
    color: var(--text-primary) !important;
}}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
}}
li[role="option"] {{
    background-color: var(--panel-dark) !important;
    color: var(--text-primary) !important;
}}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {{
    background-color: var(--panel-border) !important;
    color: var(--text-primary) !important;
}}

/* File Uploader Customization */
section[data-testid="stFileUploadDropzone"],
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"] {{
    background-color: var(--panel-dark) !important;
    border: 2px dashed var(--panel-border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
}}

section[data-testid="stFileUploadDropzone"]:hover,
div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--primary-accent) !important;
}}

section[data-testid="stFileUploadDropzone"] small {{
    color: var(--text-muted) !important;
}}

/* Slider Neutral Control Styling */
div[data-baseweb="slider"] div[role="slider"] {{
    background-color: var(--primary-accent) !important;
    border-color: var(--primary-accent) !important;
}}

div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] {{
    background-color: var(--panel-border) !important;
}}

div[data-baseweb="slider"] div[data-testid="stSliderTrack"] > div {{
    background: var(--primary-accent) !important;
}}

/* Input and Selectbox Focus Overrides */
*:focus, *:focus-visible, [data-baseweb="input"] input:focus, .stSelectbox:focus-within {{
    outline-color: var(--primary-accent) !important;
    border-color: var(--primary-accent) !important;
    box-shadow: 0 0 0 1px var(--primary-accent) !important;
}}

/* Top Header Toolbar & Deploy Menu Area */
header[data-testid="stHeader"] {{
    background-color: var(--bg-dark) !important;
}}
header[data-testid="stHeader"] * {{
    color: var(--text-muted) !important;
}}
div[data-testid="stDecoration"] {{
    display: none !important;
}}
div[data-testid="stStatusWidget"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
}}

/* Toast Notifications */
div[data-testid="stToast"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 4px 16px rgba(42, 18, 72, 0.6) !important;
}}
div[data-testid="stToast"] * {{
    color: var(--text-primary) !important;
}}

/* Plotly Chart Elements */
.js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}
.js-plotly-plot .plotly .modebar-btn path {{
    fill: var(--primary-accent) !important;
}}
.js-plotly-plot .plotly .modebar-btn:hover path {{
    fill: var(--authentic-lilac) !important;
}}
.hoverlayer .hovertext rect {{
    fill: var(--panel-dark) !important;
    stroke: var(--panel-border) !important;
}}
.hoverlayer .hovertext text {{
    fill: var(--text-primary) !important;
}}

/* Progress Bar Accent */
.stProgress > div > div > div > div {{
    background: var(--gradient-glitch) !important;
}}

/* Expander and Dataframe Overrides */
div[data-testid="stExpander"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 10px !important;
}}

div[data-testid="stDataFrame"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 8px !important;
}}

/* Streamlit Alert/Status Containers */
div[data-testid="stAlert"] {{
    background-color: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
}}
</style>
"""

APP_LOGO_SVG = f"""
<svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="48" height="48" rx="10" fill="{THEME['panel_dark']}" stroke="{THEME['primary_accent']}" stroke-width="1.5"/>
    <path d="M24 10L36 15V23C36 30.5 30.88 37.45 24 39C17.12 37.45 12 30.5 12 23V15L24 10Z"
          fill="url(#shield_uv_grad)" stroke="{THEME['primary_accent']}" stroke-width="2" stroke-linejoin="round"/>
    <path d="M20 23L23 26L28 20" stroke="{THEME['authentic']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="16" y1="21" x2="32" y2="21" stroke="{THEME['manipulated']}" stroke-width="1" stroke-dasharray="2 2" opacity="0.85"/>
    <line x1="16" y1="27" x2="32" y2="27" stroke="{THEME['manipulated']}" stroke-width="1" stroke-dasharray="2 2" opacity="0.85"/>
    <defs>
        <linearGradient id="shield_uv_grad" x1="12" y1="10" x2="36" y2="39" gradientUnits="userSpaceOnUse">
            <stop stop-color="{THEME['panel_dark']}"/>
            <stop offset="1" stop-color="{THEME['panel_border']}"/>
        </linearGradient>
    </defs>
</svg>
"""

PIPELINE_FLOW_HTML = f"""
<div style="background: {THEME['panel_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 14px; padding: 24px; margin-bottom: 24px;">
    <h3 style="margin-top: 0; color: {THEME['primary_accent']}; font-size: 1.15rem; font-weight: 700;">
        End-to-End Spatio-Temporal Pipeline Architecture
    </h3>
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-top: 18px;">
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">1</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">Video Input</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">3 segments sampled uniformly</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">2</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">Face Cascade</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">Haar frontal detection + fallback</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">3</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">Spatial Crop</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">128x128 with 25% margin</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">4</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">CNN Feature</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">4-block TimeDistributed CNN</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">5</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">LSTM Modeling</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">128-unit temporal recurrent layer</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: {THEME['bg_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="display: inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%; background: {THEME['primary_accent']}; color: {THEME['text_primary']}; font-weight: bold; font-size: 0.85rem; margin-bottom: 6px;">6</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.88rem;">P(manipulation)</div>
            <div style="font-size: 0.75rem; color: {THEME['text_muted']};">Mean clip score + band filter</div>
        </div>
    </div>
</div>
"""
