"""DeepTrace SaaS Forensic Styling and Visual Asset Module.

Custom CSS, theme variables, and inline SVG assets conforming to the
reference SaaS forensic platform aesthetic:
- Primary Background: #090D16 (Deep Navy / Near-Black)
- Secondary Background: #0E1526 (Slightly Lighter Navy)
- Cards & Surfaces: #131B2E (Dark Blue-Gray)
- Borders & Dividers: #243048 (Subtle Cool Gray/Blue)
- Primary Accent: #06B6D4 (Electric Cyan / Blue)
- Secondary Accent: #38BDF8 (Electric Sky Blue)
- Violet Accent: #8B5CF6 (Sleek Violet)
- Text Primary: #FFFFFF (Crisp White)
- Text Muted: #94A3B8 (Muted Blue-Gray)
- Authentic / REAL: #10B981 (Restrained Emerald Green)
- Manipulated / FAKE: #EF4444 (Restrained Coral Red)
- Inconclusive: #F59E0B (Restrained Amber)
"""

from theme import THEME

FORENSIC_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {{
    --bg-dark: {THEME["bg_dark"]};
    --bg-secondary: {THEME["bg_secondary"]};
    --panel-dark: {THEME["panel_dark"]};
    --panel-border: {THEME["panel_border"]};
    --text-primary: {THEME["text_primary"]};
    --text-muted: {THEME["text_muted"]};
    --primary-accent: {THEME["primary_accent"]};
    --accent-electric: {THEME["accent_electric"]};
    --accent-violet: {THEME["accent_violet"]};
    --authentic-green: {THEME["authentic"]};
    --manipulated-coral: {THEME["manipulated"]};
    --inconclusive-amber: {THEME["inconclusive"]};
    --gradient-cta: {THEME["gradient_cta"]};
    --gradient-card: linear-gradient(180deg, rgba(19, 27, 46, 0.95) 0%, rgba(14, 21, 38, 0.98) 100%);
}}

/* Global Base and Background */
html, body, [class*="css"], .stApp {{
    background-color: var(--bg-dark);
    background-image:
        radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.05) 0%, transparent 60%),
        radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.04) 0%, transparent 50%);
    background-attachment: fixed;
}}

html, body, [class*="css"], .stApp, * {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}}

div[data-st-baseweb-layer-host="true"],
div[data-st-overlay-root="true"] {{
    background-color: transparent !important;
}}

/* Code Blocks */
code, kbd, samp, pre,
div[data-testid="stCodeBlock"],
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCodeBlock"] code,
div[data-testid="stCodeBlock"] span,
div[data-testid="stCode"],
.stCode {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: var(--bg-secondary) !important;
    color: var(--accent-electric) !important;
    border: 1px solid var(--panel-border) !important;
}}

code {{
    padding: 2px 6px !important;
    border-radius: 4px !important;
}}

/* Top Header Shell */
.deeptrace-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
}}

.deeptrace-brand-col {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.deeptrace-title {{
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1.1;
    font-family: 'Space Grotesk', sans-serif;
}}

.deeptrace-title span {{
    background: linear-gradient(135deg, var(--accent-electric) 0%, var(--accent-violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.deeptrace-sub {{
    font-size: 0.76rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 3px;
}}

.status-online-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--authentic-green);
}}

.status-offline-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--manipulated-coral);
}}

.status-chip-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}}

.dot-green {{ background-color: var(--authentic-green); box-shadow: 0 0 6px var(--authentic-green); }}
.dot-red {{ background-color: var(--manipulated-coral); box-shadow: 0 0 6px var(--manipulated-coral); }}
.dot-amber {{ background-color: var(--inconclusive-amber); box-shadow: 0 0 6px var(--inconclusive-amber); }}
.dot-cyan {{ background-color: var(--primary-accent); box-shadow: 0 0 6px var(--primary-accent); }}
.dot-violet {{ background-color: var(--accent-violet); box-shadow: 0 0 6px var(--accent-violet); }}
.dot-uv {{ background-color: var(--primary-accent); box-shadow: 0 0 6px var(--primary-accent); }}
.dot-lilac {{ background-color: var(--authentic-green); box-shadow: 0 0 6px var(--authentic-green); }}
.dot-vermilion {{ background-color: var(--manipulated-coral); box-shadow: 0 0 6px var(--manipulated-coral); }}
.dot-gold {{ background-color: var(--inconclusive-amber); box-shadow: 0 0 6px var(--inconclusive-amber); }}

/* Status Chips Container */
.status-chips-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
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
}}

/* Top Navigation Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 5px;
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

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.04);
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.16) 0%, rgba(139, 92, 246, 0.16) 100%) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(6, 182, 212, 0.45) !important;
}}

/* Hero Section */
.hero-box {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6, 182, 212, 0.10);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--accent-electric);
    text-transform: uppercase;
    margin-bottom: 12px;
}}

.hero-heading {{
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
    font-family: 'Space Grotesk', sans-serif;
}}

.hero-heading-gradient {{
    background: linear-gradient(135deg, var(--accent-electric) 0%, var(--accent-violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-desc {{
    font-size: 0.95rem;
    color: var(--text-muted);
    margin: 0 0 18px 0;
    line-height: 1.5;
    max-width: 820px;
}}

/* Capability Cards Grid */
.capability-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
}}

.capability-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s ease;
}}

.capability-card:hover {{
    border-color: rgba(6, 182, 212, 0.4);
}}

.capability-card-title {{
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--accent-electric);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}}

.capability-card-desc {{
    font-size: 0.84rem;
    color: var(--text-muted);
    line-height: 1.35;
}}

/* Video Metadata Grid */
.video-meta-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 10px;
    margin: 14px 0 18px 0;
}}

.video-meta-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 10px 14px;
}}

.video-meta-label {{
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 3px;
    font-weight: 600;
}}

.video-meta-value {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}}

/* Primary & Action Buttons */
div.stButton > button[kind="primary"] {{
    background: var(--gradient-cta) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.25) !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 22px rgba(6, 182, 212, 0.4) !important;
    transform: translateY(-1px) !important;
}}

div.stButton > button[kind="secondary"], div.stDownloadButton > button {{
    background: var(--panel-dark) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button:hover {{
    border-color: var(--primary-accent) !important;
    color: var(--text-primary) !important;
}}

/* Forensic Result Card */
.result-hero-card {{
    background: var(--panel-dark);
    border-radius: 14px;
    padding: 24px 26px;
    margin: 18px 0 22px 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
}}

.result-hero-card-real {{
    border: 1px solid rgba(16, 185, 129, 0.35);
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.06) 0%, rgba(19, 27, 46, 0.98) 100%);
}}

.result-hero-card-fake {{
    border: 1px solid rgba(239, 68, 68, 0.35);
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.06) 0%, rgba(19, 27, 46, 0.98) 100%);
}}

.result-hero-card-inconclusive {{
    border: 1px solid rgba(245, 158, 11, 0.35);
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.06) 0%, rgba(19, 27, 46, 0.98) 100%);
}}

.verdict-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
}}

.verdict-pill-real {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.16);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: var(--authentic-green);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    padding: 5px 20px;
    border-radius: 10px;
    margin-bottom: 8px;
}}

.verdict-pill-fake {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(239, 68, 68, 0.45);
    color: var(--manipulated-coral);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    padding: 5px 20px;
    border-radius: 10px;
    margin-bottom: 8px;
}}

.verdict-pill-inconclusive {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: var(--inconclusive-amber);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    padding: 5px 20px;
    border-radius: 10px;
    margin-bottom: 8px;
}}

.verdict-subtitle {{
    font-size: 0.92rem;
    color: var(--text-muted);
    margin-bottom: 14px;
}}

.prob-metric-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 14px;
    margin: 8px 0 14px 0;
}}

.prob-metric-title {{
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 600;
}}

.prob-metric-value {{
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1;
}}

.prob-metric-thresh {{
    font-size: 0.90rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}}

.evidence-secondary-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--panel-border);
}}

.evidence-secondary-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 8px 12px;
}}

.evidence-secondary-label {{
    font-size: 0.70rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
}}

.evidence-secondary-val {{
    font-size: 0.90rem;
    color: var(--text-primary);
    font-weight: 700;
    margin-top: 2px;
}}

/* Frame Evidence Viewer Cards */
.frame-evidence-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    overflow: hidden;
    padding: 12px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.frame-evidence-card:hover {{
    border-color: rgba(6, 182, 212, 0.4);
    transform: translateY(-2px);
}}

.frame-evidence-rank {{
    font-size: 0.80rem;
    font-weight: 700;
    color: var(--accent-electric);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}}

.frame-evidence-meta {{
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.4;
}}

.frame-evidence-tag {{
    display: inline-block;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: var(--accent-electric);
    font-size: 0.76rem;
    font-weight: 700;
    border-radius: 6px;
    padding: 2px 8px;
    margin-top: 6px;
}}

/* Forensic Summary Blocks */
.summary-block-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}}

.summary-block-item {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 14px 16px;
}}

.summary-block-title {{
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

.summary-block-value {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
}}

/* Custom Video Player (Dark Navy) */
.video-preview-wrapper {{
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--panel-border);
    background: var(--bg-secondary);
}}

video,
div[data-testid="stVideo"] video {{
    background-color: var(--bg-secondary) !important;
}}

video::-webkit-media-controls-panel,
video::-webkit-media-controls-enclosure {{
    background-color: var(--bg-secondary) !important;
}}

.custom-video-screen {{
    position: relative;
    width: 100%;
    aspect-ratio: 16/9;
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}}

.custom-video-controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: var(--panel-dark);
    border-top: 1px solid var(--panel-border);
}}

#forensic-play-btn {{
    background: var(--primary-accent);
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 700;
    font-size: 0.90rem;
    cursor: pointer;
    transition: opacity 0.2s ease;
}}

#forensic-play-btn:hover {{
    opacity: 0.9;
}}

#forensic-seeker {{
    flex: 1;
    accent-color: var(--primary-accent);
    background: var(--bg-secondary);
    cursor: pointer;
}}

#forensic-time {{
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    font-weight: 600;
}}

/* Disclaimer Banner */
.forensic-disclaimer-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--primary-accent);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 20px 0;
    font-size: 0.86rem;
    color: var(--text-muted);
    line-height: 1.5;
}}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarNav"] {{
    background-color: var(--bg-dark) !important;
    border-right: 1px solid var(--panel-border) !important;
}}

/* File Uploader Customization */
section[data-testid="stFileUploadDropzone"],
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploader"] {{
    background-color: var(--bg-secondary) !important;
    border: 1px dashed var(--panel-border) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
}}

section[data-testid="stFileUploadDropzone"]:hover,
div[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--primary-accent) !important;
}}

/* Dropdowns & Selects */
div[data-baseweb="select"] > div {{
    background-color: var(--bg-secondary) !important;
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
    background-color: var(--bg-secondary) !important;
    color: var(--accent-electric) !important;
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

/* Top Header Toolbar */
header[data-testid="stHeader"] {{
    background-color: var(--bg-dark) !important;
}}
header[data-testid="stHeader"] * {{
    color: var(--text-muted) !important;
}}
div[data-testid="stDecoration"] {{
    display: none !important;
}}

/* Glass-style Cards */
.glass-panel {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
}}

/* Persistent Prototype Warning Banner */
.prototype-warning-banner {{
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 4px solid var(--inconclusive-amber);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
}}

.prototype-warning-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--inconclusive-amber);
    font-weight: 700;
    font-size: 0.90rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

.prototype-warning-body {{
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.45;
    margin: 0;
}}

/* Custom Scrollbars */
::-webkit-scrollbar {{
    width: 6px !important;
    height: 6px !important;
    background: var(--bg-dark) !important;
}}
::-webkit-scrollbar-track {{
    background: var(--bg-dark) !important;
}}
::-webkit-scrollbar-thumb {{
    background: var(--panel-border) !important;
    border-radius: 3px !important;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--primary-accent) !important;
}}

/* Responsive Breakpoints */
@media (max-width: 900px) {{
    .capability-grid {{
        grid-template-columns: 1fr;
    }}
    .deeptrace-header {{
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
    }}
    .prob-metric-value {{
        font-size: 2.2rem;
    }}
}}
</style>
"""

APP_LOGO_SVG = f"""
<svg width="38" height="38" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="48" height="48" rx="10" fill="{THEME['panel_dark']}" stroke="{THEME['panel_border']}" stroke-width="1.5"/>
    <path d="M24 10L36 15V23C36 30.5 30.88 37.45 24 39C17.12 37.45 12 30.5 12 23V15L24 10Z"
          fill="url(#shield_grad)" stroke="{THEME['primary_accent']}" stroke-width="2" stroke-linejoin="round"/>
    <path d="M20 23L23 26L28 20" stroke="{THEME['accent_electric']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <defs>
        <linearGradient id="shield_grad" x1="12" y1="10" x2="36" y2="39" gradientUnits="userSpaceOnUse">
            <stop stop-color="{THEME['bg_secondary']}"/>
            <stop offset="1" stop-color="{THEME['panel_dark']}"/>
        </linearGradient>
    </defs>
</svg>
"""

PIPELINE_FLOW_HTML = f"""
<div style="background: {THEME['panel_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 14px; padding: 22px; margin-bottom: 22px;">
    <h3 style="margin-top: 0; color: {THEME['accent_electric']}; font-size: 1.1rem; font-weight: 700;">
        End-to-End Spatio-Temporal Pipeline Architecture
    </h3>
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-top: 16px;">
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">1</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">Video Input</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">3 segments sampled</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.1rem;">➔</div>
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">2</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">Face Cascade</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">Haar detection + fallback</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.1rem;">➔</div>
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">3</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">Spatial Crop</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">128x128 margin crop</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.1rem;">➔</div>
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">4</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">CNN Features</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">4-block TimeDistributed</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.1rem;">➔</div>
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">5</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">BiLSTM + Attn</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">Temporal Attention (64)</div>
        </div>
        <div style="color: {THEME['primary_accent']}; font-weight: bold; font-size: 1.1rem;">➔</div>
        <div style="flex: 1; min-width: 130px; background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 10px; padding: 12px; text-align: center;">
            <div style="display: inline-block; width: 26px; height: 26px; line-height: 26px; border-radius: 50%; background: {THEME['primary_accent']}; color: #FFFFFF; font-weight: bold; font-size: 0.82rem; margin-bottom: 4px;">6</div>
            <div style="font-weight: 700; color: {THEME['text_primary']}; margin: 2px 0; font-size: 0.85rem;">Verdict</div>
            <div style="font-size: 0.74rem; color: {THEME['text_muted']};">Threshold calibrated (0.39)</div>
        </div>
    </div>
</div>
"""
