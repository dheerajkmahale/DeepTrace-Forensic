"""DeepTrace V3 Forensic Intelligence Workstation - Styles & Markup.

Defines the professional visual design system, custom CSS, layout grids,
SVG badges, and pipeline diagrams conforming strictly to:
- Deep Graphite / Near Black (#070B12) background
- Dark Slate (#0D131D) surface
- Elevated Navy/Slate (#111A26) cards
- Cool Steel Blue (#202C3A) borders
- Electric Cyan (#22D3EE) & Electric Blue (#3B82F6) accents
- Emerald Green (#22C55E) Authentic / Coral Red (#EF4444) Manipulated / Amber (#F59E0B) Inconclusive
"""

from theme import THEME

APP_LOGO_SVG = """
<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="28" height="28" rx="6" fill="#111A26"/>
  <rect x="0.5" y="0.5" width="27" height="27" rx="5.5" stroke="#202C3A"/>
  <path d="M7 14L12 8L16 13L21 6" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="21" cy="6" r="2" fill="#22D3EE"/>
  <circle cx="16" cy="13" r="1.5" fill="#3B82F6"/>
  <circle cx="12" cy="8" r="1.5" fill="#3B82F6"/>
  <path d="M7 21C9.5 19 12 18.5 14 18.5C16 18.5 18.5 19 21 21" stroke="#94A3B8" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""

FORENSIC_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {{
    --bg-dark: {THEME['bg_dark']};
    --bg-surface: {THEME['bg_secondary']};
    --panel-dark: {THEME['panel_dark']};
    --panel-border: {THEME['panel_border']};
    --primary-accent: {THEME['primary_accent']};
    --secondary-accent: {THEME['accent_blue']};
    --text-primary: {THEME['text_primary']};
    --text-muted: {THEME['text_muted']};
    --authentic-green: {THEME['authentic']};
    --manipulated-coral: {THEME['manipulated']};
    --inconclusive-amber: {THEME['inconclusive']};
    --gradient-cta: {THEME['gradient_cta']};
}}

/* Base Page Overrides */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em;
}}

/* Top Header & Chrome Minimization */
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
header[data-testid="stHeader"] {{
    background-color: transparent !important;
    height: 1.5rem !important;
}}

[data-testid="block-container"] {{
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}}

[data-testid="stSidebar"] {{
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--panel-border) !important;
}}

/* Navigation Meta Chips */
.nav-meta-chip {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}}

/* Top Navigation Bar */
.top-nav-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 20px;
}}

.nav-brand-group {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nav-brand-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: var(--text-primary);
    text-transform: uppercase;
}}

.nav-brand-tagline {{
    font-size: 0.75rem;
    color: var(--primary-accent);
    letter-spacing: 0.08em;
    font-weight: 600;
    text-transform: uppercase;
    padding-left: 8px;
    border-left: 1px solid var(--panel-border);
}}

/* Status Pills */
.status-pill-online {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #22C55E;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}

.status-pulse-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.8);
    display: inline-block;
}}

/* Compact Workspace Header */
.workspace-header {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}}

.workspace-title-area {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.workspace-badge {{
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--primary-accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}

.workspace-heading {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}}

.workspace-subtitle {{
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0;
}}

.workspace-status-chips {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}

.status-chip {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}

/* Tabs Navigation Styling */
div[data-testid="stTabs"] button[role="tab"] {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--primary-accent) !important;
    background: rgba(34, 211, 238, 0.05) !important;
}}

div[data-testid="stTabs"] button[role="tab"]:hover {{
    color: var(--text-primary) !important;
}}

/* Two-Column Analysis Workspace */
.workspace-card {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}}

.workspace-card-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--primary-accent);
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

/* Video Intelligence Pre-Analysis Grid */
.intelligence-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
    gap: 8px;
    margin: 14px 0;
}}

.intelligence-card {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 8px 10px;
}}

.intelligence-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}}

.intelligence-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-primary);
    word-break: break-all;
}}

/* Primary Forensic CTA Button */
div.stButton > button[kind="primary"] {{
    background: var(--gradient-cta) !important;
    border: none !important;
    color: #070B12 !important;
    font-size: 1.0rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 18px rgba(34, 211, 238, 0.25) !important;
    transition: all 0.2s ease !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}

div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 6px 24px rgba(34, 211, 238, 0.45) !important;
    transform: translateY(-1px) !important;
}}

div.stButton > button[kind="secondary"], div.stDownloadButton > button {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}

div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button:hover {{
    border-color: var(--primary-accent) !important;
    color: var(--primary-accent) !important;
}}

/* Forensic Verdict Section */
.verdict-master-card {{
    background: var(--panel-dark);
    border-radius: 14px;
    padding: 24px;
    margin: 18px 0 24px 0;
    border: 1px solid var(--panel-border);
    position: relative;
    overflow: hidden;
}}

.verdict-card-authentic {{
    border-color: rgba(34, 197, 94, 0.4);
    background: linear-gradient(180deg, rgba(34, 197, 94, 0.05) 0%, rgba(17, 26, 38, 0.98) 100%);
}}

.verdict-card-manipulated {{
    border-color: rgba(239, 68, 68, 0.4);
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.05) 0%, rgba(17, 26, 38, 0.98) 100%);
}}

.verdict-card-inconclusive {{
    border-color: rgba(245, 158, 11, 0.4);
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.05) 0%, rgba(17, 26, 38, 0.98) 100%);
}}

.verdict-badge-authentic {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.5);
    color: #22C55E;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    padding: 6px 18px;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
}}

.verdict-badge-manipulated {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.16);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #EF4444;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    padding: 6px 18px;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
}}

.verdict-badge-inconclusive {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(245, 158, 11, 0.16);
    border: 1px solid rgba(245, 158, 11, 0.5);
    color: #F59E0B;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    padding: 6px 18px;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
}}

.verdict-explanation-box {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--primary-accent);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: var(--text-primary);
    margin: 12px 0 16px 0;
    line-height: 1.45;
}}

/* Probability Visualizer Meter */
.prob-meter-container {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 14px 0;
}}

.prob-meter-row {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.prob-meter-name {{
    width: 120px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}}

.prob-meter-bar-bg {{
    flex: 1;
    height: 12px;
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 9999px;
    overflow: hidden;
    position: relative;
}}

.prob-meter-bar-fill {{
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}}

.prob-meter-percent {{
    width: 65px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.90rem;
    font-weight: 700;
    text-align: right;
}}

/* Temporal Frame Evidence Cards */
.evidence-card {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    transition: all 0.2s ease;
}}

.evidence-card:hover {{
    border-color: var(--primary-accent);
    transform: translateY(-2px);
}}

.evidence-rank {{
    font-size: 0.70rem;
    font-weight: 700;
    color: var(--primary-accent);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}}

.evidence-meta {{
    font-size: 0.76rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 6px;
    line-height: 1.35;
}}

.evidence-weight-tag {{
    display: inline-block;
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.35);
    color: var(--primary-accent);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Video Timeline Thumbnail Strip */
.timeline-strip-container {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 10px 0;
    margin-top: 10px;
}}

.timeline-frame-item {{
    flex: 0 0 auto;
    width: 68px;
    text-align: center;
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 4px;
    background: var(--bg-surface);
}}

.timeline-frame-item.top-focus {{
    border-color: var(--primary-accent);
    box-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
}}

/* Summary Block Metric Grid */
.summary-block-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
    gap: 10px;
    margin: 16px 0;
}}

.summary-block-item {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 10px 12px;
}}

.summary-block-title {{
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 4px;
}}

.summary-block-value {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}}

/* Benchmark Evaluation Cards */
.benchmark-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    margin: 14px 0;
}}

.benchmark-card {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 12px;
}}

.benchmark-label {{
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}

.benchmark-value {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--primary-accent);
    font-family: 'Space Grotesk', sans-serif;
}}

/* Status Chip Dots */
.status-chip-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}}
.dot-green {{ background: #22C55E; box-shadow: 0 0 6px rgba(34, 197, 94, 0.6); }}
.dot-cyan {{ background: #22D3EE; box-shadow: 0 0 6px rgba(34, 211, 238, 0.6); }}
.dot-amber {{ background: #F59E0B; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }}
.dot-red {{ background: #EF4444; box-shadow: 0 0 6px rgba(239, 68, 68, 0.6); }}
.dot-blue {{ background: #3B82F6; box-shadow: 0 0 6px rgba(59, 130, 246, 0.6); }}

/* Prototype Warning Banner */
.prototype-warning-banner {{
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 14px 0 20px 0;
}}
.prototype-warning-header {{
    font-size: 0.88rem;
    font-weight: 700;
    color: #F59E0B;
    margin-bottom: 4px;
}}
.prototype-warning-body {{
    font-size: 0.82rem;
    color: #94A3B8;
    margin: 0;
    line-height: 1.5;
}}

/* Glass Panels & Disclaimers */
.glass-panel {{
    background: var(--panel-dark);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}}
.forensic-disclaimer-card {{
    background: var(--bg-surface);
    border: 1px solid var(--panel-border);
    border-left: 3px solid var(--primary-accent);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin-top: 16px;
}}
.forensic-footer-bar {{
    text-align: center;
    padding: 24px 0 12px 0;
    font-size: 0.78rem;
    color: var(--text-muted);
    border-top: 1px solid var(--panel-border);
    margin-top: 40px;
}}

/* Custom Premium File Uploader */
[data-testid="stFileUploader"] {{
    background: var(--panel-dark);
    border: 1px dashed var(--panel-border);
    border-radius: 10px;
    padding: 12px;
    transition: border-color 0.2s ease, background 0.2s ease;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: var(--primary-accent);
    background: rgba(34, 211, 238, 0.02);
}}

[data-testid="stFileUploader"] section {{
    background: transparent !important;
    padding: 8px 12px !important;
}}

[data-testid="stFileUploader"] button {{
    border: 1px solid var(--panel-border) !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stFileUploader"] button:hover {{
    border-color: var(--primary-accent) !important;
    color: var(--primary-accent) !important;
}}

/* Custom Video Preview Styling */
[data-testid="stVideo"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--panel-border);
    background: var(--panel-dark);
    margin: 8px 0;
}}

[data-testid="stVideo"] video {{
    border-radius: 8px;
    max-height: 380px;
    width: 100%;
    object-fit: contain;
    background: var(--bg-dark);
}}



/* Responsive Adjustments */
@media (max-width: 768px) {{
    .top-nav-bar {{
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }}
    .workspace-header {{
        flex-direction: column;
        align-items: flex-start;
    }}
    .prob-meter-name {{
        width: 90px;
        font-size: 0.70rem;
    }}
}}
</style>
"""

PIPELINE_FLOW_HTML = f"""
<div style="background: {THEME['panel_dark']}; border: 1px solid {THEME['panel_border']}; border-radius: 12px; padding: 18px; margin: 16px 0;">
    <div style="font-size: 0.72rem; font-weight: 700; color: {THEME['primary_accent']}; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace;">
        FORENSIC PIPELINE ARCHITECTURE (V2 TEMPORAL ATTENTION)
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px;">
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 01</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 2px;">Uniform Sampling</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">10 frames / segment</div>
        </div>
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 02</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 2px;">Face Detection</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">Haar + Interpolation</div>
        </div>
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 03</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 2px;">Spatial CNN</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">4-Block Conv2D</div>
        </div>
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 04</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 2px;">BiLSTM Sequence</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">128 Hidden Units</div>
        </div>
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 05</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['text_primary']}; margin-top: 2px;">Temporal Attention</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">Weighted Pooling</div>
        </div>
        <div style="background: {THEME['bg_secondary']}; border: 1px solid {THEME['panel_border']}; border-radius: 8px; padding: 10px;">
            <div style="font-size: 0.65rem; color: {THEME['text_muted']}; font-weight: 700;">STAGE 06</div>
            <div style="font-size: 0.82rem; font-weight: 700; color: {THEME['primary_accent']}; margin-top: 2px;">Calibrated Verdict</div>
            <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">Cutoff = 0.39</div>
        </div>
    </div>
</div>
"""
