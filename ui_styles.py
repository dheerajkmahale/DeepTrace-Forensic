"""Forensic Glitch styling and visual asset module for Deepfake Detector.

Provides custom CSS, theme variables, animations, and inline SVG assets
conforming to the Forensic Glitch design specification:
- Background: #0A0E1A
- Panels & Cards: #121A2B
- Text: #E6EAF2
- Muted Text: #8B97B1
- Authentic / Real: #2DE2C4 (Scan Teal)
- Manipulated / Fake: #FF3D81 (Glitch Magenta)
- Inconclusive: #FFB020 (Warning Amber)
- Accent Gradient: Cyan (#22D3EE) -> Magenta (#FF3D81)
"""

FORENSIC_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #0A0E1A;
    --panel-dark: #121A2B;
    --panel-border: rgba(34, 211, 238, 0.18);
    --text-primary: #E6EAF2;
    --text-muted: #8B97B1;
    --teal-scan: #2DE2C4;
    --magenta-glitch: #FF3D81;
    --amber-warning: #FFB020;
    --cyan-accent: #22D3EE;
    --gradient-glitch: linear-gradient(135deg, #22D3EE 0%, #FF3D81 100%);
    --gradient-card: linear-gradient(180deg, rgba(18, 26, 43, 0.85) 0%, rgba(10, 14, 26, 0.95) 100%);
}

/* Global Font and Base Overrides */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glitch Title Effect with Reduced Motion Safety */
.forensic-title-container {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 4px;
}

.glitch-title {
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    position: relative;
    display: inline-block;
    text-shadow: -1px -1px 0px rgba(34, 211, 238, 0.6), 1px 1px 0px rgba(255, 61, 129, 0.6);
}

@keyframes titleGlitch {
    0% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
    49% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
    50% {
        text-shadow: 2px -1px #22D3EE, -2px 1px #FF3D81;
    }
    52% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
    90% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
    91% {
        text-shadow: -2px 1px #22D3EE, 2px -1px #FF3D81;
    }
    93% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
    100% {
        text-shadow: -1px 0 #22D3EE, 1px 0 #FF3D81;
    }
}

@media (prefers-reduced-motion: no-preference) {
    .glitch-title {
        animation: titleGlitch 6s infinite ease-in-out;
    }
}

.forensic-subtitle {
    font-size: 0.98rem;
    color: var(--text-muted);
    margin-top: -4px;
    margin-bottom: 20px;
    letter-spacing: 0.02em;
}

/* Status Chips Container */
.status-chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}

.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(18, 26, 43, 0.85);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-primary);
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}

.status-chip-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

.dot-cyan { background-color: var(--cyan-accent); box-shadow: 0 0 6px var(--cyan-accent); }
.dot-teal { background-color: var(--teal-scan); box-shadow: 0 0 6px var(--teal-scan); }
.dot-magenta { background-color: var(--magenta-glitch); box-shadow: 0 0 6px var(--magenta-glitch); }
.dot-amber { background-color: var(--amber-warning); box-shadow: 0 0 6px var(--amber-warning); }

/* Persistent Prototype Banner */
.prototype-warning-banner {
    background: linear-gradient(135deg, rgba(255, 176, 32, 0.12) 0%, rgba(255, 61, 129, 0.12) 100%);
    border: 1px solid rgba(255, 176, 32, 0.4);
    border-left: 5px solid var(--amber-warning);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 22px;
    backdrop-filter: blur(10px);
}

.prototype-warning-header {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--amber-warning);
    font-weight: 700;
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.prototype-warning-body {
    font-size: 0.86rem;
    color: #CBD5E1;
    line-height: 1.45;
    margin: 0;
}

/* Glass-style Cards */
.glass-panel {
    background: var(--gradient-card);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

/* Verdict Display Cards */
.verdict-banner {
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 18px 0 24px 0;
    backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
}

.verdict-banner::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-glitch);
}

.verdict-banner-fake {
    background: linear-gradient(180deg, rgba(255, 61, 129, 0.16) 0%, rgba(18, 26, 43, 0.95) 100%);
    border: 2px solid var(--magenta-glitch);
}

.verdict-banner-real {
    background: linear-gradient(180deg, rgba(45, 226, 196, 0.16) 0%, rgba(18, 26, 43, 0.95) 100%);
    border: 2px solid var(--teal-scan);
}

.verdict-banner-inconclusive {
    background: linear-gradient(180deg, rgba(255, 176, 32, 0.16) 0%, rgba(18, 26, 43, 0.95) 100%);
    border: 2px solid var(--amber-warning);
}

.verdict-headline {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.verdict-headline-fake { color: var(--magenta-glitch); }
.verdict-headline-real { color: var(--teal-scan); }
.verdict-headline-inconclusive { color: var(--amber-warning); }

.verdict-confidence-text {
    font-size: 0.92rem;
    color: var(--text-muted);
    margin-top: 6px;
    font-style: italic;
}

/* Video Scanning Preview Wrapper */
.video-preview-wrapper {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--panel-border);
    background: #000000;
}

.scanner-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    background: linear-gradient(
        to bottom,
        rgba(34, 211, 238, 0) 0%,
        rgba(34, 211, 238, 0.08) 50%,
        rgba(45, 226, 196, 0.25) 51%,
        rgba(34, 211, 238, 0) 55%
    );
    background-size: 100% 200%;
}

@keyframes scanAnimation {
    0% { background-position: 0% 0%; }
    100% { background-position: 0% 200%; }
}

@media (prefers-reduced-motion: no-preference) {
    .scanner-overlay {
        animation: scanAnimation 4s linear infinite;
    }
}

/* Face Crop Strip */
.face-crop-strip-container {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 10px 4px;
    scrollbar-width: thin;
    scrollbar-color: var(--cyan-accent) var(--panel-dark);
}

.face-crop-thumbnail {
    flex: 0 0 auto;
    width: 80px;
    height: 80px;
    border-radius: 8px;
    border: 1px solid var(--panel-border);
    object-fit: cover;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.face-crop-thumbnail:hover {
    transform: scale(1.08);
    border-color: var(--teal-scan);
}

/* Styled Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(18, 26, 43, 0.6);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid var(--panel-border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(34, 211, 238, 0.12) !important;
    color: var(--cyan-accent) !important;
    font-weight: 700 !important;
    border: 1px solid rgba(34, 211, 238, 0.35) !important;
}

/* Primary Button Glow */
div.stButton > button[kind="primary"] {
    background: var(--gradient-glitch);
    border: none;
    color: #FFFFFF;
    font-weight: 700;
    letter-spacing: 0.03em;
    border-radius: 10px;
    padding: 10px 20px;
    box-shadow: 0 0 15px rgba(255, 61, 129, 0.35);
    transition: all 0.25s ease;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 25px rgba(255, 61, 129, 0.6);
    transform: translateY(-1px);
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: var(--panel-dark);
    border-right: 1px solid var(--panel-border);
}

/* Progress bar cyber accent */
.stProgress > div > div > div > div {
    background: var(--gradient-glitch);
}
</style>
"""

APP_LOGO_SVG = """
<svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="48" height="48" rx="10" fill="#121A2B" stroke="#22D3EE" stroke-width="1.5"/>
    <path d="M24 10L36 15V23C36 30.5 30.88 37.45 24 39C17.12 37.45 12 30.5 12 23V15L24 10Z"
          fill="url(#shield_grad)" stroke="#22D3EE" stroke-width="2" stroke-linejoin="round"/>
    <path d="M20 23L23 26L28 20" stroke="#2DE2C4" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <line x1="16" y1="21" x2="32" y2="21" stroke="#FF3D81" stroke-width="1" stroke-dasharray="2 2" opacity="0.75"/>
    <line x1="16" y1="27" x2="32" y2="27" stroke="#FF3D81" stroke-width="1" stroke-dasharray="2 2" opacity="0.75"/>
    <defs>
        <linearGradient id="shield_grad" x1="12" y1="10" x2="36" y2="39" gradientUnits="userSpaceOnUse">
            <stop stop-color="#121A2B"/>
            <stop offset="1" stop-color="#1E293B"/>
        </linearGradient>
    </defs>
</svg>
"""

PIPELINE_FLOW_HTML = """
<div style="background: rgba(18, 26, 43, 0.85); border: 1px solid rgba(34, 211, 238, 0.2); border-radius: 14px; padding: 24px; margin-bottom: 24px;">
    <h3 style="margin-top: 0; color: #22D3EE; font-size: 1.15rem; display: flex; align-items: center; gap: 8px;">
        <span>🔬</span> End-to-End Spatio-Temporal Pipeline Architecture
    </h3>
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-top: 18px;">
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">🎬</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">1. Video Input</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">3 segments sampled uniformly</div>
        </div>
        <div style="color: #22D3EE; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">👤</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">2. Face Cascade</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">Haar frontal detection + fallback</div>
        </div>
        <div style="color: #22D3EE; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">✂️</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">3. Spatial Crop</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">128×128 with 25% margin</div>
        </div>
        <div style="color: #22D3EE; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">🧩</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">4. CNN Feature</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">4-block TimeDistributed CNN</div>
        </div>
        <div style="color: #22D3EE; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">⏱️</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">5. LSTM Modeling</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">128-unit temporal recurrent layer</div>
        </div>
        <div style="color: #22D3EE; font-weight: bold; font-size: 1.2rem;">➔</div>
        <div style="flex: 1; min-width: 140px; background: rgba(10, 14, 26, 0.9); border: 1px solid #334155; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 1.4rem;">📈</div>
            <div style="font-weight: 700; color: #E6EAF2; margin: 6px 0 2px 0; font-size: 0.88rem;">6. P(manipulation)</div>
            <div style="font-size: 0.75rem; color: #8B97B1;">Mean clip score + band filter</div>
        </div>
    </div>
</div>
"""
