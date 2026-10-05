"""Programmatic WCAG 2.1 Contrast Audit across all UI components and modules.

Audits:
- theme.py tokens
- ui_styles.py CSS rules and inline components
- ui_helpers.py callouts, charts, tables, and HTML report generator
- app.py headers, status chips, sidebars, verdicts, and tables
"""

import json

def srgb_to_lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def relative_luminance(hex_code):
    h = hex_code.lstrip('#')
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return 0.2126 * srgb_to_lin(r) + 0.7152 * srgb_to_lin(g) + 0.0722 * srgb_to_lin(b)

def compute_contrast_ratio(c1, c2):
    l1 = relative_luminance(c1)
    l2 = relative_luminance(c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

# UI Color Token Definitions
THEME = {
    "bg_dark": "#2A1248",
    "panel_dark": "#3A1A63",
    "panel_border": "#5B2E91",
    "text_primary": "#F6EEFF",
    "text_muted": "#B9A4D6",
    "primary_accent": "#8B5CF6",
    "authentic": "#C4A1FF",
    "manipulated": "#FF643D",
    "inconclusive": "#FFC247",
    "white": "#FFFFFF",
}

PAIRS = [
    # General Page & Panel Typography
    ("Primary Body Text (app.py, ui_styles.py)", THEME["text_primary"], THEME["bg_dark"], "Normal", ">= 14px regular", 4.5),
    ("Primary Panel Text (cards, forms, metrics)", THEME["text_primary"], THEME["panel_dark"], "Normal", ">= 14px regular", 4.5),
    ("Muted Text on Page (subtitles, disclaimers)", THEME["text_muted"], THEME["bg_dark"], "Normal", ">= 14px regular", 4.5),
    ("Muted Text on Panel (meta labels, hints)", THEME["text_muted"], THEME["panel_dark"], "Normal", ">= 14px regular", 4.5),

    # Navigation & Controls
    ("Active Tab Label (.stTabs selected)", THEME["text_primary"], THEME["panel_dark"], "Large/Bold", "15.2px (0.95rem) Bold", 3.0),
    ("Inactive Tab Label (.stTabs regular)", THEME["text_muted"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Primary Button Text (Run Analysis, Process)", THEME["white"], THEME["primary_accent"], "Large/Bold", "18.88px (1.18rem) Bold", 3.0),
    ("Secondary Button Text (Clear, Download)", THEME["text_primary"], THEME["panel_dark"], "Normal", "14.5px regular", 4.5),
    ("Custom Player Play Button (#forensic-play-btn)", THEME["white"], THEME["primary_accent"], "Large/Bold", "15.2px (0.95rem) Bold", 3.0),
    ("Custom Player Time Display (#forensic-time)", THEME["text_primary"], THEME["bg_dark"], "Normal", "14px Monospace Bold", 4.5),

    # Verdict Classifications (Strictly reserved)
    ("Authentic Verdict Text (chips, cards, tables)", THEME["authentic"], THEME["panel_dark"], "Normal", ">= 14px regular/bold", 4.5),
    ("Authentic Verdict Headline (banner)", THEME["authentic"], THEME["panel_dark"], "Large/Bold", "35.2px (2.2rem) Bold", 3.0),
    ("Manipulated Verdict Text (chips, cards, tables)", THEME["manipulated"], THEME["panel_dark"], "Normal", ">= 14px regular/bold", 4.5),
    ("Manipulated Verdict Headline (banner)", THEME["manipulated"], THEME["panel_dark"], "Large/Bold", "35.2px (2.2rem) Bold", 3.0),
    ("Inconclusive Verdict Text (chips, cards, tables)", THEME["inconclusive"], THEME["panel_dark"], "Normal", ">= 14px regular/bold", 4.5),
    ("Inconclusive Verdict Headline (banner)", THEME["inconclusive"], THEME["panel_dark"], "Large/Bold", "35.2px (2.2rem) Bold", 3.0),

    # Sidebar Controls & Notifications
    ("Sidebar Header Text (Configuration, Threshold)", THEME["text_primary"], THEME["panel_dark"], "Large/Bold", "18px Bold", 3.0),
    ("Sidebar Label Text (Model select, Slider)", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Sidebar Caption Text (detection modes)", THEME["text_muted"], THEME["panel_dark"], "Normal", "13px regular", 4.5),
    ("Toast Notification Text (stToast)", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),

    # Callouts (ui_helpers.py)
    ("Callout Info Text", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Callout Warning/Notice Text", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Callout Error Text", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Callout Verified/Success Text", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),

    # Plotly Visualizations (ui_helpers.py)
    ("Plotly Gauge Value Text", THEME["text_primary"], THEME["panel_dark"], "Large/Bold", "36px Bold", 3.0),
    ("Plotly Gauge Title & Subtitle", THEME["text_muted"], THEME["panel_dark"], "Normal", "13px regular", 4.5),
    ("Plotly Gauge Tick Labels", THEME["text_muted"], THEME["panel_dark"], "Normal", "10px regular", 4.5),
    ("Plotly Clip Bar Chart Title", THEME["text_primary"], THEME["panel_dark"], "Large/Bold", "13px Bold", 3.0),
    ("Plotly Clip Bar Value Annotation (outside)", THEME["text_primary"], THEME["bg_dark"], "Normal", "11px Regular", 4.5),
    ("Plotly Clip Axis Ticks (Y)", THEME["text_muted"], THEME["panel_dark"], "Normal", "10px regular", 4.5),
    ("Plotly Clip Axis Labels (X)", THEME["text_primary"], THEME["bg_dark"], "Normal", "11px regular", 4.5),

    # Code Blocks & Diagnostics
    ("Code Block Text (.stCode, git hash, pre)", THEME["authentic"], THEME["panel_dark"], "Normal", "13px Monospace", 4.5),

    # HTML Forensic Report (ui_helpers.py build_html_report)
    ("Report Title (h1)", THEME["text_primary"], THEME["panel_dark"], "Large/Bold", "28.8px (1.8rem) Bold", 3.0),
    ("Report Subtitle", THEME["text_muted"], THEME["panel_dark"], "Normal", "15.2px regular", 4.5),
    ("Report Verdict Title", THEME["manipulated"], THEME["panel_dark"], "Large/Bold", "32px (2rem) Bold", 3.0),
    ("Report Meta Item Label", THEME["text_muted"], THEME["bg_dark"], "Normal", "12.8px regular", 4.5),
    ("Report Meta Item Value", THEME["text_primary"], THEME["bg_dark"], "Normal", "14.4px Bold", 4.5),
    ("Report Table Header Text", THEME["text_muted"], THEME["bg_dark"], "Normal", "12.8px regular", 4.5),
    ("Report Table Row Text", THEME["text_primary"], THEME["panel_dark"], "Normal", "14px regular", 4.5),
    ("Report Table Verdict (Manipulated)", THEME["manipulated"], THEME["panel_dark"], "Normal", "14px Bold", 4.5),
    ("Report Table Verdict (Authentic)", THEME["authentic"], THEME["panel_dark"], "Normal", "14px Bold", 4.5),
    ("Report Disclaimer Text", THEME["text_muted"], THEME["panel_dark"], "Normal", "12.8px regular", 4.5),
]

def run_audit():
    results = []
    all_passed = True
    print(f"{'UI Component / Location':<45} | {'Text':<8} | {'Background':<8} | {'Class':<10} | {'Ratio':<8} | {'Req':<6} | {'Status'}")
    print("-" * 105)

    for desc, text_col, bg_col, classification, font_info, required_ratio in PAIRS:
        ratio = compute_contrast_ratio(text_col, bg_col)
        passed = ratio >= required_ratio
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        results.append({
            "component": desc,
            "text_color": text_col,
            "bg_color": bg_col,
            "classification": classification,
            "font_info": font_info,
            "ratio": round(ratio, 2),
            "required": required_ratio,
            "status": status,
        })
        print(f"{desc:<45} | {text_col:<8} | {bg_col:<8} | {classification:<10} | {ratio:>5.2f}:1 | >={required_ratio:.1f}:1 | {status}")

    print("-" * 105)
    print(f"Total pairs audited: {len(results)}. All passed: {all_passed}")

    with open("outputs/contrast_audit_report.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": results, "all_passed": all_passed}, f, indent=2)

    return results, all_passed

if __name__ == "__main__":
    run_audit()
