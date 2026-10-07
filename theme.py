"""Centralized DeepTrace Theme Palette - V3 Forensic Intelligence Workstation.

This module defines the color tokens for DeepTrace:
- Primary background: Deep Graphite / Near Black (#070B12)
- Secondary background / Surface: Dark Slate (#0D131D)
- Elevated Surface: Elevated Navy/Slate (#111A26)
- Borders & Dividers: Cool Steel Blue (#202C3A)
- Text Primary: Crisp Off-White (#F8FAFC)
- Text Muted: Cool Muted Slate (#94A3B8)
- Primary Accent: Electric Cyan (#22D3EE)
- Secondary Accent: Electric Blue (#3B82F6)
- Authentic / REAL: Emerald Green (#22C55E)
- Manipulated / FAKE: Coral Red (#EF4444)
- Inconclusive: Amber (#F59E0B)
"""

from typing import Dict

THEME: Dict[str, str] = {
    # Surfaces & Structural Backgrounds (Forensic Workstation Aesthetic)
    "bg_dark": "#070B12",         # Deep Graphite / Near Black primary background
    "bg_secondary": "#0D131D",    # Dark Slate secondary surface
    "panel_dark": "#111A26",      # Elevated dark slate cards and panels
    "panel_border": "#202C3A",    # Cool steel blue borders and dividers
    "text_primary": "#F8FAFC",    # Crisp off-white primary text
    "text_muted": "#94A3B8",      # Cool muted slate secondary text

    # Primary Accents & Neutral Controls
    "primary_accent": "#22D3EE",  # Electric Cyan accent
    "accent_electric": "#22D3EE", # Electric Cyan
    "accent_blue": "#3B82F6",     # Electric Blue
    "accent_violet": "#3B82F6",   # Supporting blue accent

    # Verdict Classifications (Semantic State Colors)
    "authentic": "#22C55E",       # Emerald green (Authentic / Real)
    "manipulated": "#EF4444",     # Coral red (Manipulated / Fake)
    "inconclusive": "#F59E0B",    # Amber (Inconclusive / Ambiguous)

    # Gradients & Highlights (Subtle & Professional)
    "gradient_start": "#22D3EE",  # Cyan
    "gradient_mid": "#3B82F6",    # Blue
    "gradient_end": "#1D4ED8",    # Deep Blue
    "gradient_cta": "linear-gradient(135deg, #22D3EE 0%, #3B82F6 100%)",
}
