"""Centralized DeepTrace Theme Palette based on SaaS Reference Design.

This module defines the color tokens for DeepTrace:
- Primary background: Deep navy / near-black (#090D16)
- Secondary background: Slightly lighter navy (#0E1526)
- Panels & Cards: Dark blue-gray (#131B2E)
- Borders & Dividers: Subtle cool gray/blue (#243048)
- Text Primary: Crisp white (#FFFFFF)
- Text Muted: Muted blue-gray (#94A3B8)
- Primary Accent: Electric cyan / blue (#06B6D4)
- Secondary Accent: Electric sky blue (#38BDF8)
- Violet Accent: Sleek violet (#8B5CF6)
- Authentic / REAL: Emerald green (#10B981)
- Manipulated / FAKE: Coral red (#EF4444)
- Inconclusive: Amber (#F59E0B)
"""

from typing import Dict

THEME: Dict[str, str] = {
    # Surfaces & Structural Backgrounds (Dark Navy SaaS Aesthetic)
    "bg_dark": "#090D16",         # Deep navy / near-black primary background
    "bg_secondary": "#0E1526",    # Slightly lighter navy secondary background
    "panel_dark": "#131B2E",      # Dark blue-gray cards and surfaces
    "panel_border": "#243048",    # Subtle cool gray/blue borders and dividers
    "text_primary": "#FFFFFF",    # Crisp white primary text
    "text_muted": "#94A3B8",      # Muted blue-gray secondary text

    # Primary Accents & Neutral Controls
    "primary_accent": "#06B6D4",  # Electric cyan / blue accent
    "accent_electric": "#38BDF8", # Electric sky blue
    "accent_violet": "#8B5CF6",   # Violet accent

    # Verdict Classifications (Semantic State Colors)
    "authentic": "#10B981",       # Emerald green (Authentic / Real)
    "manipulated": "#EF4444",     # Coral red (Manipulated / Fake)
    "inconclusive": "#F59E0B",    # Amber (Inconclusive / Ambiguous)

    # Gradients & Highlights
    "gradient_start": "#06B6D4",  # Cyan
    "gradient_mid": "#3B82F6",    # Blue
    "gradient_end": "#8B5CF6",    # Violet
    "gradient_cta": "linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #8B5CF6 100%)",
}

