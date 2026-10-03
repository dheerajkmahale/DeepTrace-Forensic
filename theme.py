"""Centralized Ultraviolet Forensics Theme Palette.

This single module defines the definitive color tokens for the application.
Theme modifications should be made here to cascade throughout the UI.

HARD CONSTRAINT: No black, blue, or green anywhere in the UI.
"""

from typing import Dict

THEME: Dict[str, str] = {
    # Surfaces & Structural Backgrounds (no black, blue, or green)
    "bg_dark": "#2A1248",        # Deep plum background
    "panel_dark": "#3A1A63",     # Rich plum panels & cards
    "panel_border": "#5B2E91",   # Medium ultraviolet borders & dividers
    "text_primary": "#F6EEFF",   # Crisp lilac-white primary text
    "text_muted": "#B9A4D6",     # Muted lavender-violet text

    # Primary Accent & Neutral Controls
    "primary_accent": "#8B5CF6", # Ultraviolet for buttons, tabs, sliders, focus rings

    # Verdict Classifications (Strictly reserved for verdicts only)
    "authentic": "#C4A1FF",      # Lilac (Authentic / Real)
    "manipulated": "#FF5A36",    # Vermilion (Manipulated / Fake)
    "inconclusive": "#FFC247",   # Gold (Inconclusive / Ambiguous)

    # Gradients & Highlights
    "gradient_start": "#8B5CF6", # Ultraviolet
    "gradient_end": "#FF5A36",   # Vermilion
}
