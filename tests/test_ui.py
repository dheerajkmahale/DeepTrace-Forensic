"""Unit tests for Streamlit UI helpers, verdict calibration, report builders, and AppTest.

Tests:
1. Verdict calculation logic (below cutoff, above cutoff, inconclusive interval, and boundary edges).
2. JSON and HTML forensic report builders (serialization, keys, honest disclosures, and provenance).
3. Streamlit AppTest integration:
   - App loads without exceptions.
   - All 6 forensic tabs render.
   - Real experiment is clearly disabled/marked unavailable when best_model.keras is missing.
   - Prototype banner is prominently rendered for the prototype model.
"""

import json
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest
from ui_helpers import (
    build_html_report,
    build_json_report,
    compute_verdict,
    get_git_commit,
    html_block,
)


class TestHtmlBlockHelper:
    """Unit tests for html_block() helper ensuring safe CommonMark rendering."""

    def test_strips_leading_spaces_and_tabs(self):
        """Ensure no line starts with 4+ spaces or tabs, preventing code block conversion."""
        indented_html = "    <div class=\"test\">\n        <p>Text</p>\n    </div>"
        cleaned = html_block(indented_html)
        for line in cleaned.splitlines():
            assert not line.startswith("    "), f"Line starts with 4+ spaces: '{line}'"
            assert not line.startswith("\t"), f"Line starts with tab: '{line}'"
            assert line == line.strip(), f"Line has leading/trailing whitespace: '{line}'"

    def test_removes_all_blank_lines(self):
        """Ensure all empty lines and whitespace-only lines are removed."""
        html_with_blanks = "<div>\n\n   \n\t\n<p>Hello</p>\n\n</div>"
        cleaned = html_block(html_with_blanks)
        lines = cleaned.splitlines()
        assert len(lines) == 3
        for line in lines:
            assert len(line.strip()) > 0, "Found blank line in output"

    def test_handles_empty_or_none(self):
        """Ensure empty strings or None produce an empty string without crashing."""
        assert html_block("") == ""
        assert html_block("   \n\n  ") == ""
        assert html_block(None) == ""

    def test_preserves_html_tags_and_content(self):
        """Ensure tag structures and contents remain intact."""
        raw = """
        <div class="glitch-title">
            Deepfake Detector
        </div>
        """
        cleaned = html_block(raw)
        assert '<div class="glitch-title">' in cleaned
        assert "Deepfake Detector" in cleaned
        assert "</div>" in cleaned


class TestVerdictLogic:
    """Unit tests for compute_verdict under various probability, threshold, and band settings."""

    def test_verdict_authentic_below_threshold(self):
        """Test probability well below threshold and outside inconclusive band."""
        res = compute_verdict(p_fake=0.10, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res["verdict"] == "REAL"
        assert res["label"] == "AUTHENTIC"
        assert res["icon"] == "✓"
        assert res["status"] == "real"
        assert res["color"] == "#C4A1FF"
        assert "authentic" in res["confidence"].lower()
        assert res["p_fake"] == pytest.approx(0.10)
        assert res["p_real"] == pytest.approx(0.90)

    def test_verdict_manipulated_above_threshold(self):
        """Test probability well above threshold and outside inconclusive band."""
        res = compute_verdict(p_fake=0.92, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res["verdict"] == "FAKE"
        assert res["label"] == "MANIPULATED"
        assert res["icon"] == "!"
        assert res["status"] == "fake"
        assert res["color"] == "#FF5A36"
        assert "manipulation" in res["confidence"].lower()
        assert res["p_fake"] == pytest.approx(0.92)

    def test_verdict_inconclusive_within_band(self):
        """Test probability inside the inconclusive band (0.40 - 0.60)."""
        # Exactly in the middle
        res_mid = compute_verdict(p_fake=0.50, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_mid["verdict"] == "INCONCLUSIVE"
        assert res_mid["label"] == "INCONCLUSIVE"
        assert res_mid["icon"] == "~"
        assert res_mid["status"] == "inconclusive"
        assert res_mid["color"] == "#FFC247"

        # On the lower boundary
        res_low = compute_verdict(p_fake=0.40, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_low["verdict"] == "INCONCLUSIVE"

        # On the upper boundary
        res_high = compute_verdict(p_fake=0.60, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_high["verdict"] == "INCONCLUSIVE"

    def test_verdict_edge_values_zero_and_one(self):
        """Test edge values 0.0 and 1.0."""
        res_zero = compute_verdict(p_fake=0.0, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_zero["verdict"] == "REAL"
        assert res_zero["p_fake"] == 0.0
        assert res_zero["p_real"] == 1.0

        res_one = compute_verdict(p_fake=1.0, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_one["verdict"] == "FAKE"
        assert res_one["p_fake"] == 1.0
        assert res_one["p_real"] == 0.0

    def test_verdict_clamping_out_of_bounds(self):
        """Test that values < 0 or > 1 are safely clamped to [0.0, 1.0]."""
        res_neg = compute_verdict(p_fake=-0.25, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_neg["p_fake"] == 0.0
        assert res_neg["verdict"] == "REAL"

        res_over = compute_verdict(p_fake=1.75, threshold=0.50, inconclusive_band=(0.40, 0.60))
        assert res_over["p_fake"] == 1.0
        assert res_over["verdict"] == "FAKE"

    def test_verdict_unsorted_band_handles_gracefully(self):
        """Test that (0.60, 0.40) is automatically sorted to (0.40, 0.60)."""
        res = compute_verdict(p_fake=0.48, threshold=0.50, inconclusive_band=(0.60, 0.40))
        assert res["verdict"] == "INCONCLUSIVE"
        assert res["inconclusive_band"] == (0.40, 0.60)


class TestReportBuilders:
    """Unit tests for JSON and HTML forensic report generation."""

    @pytest.fixture
    def sample_analysis_data(self):
        v_info = compute_verdict(p_fake=0.88, threshold=0.50, inconclusive_band=(0.40, 0.60))
        return {
            "generated_at": "2026-10-03T10:00:00Z",
            "git_commit": "abcdef1",
            "video_filename": "test_sample.mp4",
            "model_used": "outputs/demo/best_model.keras",
            "model_provenance": "prototype_synthetic",
            "p_fake": 0.88,
            "p_real": 0.12,
            "verdict": v_info["verdict"],
            "label": v_info["label"],
            "confidence": v_info["confidence"],
            "threshold": 0.50,
            "inconclusive_band": (0.40, 0.60),
            "clips_analyzed": 3,
            "clip_probabilities": [0.85, 0.90, 0.89],
            "face_stats": {
                "total_frames": 90,
                "frames_inspected": 30,
                "detected_faces_count": 30,
                "face_detection_rate": 1.0,
                "fallback_used": False,
                "sample_crops": [__import__("numpy").zeros((128, 128, 3), dtype="uint8")],
            },
            "verdict_info": v_info,
        }

    def test_json_report_structure_and_serialization(self, sample_analysis_data):
        """Test that build_json_report returns valid JSON with all required keys and no binary leaks."""
        json_str = build_json_report(sample_analysis_data)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["report_version"] == "1.0"
        assert parsed["video_filename"] == "test_sample.mp4"
        assert parsed["model_used"] == "outputs/demo/best_model.keras"
        assert parsed["model_provenance"] == "prototype_synthetic"
        assert parsed["manipulation_probability"] == 0.88
        assert parsed["authentic_probability"] == 0.12
        assert parsed["verdict"] == "FAKE"
        assert parsed["verdict_label"] == "MANIPULATED"
        assert parsed["clips_analyzed"] == 3
        assert len(parsed["clip_probabilities"]) == 3
        assert "disclaimer" in parsed["legal_and_technical_disclaimer"].lower() or "prototype" in parsed["legal_and_technical_disclaimer"].lower()

    def test_html_report_structure(self, sample_analysis_data):
        """Test that build_html_report returns a valid HTML document containing theme styling and prototype banner."""
        html_str = build_html_report(sample_analysis_data)
        assert isinstance(html_str, str)
        assert "<!DOCTYPE html>" in html_str
        assert "Forensic Video Analysis Report" in html_str
        assert "PROTOTYPE DEMONSTRATION MODEL" in html_str
        assert "MANIPULATED" in html_str
        assert "88.0%" in html_str
        assert "test_sample.mp4" in html_str
        assert "Forensic Limitations & Technical Notice" in html_str


class TestStreamlitAppTest:
    """Streamlit AppTest suite verifying app initialization, tabs, and honesty controls."""

    def test_app_loads_without_exceptions(self):
        """Test that app.py compiles and runs without throwing any uncaught exceptions."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        assert len(at.exception) == 0, f"App execution produced exceptions: {at.exception}"

    def test_app_renders_all_six_tabs(self):
        """Test that all 6 specified tabs are rendered properly."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        assert len(at.tabs) == 6
        expected_labels = ["Analyze", "Batch", "History", "Model & Results", "How It Works", "About & Limitations"]
        for expected in expected_labels:
            assert any(expected in tab.label for tab in at.tabs), f"Tab '{expected}' not found in rendered tabs"

    def test_real_experiment_disabled_when_model_missing(self):
        """Test that Real Experiment Model is marked unavailable in the sidebar when outputs/best_model.keras is missing."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()

        # The model selector selectbox is the first selectbox in the sidebar
        assert len(at.selectbox) >= 1
        model_selectbox = at.selectbox[0]
        options = model_selectbox.options

        # Since outputs/best_model.keras does not exist, it must be marked unavailable
        assert not os.path.exists("outputs/best_model.keras")
        unavailable_option = [opt for opt in options if "Unavailable" in opt or "missing" in opt]
        assert len(unavailable_option) > 0, f"Expected unavailable option in selectbox, found: {options}"

    def test_prototype_banner_appears_for_prototype_model(self):
        """Test that the persistent prototype warning banner is present in rendered markdown."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()

        banner_found = False
        for md in at.markdown:
            if "Prototype model trained on synthetic data. Results on real face videos are NOT meaningful." in md.value:
                banner_found = True
                break

        assert banner_found, "Persistent prototype warning banner was not found in rendered markdown!"


class TestForbiddenColors:
    """Test suite ensuring strict compliance with the Ultraviolet Forensics palette.

    HARD CONSTRAINT: Strictly zero black, blue, or green anywhere in the UI.
    """

    def test_no_forbidden_color_families_in_ui_assets(self):
        """Parse all hex colors from ui_styles.py, ui_helpers.py, and config.toml,

        convert to HSL, and fail if any is pure/near black (lightness < 8%) or
        falls in a blue or green hue range (allowing near-white neutral text).
        """
        import colorsys
        import re
        from pathlib import Path

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
            Path(PROJECT_ROOT) / ".streamlit" / "config.toml",
        ]

        hex_pattern = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
        discovered_hexes = set()

        for f_path in target_files:
            assert f_path.exists(), f"Target UI file missing: {f_path}"
            content = f_path.read_text(encoding="utf-8")
            matches = hex_pattern.findall(content)
            discovered_hexes.update(matches)

        assert len(discovered_hexes) > 0, "No hex colors discovered in UI files"

        for hex_code in sorted(discovered_hexes):
            clean_hex = hex_code.lstrip("#")
            if len(clean_hex) == 3:
                clean_hex = "".join([c * 2 for c in clean_hex])

            r = int(clean_hex[0:2], 16) / 255.0
            g = int(clean_hex[2:4], 16) / 255.0
            b = int(clean_hex[4:6], 16) / 255.0

            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hue_deg = h * 360.0
            lightness_pct = l * 100.0
            saturation_pct = s * 100.0

            # 1. Pure or near black check: lightness < 8%
            assert lightness_pct >= 8.0, (
                f"Forbidden near-black color detected: {hex_code} "
                f"(Lightness: {lightness_pct:.1f}% < 8%)"
            )

            # Allow near-white neutral text (lightness >= 88% or saturation < 10%)
            is_neutral_light = lightness_pct >= 88.0 or saturation_pct < 10.0

            if not is_neutral_light:
                # 2. Green hue range: 65 deg to 165 deg
                assert not (65.0 <= hue_deg <= 165.0), (
                    f"Forbidden green hue detected: {hex_code} "
                    f"(Hue: {hue_deg:.1f}deg, Sat: {saturation_pct:.1f}%, Light: {lightness_pct:.1f}%)"
                )

                # 3. Blue hue range: 170 deg to 255 deg
                assert not (170.0 <= hue_deg <= 255.0), (
                    f"Forbidden blue hue detected: {hex_code} "
                    f"(Hue: {hue_deg:.1f}deg, Sat: {saturation_pct:.1f}%, Light: {lightness_pct:.1f}%)"
                )

    def test_no_forbidden_named_colors_in_ui_assets(self):
        """Scan UI assets for forbidden named CSS colors (black, blue, green, cyan, teal, etc.).

        Ensures no CSS style properties or markup attributes use forbidden named color keywords.
        Comments and docstrings are excluded from matching.
        """
        import re
        from pathlib import Path

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
            Path(PROJECT_ROOT) / ".streamlit" / "config.toml",
        ]

        forbidden_names = [
            "black",
            "blue",
            "green",
            "cyan",
            "teal",
            "navy",
            "lime",
            "aqua",
            "darkblue",
            "mediumblue",
            "seagreen",
            "darkgreen",
            "forestgreen",
            "midnightblue",
        ]

        # Pattern targeting CSS property or HTML attribute color assignments
        # e.g., color: blue; fill="black"; stroke="cyan"; border: 1px solid green;
        css_attr_pattern = re.compile(
            r"""(?ix)
            (?:color|background|background-color|border|border-color|fill|stroke|outline)\s*[:=]\s*['"]?[^'";\n>]*?\b("""
            + "|".join(forbidden_names)
            + r""")\b"""
        )

        violations = []
        for f_path in target_files:
            assert f_path.exists(), f"Target UI file missing: {f_path}"
            content = f_path.read_text(encoding="utf-8")
            # Strip comments to prevent false positives in descriptive explanations
            cleaned_lines = []
            in_multiline_docstring = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                        in_multiline_docstring = not in_multiline_docstring
                        continue
                if in_multiline_docstring:
                    continue
                # Remove single-line comment
                code_only = line.split("#")[0]
                cleaned_lines.append(code_only)

            code_text = "\n".join(cleaned_lines)
            for m in css_attr_pattern.finditer(code_text):
                matched_color = m.group(1).lower()
                violations.append(f"{f_path.name}: matched '{matched_color}' in '{m.group(0)}'")

        assert len(violations) == 0, f"Found forbidden named CSS colors in UI assets:\n" + "\n".join(violations)

    def test_no_forbidden_rgb_or_hsl_values_in_ui_assets(self):
        """Scan UI assets for rgb()/rgba() and hsl()/hsla() definitions and verify hue and lightness."""
        import colorsys
        import re
        from pathlib import Path

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
            Path(PROJECT_ROOT) / ".streamlit" / "config.toml",
        ]

        rgb_pattern = re.compile(r"rgba?\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)")
        hsl_pattern = re.compile(r"hsla?\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)%?\s*,\s*([0-9.]+)%?(?:\s*,\s*([0-9.]+))?\s*\)")

        for f_path in target_files:
            content = f_path.read_text(encoding="utf-8")
            for m in rgb_pattern.finditer(content):
                r = int(m.group(1)) / 255.0
                g = int(m.group(2)) / 255.0
                b = int(m.group(3)) / 255.0
                alpha = float(m.group(4)) if m.group(4) else 1.0

                if alpha == 0:
                    continue  # fully transparent

                h, l, s = colorsys.rgb_to_hls(r, g, b)
                hue_deg = h * 360.0
                lightness_pct = l * 100.0
                saturation_pct = s * 100.0

                # Check near-black
                assert lightness_pct >= 8.0, (
                    f"Forbidden near-black rgb found in {f_path.name}: {m.group(0)} (Lightness {lightness_pct:.1f}%)"
                )

                # Skip neutral/near-white
                if lightness_pct < 88.0 and saturation_pct >= 10.0:
                    assert not (65.0 <= hue_deg <= 165.0), (
                        f"Forbidden green rgb found in {f_path.name}: {m.group(0)} (Hue {hue_deg:.1f}deg)"
                    )
                    assert not (170.0 <= hue_deg <= 255.0), (
                        f"Forbidden blue rgb found in {f_path.name}: {m.group(0)} (Hue {hue_deg:.1f}deg)"
                    )

            for m in hsl_pattern.finditer(content):
                hue_deg = float(m.group(1))
                saturation_pct = float(m.group(2))
                lightness_pct = float(m.group(3))
                alpha = float(m.group(4)) if m.group(4) else 1.0

                if alpha == 0:
                    continue

                assert lightness_pct >= 8.0, (
                    f"Forbidden near-black hsl found in {f_path.name}: {m.group(0)} (Lightness {lightness_pct:.1f}%)"
                )
                if lightness_pct < 88.0 and saturation_pct >= 10.0:
                    assert not (65.0 <= hue_deg <= 165.0), (
                        f"Forbidden green hsl found in {f_path.name}: {m.group(0)} (Hue {hue_deg:.1f}deg)"
                    )
                    assert not (170.0 <= hue_deg <= 255.0), (
                        f"Forbidden blue hsl found in {f_path.name}: {m.group(0)} (Hue {hue_deg:.1f}deg)"
                    )
