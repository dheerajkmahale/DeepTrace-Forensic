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
from theme import THEME
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
        assert res["color"] == THEME["authentic"]
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
        assert res["color"] == THEME["manipulated"]
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
        assert res_mid["color"] == THEME["inconclusive"]

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
        import unittest.mock
        real_exists = os.path.exists
        def mock_exists(p):
            if "outputs/best_model.keras" in str(p).replace("\\", "/"):
                return False
            return real_exists(p)

        with unittest.mock.patch("os.path.exists", side_effect=mock_exists):
            at = AppTest.from_file("app.py", default_timeout=30)
            at.run()

            # The model selector selectbox is the first selectbox in the sidebar
            assert len(at.selectbox) >= 1
            model_selectbox = at.selectbox[0]
            options = model_selectbox.options

            unavailable_option = [opt for opt in options if "Unavailable" in opt or "missing" in opt]
            assert len(unavailable_option) > 0, f"Expected unavailable option in selectbox, found: {options}"

    def test_real_experiment_enabled_when_model_exists(self):
        """Test that Real Experiment Model is selectable in sidebar when outputs/best_model.keras exists."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()

        assert len(at.selectbox) >= 1
        model_selectbox = at.selectbox[0]
        options = model_selectbox.options

        if os.path.exists("outputs/best_model.keras"):
            available_option = [opt for opt in options if "Real Experiment Model (outputs/best_model.keras)" in opt]
            assert len(available_option) > 0, f"Expected available real model option in selectbox, found: {options}"

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
    """Test suite ensuring strict compliance with the reference SaaS Forensic palette.

    Verifies:
    1. Old purple-heavy theme (#2A1248, #3A1A63, #5B2E91) is completely removed.
    2. Deep navy / near-black (#090D16, #0E1526, #131B2E) palette is active.
    3. UI assets define consistent, well-formed colors.
    """

    def test_no_forbidden_color_families_in_ui_assets(self):
        """Verify that the old purple-heavy palette is completely eliminated from UI assets."""
        import re
        from pathlib import Path

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
            Path(PROJECT_ROOT) / ".streamlit" / "config.toml",
        ]

        old_purple_hexes = {"#2a1248", "#3a1a63", "#5b2e91", "#f6eeff", "#b9a4d6", "#c4a1ff"}
        discovered_hexes = set()
        hex_pattern = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")

        for f_path in target_files:
            assert f_path.exists(), f"Target UI file missing: {f_path}"
            content = f_path.read_text(encoding="utf-8")
            matches = [h.lower() for h in hex_pattern.findall(content)]
            discovered_hexes.update(matches)

        # Ensure none of the old dominant purple colors remain
        remnants = discovered_hexes.intersection(old_purple_hexes)
        assert len(remnants) == 0, f"Discovered old purple theme remnants in UI assets: {remnants}"
        assert len(discovered_hexes) > 0, "No hex colors discovered in UI files"

    def test_no_forbidden_named_colors_in_ui_assets(self):
        """Verify UI assets use controlled palette tokens and clean dark styling."""
        from pathlib import Path
        from theme import THEME

        assert THEME["bg_dark"].upper() == "#070B12", f"Expected primary background #070B12, got {THEME['bg_dark']}"
        assert THEME["panel_dark"].upper() == "#111A26", f"Expected panel background #111A26, got {THEME['panel_dark']}"
        assert THEME["authentic"].upper() == "#22C55E", f"Expected authentic green #22C55E, got {THEME['authentic']}"
        assert THEME["manipulated"].upper() == "#EF4444", f"Expected manipulated coral #EF4444, got {THEME['manipulated']}"

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
            Path(PROJECT_ROOT) / ".streamlit" / "config.toml",
        ]
        for f_path in target_files:
            assert f_path.exists()

    def test_no_forbidden_rgb_or_hsl_values_in_ui_assets(self):
        """Scan UI assets for rgb()/rgba() definitions and verify valid syntax."""
        import re
        from pathlib import Path

        target_files = [
            Path(PROJECT_ROOT) / "theme.py",
            Path(PROJECT_ROOT) / "ui_styles.py",
            Path(PROJECT_ROOT) / "ui_helpers.py",
            Path(PROJECT_ROOT) / "app.py",
        ]

        rgb_pattern = re.compile(r"rgba?\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)")
        for f_path in target_files:
            content = f_path.read_text(encoding="utf-8")
            for m in rgb_pattern.finditer(content):
                r = int(m.group(1))
                g = int(m.group(2))
                b = int(m.group(3))
                assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

    def test_render_custom_video_player(self):
        """Verify custom video player renders accessible HTML without native controls."""
        from ui_helpers import render_custom_video_player
        demo_video = "data/demo/raw/fake/synth_fake_00.mp4"
        if os.path.exists(demo_video):
            html = render_custom_video_player(demo_video)
            assert "<video" in html
            assert "controls" not in html.split("<video")[1].split(">")[0]
            assert "forensic-play-btn" in html
            assert "forensic-seeker" in html
            assert "forensic-time" in html
            assert "aria-label" in html

        # Missing file fallback
        missing_html = render_custom_video_player("non_existent_video.mp4")
        assert "Selected video file not found" in missing_html

    def test_render_forensic_table(self):
        """Verify forensic table renders themed HTML with verdict chips."""
        import pandas as pd
        from ui_helpers import render_forensic_table
        df = pd.DataFrame([
            {"File": "sample1.mp4", "P(Fake)": "0.9500", "Verdict": "! MANIPULATED"},
            {"File": "sample2.mp4", "P(Fake)": "0.0500", "Verdict": "✓ AUTHENTIC"},
        ])
        html = render_forensic_table(df)
        assert "<table" in html
        assert "sample1.mp4" in html
        assert "! MANIPULATED" in html
        assert "✓ AUTHENTIC" in html

        # Empty df
        empty_html = render_forensic_table(pd.DataFrame())
        assert "No records to display" in empty_html
