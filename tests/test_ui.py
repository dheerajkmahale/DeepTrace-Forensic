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
)


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
        assert res["icon"] == "⚠️"
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
        assert res_mid["icon"] == "⚡"
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
        expected_labels = ["🔍 Analyze", "📁 Batch", "📜 History", "📊 Model & Results", "⚙️ How It Works", "ℹ️ About & Limitations"]
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
