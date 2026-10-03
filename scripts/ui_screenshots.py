"""Visual verification script for Streamlit Deepfake Detector app.

Uses Playwright Chromium to:
1. Connect to http://localhost:8501.
2. Capture full-page screenshots at 1280x800 (desktop) and 390x844 (mobile) for:
   - Tab 1: Analyze
   - Tab 2: Batch
   - Tab 3: History
   - Tab 4: Model & Results
   - Tab 5: How It Works
   - Tab 6: About & Limitations
   - Synthetic Fake preset analysis run
   - Synthetic Real preset analysis run
3. Save screenshots to outputs/screenshots/ (git-ignored).
4. Sample actual rendered pixels from UI elements (page background, sidebar, code areas, tabs, buttons).
5. Audit sampled pixels for forbidden hues (blue: 170-255 deg, green: 65-165 deg) and near-black (lightness < 8%).
"""

import colorsys
import os
import shutil
import sys
import time
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("outputs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also copy to artifact directory if available
ARTIFACT_DIR = Path(r"C:\Users\dheer\.gemini\antigravity-ide\brain\35cae5b8-5ea6-49be-84c3-219dd6a58af1\screenshots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"

VIEWPORTS = [
    ("desktop", 1280, 800),
    ("mobile", 390, 844),
]

TAB_NAMES = [
    "Analyze",
    "Batch",
    "History",
    "Model & Results",
    "How It Works",
    "About & Limitations",
]


def wait_for_streamlit(page, timeout=30000):
    """Wait for Streamlit app to finish rendering and loading scripts."""
    page.wait_for_selector(".stApp", timeout=timeout)
    # Wait until status widget is idle
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=10000)
    except Exception:
        pass
    time.sleep(1.5)


def capture_all_screenshots():
    """Capture full-page desktop and mobile screenshots for all tabs and presets."""
    saved_files = []
    element_crops = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-lcd-text", "--disable-font-subpixel-positioning"],
        )

        for vp_name, width, height in VIEWPORTS:
            print(f"\n--- Capturing screenshots for viewport: {vp_name} ({width}x{height}) ---")
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            # Open main page
            page.goto(APP_URL, wait_until="networkidle", timeout=60000)
            wait_for_streamlit(page)

            # On mobile, close sidebar overlay if open so tabs and main page content are accessible
            if vp_name == "mobile":
                try:
                    close_btn = page.locator('button[aria-label="Close sidebar"], [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebar"] button').first
                    if close_btn.is_visible():
                        close_btn.click()
                        time.sleep(1.0)
                        wait_for_streamlit(page)
                except Exception as e:
                    print(f"  Note closing mobile sidebar: {e}")

            # 1. Capture each tab
            for tab_idx, tab_name in enumerate(TAB_NAMES):
                print(f"Switching to tab: {tab_name}")
                tab_locator = page.locator('[role="tab"]').filter(has_text=tab_name).first
                tab_locator.click(force=True)
                time.sleep(1.5)
                wait_for_streamlit(page)

                filename = f"{vp_name}_tab_{tab_idx+1}_{tab_name.lower().replace(' & ', '_').replace(' ', '_')}.png"
                out_path = OUTPUT_DIR / filename
                page.screenshot(path=str(out_path), full_page=True)
                shutil.copy(out_path, ARTIFACT_DIR / filename)
                saved_files.append((filename, out_path))
                print(f"  Saved: {out_path}")

                # For desktop tab 1, grab bounding boxes of key UI components for targeted pixel analysis
                if vp_name == "desktop" and tab_name == "Analyze":
                    try:
                        # Page background
                        st_app = page.query_selector(".stApp")
                        if st_app:
                            element_crops["page_bg"] = st_app.bounding_box()

                        # Sidebar
                        sidebar = page.query_selector('section[data-testid="stSidebar"]')
                        if sidebar:
                            element_crops["sidebar"] = sidebar.bounding_box()

                        # Tabs container
                        tab_list = page.query_selector('[data-baseweb="tab-list"]')
                        if tab_list:
                            element_crops["tabs"] = tab_list.bounding_box()

                        # Button
                        btn = page.query_selector('div.stButton > button[kind="primary"]')
                        if btn:
                            element_crops["button_primary"] = btn.bounding_box()

                        # Status chips / code
                        code_el = page.query_selector("code")
                        if code_el:
                            element_crops["code"] = code_el.bounding_box()
                    except Exception as e:
                        print(f"  Note: error collecting element bounding boxes: {e}")

            # 2. Run Analyze on Synthetic Fake Preset
            print("\nRunning Analyze on Synthetic Fake preset...")
            # Switch back to Analyze tab
            page.locator('[role="tab"]').filter(has_text="Analyze").first.click(force=True)
            time.sleep(1.0)

            # Select Synthetic Fake radio option
            page.locator('label').filter(has_text="Synthetic Fake Sample").first.click(force=True)
            time.sleep(1.5)
            wait_for_streamlit(page)

            # Click "Run Forensic Analysis" button
            run_btn = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn.is_enabled():
                run_btn.click(force=True)
                print("  Clicked 'Run Forensic Analysis' (Fake)")
                page.wait_for_selector(".verdict-banner", timeout=45000)
                time.sleep(2.0)
                wait_for_streamlit(page)

                fake_filename = f"{vp_name}_preset_fake_analyzed.png"
                fake_path = OUTPUT_DIR / fake_filename
                page.screenshot(path=str(fake_path), full_page=True)
                shutil.copy(fake_path, ARTIFACT_DIR / fake_filename)
                saved_files.append((fake_filename, fake_path))
                print(f"  Saved: {fake_path}")

            # 3. Run Analyze on Synthetic Real Preset
            print("\nRunning Analyze on Synthetic Real preset...")
            # Select Synthetic Real radio option
            page.locator('label').filter(has_text="Synthetic Real Sample").first.click(force=True)
            time.sleep(1.5)
            wait_for_streamlit(page)

            run_btn_real = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn_real.is_enabled():
                run_btn_real.click(force=True)
                print("  Clicked 'Run Forensic Analysis' (Real)")
                page.wait_for_selector(".verdict-banner", timeout=45000)
                time.sleep(2.0)
                wait_for_streamlit(page)

                real_filename = f"{vp_name}_preset_real_analyzed.png"
                real_path = OUTPUT_DIR / real_filename
                page.screenshot(path=str(real_path), full_page=True)
                shutil.copy(real_path, ARTIFACT_DIR / real_filename)
                saved_files.append((real_filename, real_path))
                print(f"  Saved: {real_path}")

            context.close()

        browser.close()

    return saved_files, element_crops


def sample_and_audit_pixels(desktop_screenshot_path: Path, element_crops: dict):
    """Sample actual rendered pixels of the page background, sidebar, code areas, tabs, and buttons.

    Reports any pixel whose hue is:
    - blue: 170-255 deg
    - green: 65-165 deg
    - lightness < 8% (near-black)
    """
    img = Image.open(desktop_screenshot_path).convert("RGB")
    width, height = img.size

    regions = {}
    # 1. Page background (clean background canvas)
    regions["page_background"] = (
        int(width * 0.70),
        30,
        int(width * 0.95),
        220,
    )

    # 2. Sidebar
    if element_crops.get("sidebar"):
        sb = element_crops["sidebar"]
        regions["sidebar"] = (
            int(sb["x"] + 10),
            int(sb["y"] + 20),
            int(sb["x"] + sb["width"] - 10),
            int(min(height, sb["y"] + sb["height"]) - 20),
        )
    else:
        regions["sidebar"] = (10, 20, 290, min(height, 780))

    # 3. Tabs
    if element_crops.get("tabs"):
        tb = element_crops["tabs"]
        regions["tabs"] = (
            int(tb["x"]),
            int(tb["y"]),
            int(tb["x"] + tb["width"]),
            int(tb["y"] + tb["height"]),
        )
    else:
        regions["tabs"] = (380, 310, 1200, 360)

    # 4. Buttons
    if element_crops.get("button_primary"):
        btn = element_crops["button_primary"]
        regions["buttons"] = (
            int(btn["x"]),
            int(btn["y"]),
            int(btn["x"] + btn["width"]),
            int(btn["y"] + btn["height"]),
        )
    else:
        regions["buttons"] = (380, 780, 823, min(height, 800))

    # 5. Code areas
    if element_crops.get("code"):
        cd = element_crops["code"]
        regions["code_areas"] = (
            int(cd["x"]),
            int(cd["y"]),
            int(cd["x"] + cd["width"]),
            int(cd["y"] + cd["height"]),
        )
    else:
        regions["code_areas"] = (427, 260, 480, 279)

    print("\n" + "=" * 70)
    print("PIXEL COLOR AUDIT REPORT ACROSS RENDERED UI COMPONENTS")
    print("=" * 70)

    audit_results = {}
    total_forbidden_pixels = 0

    for reg_name, (x1, y1, x2, y2) in regions.items():
        # Clamp to image size
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        crop = img.crop((x1, y1, x2, y2))
        pixels = list(crop.getdata())
        n_pixels = len(pixels)

        blue_pixels = 0
        green_pixels = 0
        black_pixels = 0
        dominant_colors = {}

        for r, g, b in pixels:
            # Quantize for dominant display
            q_rgb = (r // 16 * 16, g // 16 * 16, b // 16 * 16)
            dominant_colors[q_rgb] = dominant_colors.get(q_rgb, 0) + 1

            rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
            h, l, s = colorsys.rgb_to_hls(rf, gf, bf)
            hue_deg = h * 360.0
            light_pct = l * 100.0
            sat_pct = s * 100.0

            # 1. Near black: lightness < 8%
            if light_pct < 8.0:
                black_pixels += 1

            # 2. Green hue: 65 - 165 deg (excluding neutral/desaturated near-white)
            elif sat_pct >= 10.0 and light_pct < 88.0 and (65.0 <= hue_deg <= 165.0):
                green_pixels += 1

            # 3. Blue hue: 170 - 255 deg (excluding violet >= 260 deg and neutral)
            elif sat_pct >= 10.0 and light_pct < 88.0 and (170.0 <= hue_deg <= 255.0):
                blue_pixels += 1

        reg_violations = blue_pixels + green_pixels + black_pixels
        total_forbidden_pixels += reg_violations

        # Find top 2 dominant hexes
        sorted_dom = sorted(dominant_colors.items(), key=lambda x: x[1], reverse=True)[:2]
        dom_hexes = [f"#{r:02x}{g:02x}{b:02x} ({cnt/n_pixels*100:.1f}%)" for (r, g, b), cnt in sorted_dom]

        audit_results[reg_name] = {
            "total_pixels": n_pixels,
            "blue_pixels": blue_pixels,
            "green_pixels": green_pixels,
            "black_pixels": black_pixels,
            "violations": reg_violations,
            "dom_hexes": dom_hexes,
        }

        status = "PASSED (0 violations)" if reg_violations == 0 else f"FAILED ({reg_violations} violations)"
        print(f"[{reg_name.upper()}]: {status}")
        print(f"  Sampled: {n_pixels:,} pixels | Box: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"  Dominant: {', '.join(dom_hexes)}")
        print(f"  Blue pixels (170-255 deg): {blue_pixels}")
        print(f"  Green pixels (65-165 deg): {green_pixels}")
        print(f"  Near-black pixels (< 8% L): {black_pixels}")
        print("-" * 70)

    print(f"\nAUDIT SUMMARY: Total UI Violations = {total_forbidden_pixels}")
    if total_forbidden_pixels == 0:
        print("PERFECT COMPLIANCE: ZERO blue, green, or near-black pixels found in rendered UI components!")
    else:
        print(f"WARNING: Found {total_forbidden_pixels} forbidden pixels in UI elements.")

    return audit_results, total_forbidden_pixels


if __name__ == "__main__":
    print("Starting Playwright UI screenshot capture and pixel audit...")
    screenshots, crops = capture_all_screenshots()
    print(f"\nSuccessfully captured {len(screenshots)} screenshots in outputs/screenshots/")

    # Audit the desktop Tab 1 screenshot
    desktop_tab1 = OUTPUT_DIR / "desktop_tab_1_analyze.png"
    if desktop_tab1.exists():
        audit_results, violations = sample_and_audit_pixels(desktop_tab1, crops)
    else:
        print(f"Error: {desktop_tab1} not found")
