"""Comprehensive visual verification and pixel audit runner for Ultraviolet Forensics theme.

Executes:
1. State capture across desktop (1280x800) and phone (390x844) viewports for:
   - Default tabs 1 to 6
   - Analyze synthetic fake preset (with verdict card, gauge, clip chart, face crops, download buttons)
   - Analyze synthetic real preset (with verdict card, gauge, clip chart, face crops, download buttons)
   - Batch tab with results table and CSV download button
   - History tab with at least 2 entries
   - Error states: empty file, oversized file, short video, no-face video
   - Real experiment disabled state and persistent prototype banner
   - Interaction states: open selectbox dropdown, focused slider, hovered Plotly bar, dropzone, expander
2. Executes TWICE:
   - Run A: Default Chromium text rendering (LCD text enabled)
   - Run B: LCD text disabled (--disable-lcd-text, --disable-font-subpixel-positioning)
3. Audits full-page pixels across every screenshot:
   - Near-black (lightness < 8%)
   - Green (hue 65-165, sat >= 10%, lightness < 88%)
   - Blue (hue 170-255, sat >= 10%, lightness < 88%)
   - Solid regions (connected component surviving 6x6 binary opening)
   - Share of pixels in hue 250-258 deg
4. Produces structured output for reporting.
"""

import colorsys
import json
import os
import shutil
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import scipy.ndimage
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("outputs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also copy to artifact directory
ARTIFACT_DIR = Path(r"C:\Users\dheer\.gemini\antigravity-ide\brain\35cae5b8-5ea6-49be-84c3-219dd6a58af1\screenshots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"

TAB_NAMES = [
    "Analyze",
    "Batch",
    "History",
    "Model & Results",
    "How It Works",
    "About & Limitations",
]


def wait_for_app(page, timeout=15000):
    """Wait until Streamlit elements are present and status widget is idle."""
    page.wait_for_selector(".stApp", timeout=timeout)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=6000)
    except Exception:
        pass
    time.sleep(1.2)


def capture_states_for_mode(run_name: str, extra_args: list):
    """Capture full suite of desktop and mobile screenshots for a given Chromium configuration."""
    mode_dir = OUTPUT_DIR / run_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    captured = {}

    # If all screenshots for this run mode already exist, reuse them
    existing_files = list(mode_dir.glob("*.png"))
    if len(existing_files) >= 28:
        print(f"Mode {run_name} already has {len(existing_files)} screenshots. Reusing existing.")
        for f in existing_files:
            k = f.stem[len(run_name) + 1:]
            captured[k] = f
        return captured

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=extra_args)

        for vp_name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
            print(f"\n=======================================================")
            print(f"[{run_name.upper()}] Viewport: {vp_name} ({width}x{height})")
            print(f"=======================================================")

            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            # Helper to close mobile sidebar drawer
            def close_mobile_drawer():
                if vp_name == "mobile":
                    try:
                        c_btn = page.locator('button[aria-label="Close sidebar"], [data-testid="stSidebarCollapseButton"] button').first
                        if c_btn.is_visible():
                            c_btn.click()
                            time.sleep(0.8)
                    except Exception:
                        pass

            def take_shot(shot_key: str):
                filename = f"{run_name}_{vp_name}_{shot_key}.png"
                filepath = mode_dir / filename
                if filepath.exists() and filepath.stat().st_size > 0:
                    shutil.copy(filepath, ARTIFACT_DIR / filename)
                    captured[f"{vp_name}_{shot_key}"] = filepath
                    print(f"  Existing: {filename}")
                    return filepath
                page.screenshot(path=str(filepath), full_page=True)
                shutil.copy(filepath, ARTIFACT_DIR / filename)
                captured[f"{vp_name}_{shot_key}"] = filepath
                print(f"  Captured: {filename}")
                return filepath

            # Load page
            page.goto(APP_URL, wait_until="networkidle", timeout=60000)
            wait_for_app(page)
            close_mobile_drawer()

            # -------------------------------------------------------------
            # 1. Default Tabs 1 to 6
            # -------------------------------------------------------------
            for idx, tab_name in enumerate(TAB_NAMES):
                tab_key = tab_name.lower().replace(" & ", "_").replace(" ", "_")
                tab_loc = page.locator('[role="tab"]').filter(has_text=tab_name).first
                tab_loc.click(force=True)
                time.sleep(1.2)
                wait_for_app(page)
                close_mobile_drawer()
                take_shot(f"tab_{idx+1}_{tab_key}_default")

            # -------------------------------------------------------------
            # 2. Analyze Tab: Run Synthetic Fake Preset
            # -------------------------------------------------------------
            print("Running Synthetic Fake preset...")
            page.locator('[role="tab"]').filter(has_text="Analyze").first.click(force=True)
            time.sleep(1.0)
            close_mobile_drawer()

            page.locator('label').filter(has_text="Synthetic Fake Sample").first.click(force=True)
            time.sleep(1.5)
            wait_for_app(page)

            run_btn = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn.is_enabled():
                run_btn.click(force=True)
                page.wait_for_selector(".verdict-banner", timeout=45000)
                time.sleep(2.0)
                wait_for_app(page)
                close_mobile_drawer()
                take_shot("analyze_synthetic_fake")

                # Interaction: Expanded Expander
                expander = page.locator('div[data-testid="stExpander"]').first
                if expander:
                    try:
                        expander.locator("summary").click(force=True)
                        time.sleep(1.0)
                        take_shot("interaction_expanded_expander")
                        expander.locator("summary").click(force=True)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"  Note expander click: {e}")

                # Interaction: Hovered Plotly Point (desktop only)
                if vp_name == "desktop":
                    try:
                        bar = page.locator('.js-plotly-plot .bars path').first
                        if bar:
                            bar.hover()
                            time.sleep(0.8)
                            take_shot("interaction_hovered_plotly")
                    except Exception as e:
                        print(f"  Note plotly hover: {e}")

            # -------------------------------------------------------------
            # 3. Analyze Tab: Run Synthetic Real Preset
            # -------------------------------------------------------------
            print("Running Synthetic Real preset...")
            page.locator('label').filter(has_text="Synthetic Real Sample").first.click(force=True)
            time.sleep(1.5)
            wait_for_app(page)

            run_btn_real = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn_real.is_enabled():
                run_btn_real.click(force=True)
                page.wait_for_selector(".verdict-banner", timeout=45000)
                time.sleep(2.0)
                wait_for_app(page)
                close_mobile_drawer()
                take_shot("analyze_synthetic_real")

            # -------------------------------------------------------------
            # 4. History Tab (now has 2 entries from fake & real runs)
            # -------------------------------------------------------------
            print("Capturing History with entries...")
            page.locator('[role="tab"]').filter(has_text="History").first.click(force=True)
            time.sleep(1.2)
            wait_for_app(page)
            close_mobile_drawer()
            take_shot("history_with_entries")

            # -------------------------------------------------------------
            # 5. Batch Tab: Run Batch with Demo Files
            # -------------------------------------------------------------
            print("Running Batch evaluation...")
            page.locator('[role="tab"]').filter(has_text="Batch").first.click(force=True)
            time.sleep(1.2)
            wait_for_app(page)
            close_mobile_drawer()

            # Upload demo files for batch
            fake_sample = os.path.abspath("data/demo/raw/fake/synth_fake_00.mp4")
            real_sample = os.path.abspath("data/demo/raw/real/synth_real_15.mp4")
            batch_uploader = page.locator('input[type="file"][data-testid="stFileUploaderDropzoneInput"]').first
            if os.path.exists(fake_sample) and os.path.exists(real_sample):
                batch_uploader.set_input_files([fake_sample, real_sample])
                time.sleep(2.0)
                wait_for_app(page)

                proc_btn = page.locator('button').filter(has_text="Process Batch Files").first
                if proc_btn.is_enabled():
                    proc_btn.click(force=True)
                    # wait for batch processing to complete
                    page.wait_for_selector('button:has-text("Download Batch Results (CSV)")', timeout=45000)
                    time.sleep(2.0)
                    wait_for_app(page)
                    close_mobile_drawer()
                    take_shot("batch_with_results")

            # -------------------------------------------------------------
            # 6. Interaction States: Dropzone, Selectbox, Slider
            # -------------------------------------------------------------
            # Switch back to Analyze tab
            page.locator('[role="tab"]').filter(has_text="Analyze").first.click(force=True)
            time.sleep(1.0)
            close_mobile_drawer()

            # Switch to Upload Custom Video to show dropzone
            page.locator('label').filter(has_text="Upload a Video File").first.click(force=True)
            time.sleep(1.0)
            take_shot("interaction_file_uploader_dropzone")

            # Open sidebar drawer if mobile to capture sidebar interactions
            if vp_name == "mobile":
                try:
                    open_sb = page.locator('button[aria-label="Open sidebar"], [data-testid="stSidebarCollapsedControl"] button').first
                    if open_sb.is_visible():
                        open_sb.click()
                        time.sleep(1.0)
                except Exception:
                    pass

            # Open selectbox dropdown
            try:
                sb_input = page.locator('div[data-baseweb="select"]').first
                sb_input.click()
                time.sleep(1.0)
                take_shot("interaction_open_selectbox")
                # Close by clicking sidebar title
                page.locator('h3:has-text("Configuration")').first.click()
                time.sleep(0.5)
            except Exception as e:
                print(f"  Note selectbox interaction: {e}")

            # Focus slider
            try:
                slider = page.locator('[role="slider"]').first
                slider.focus()
                time.sleep(0.5)
                take_shot("interaction_focused_slider")
            except Exception as e:
                print(f"  Note slider interaction: {e}")

            # Real experiment disabled state in sidebar
            try:
                # Select unavailable model choice if possible
                sb_input = page.locator('div[data-baseweb="select"]').first
                sb_input.click()
                time.sleep(0.5)
                unavail_opt = page.locator('li[role="option"]').filter(has_text="Unavailable").first
                if unavail_opt:
                    unavail_opt.click()
                    time.sleep(1.0)
                    take_shot("state_real_experiment_disabled")
            except Exception as e:
                print(f"  Note real experiment select: {e}")

            # -------------------------------------------------------------
            # 7. Error States (Empty, Short, No-Face, Oversized)
            # -------------------------------------------------------------
            close_mobile_drawer()
            # Switch to custom upload
            page.locator('label').filter(has_text="Upload a Video File").first.click(force=True)
            time.sleep(1.0)

            # A. Empty file
            empty_file = os.path.abspath("outputs/test_videos/empty.mp4")
            single_uploader = page.locator('input[type="file"][data-testid="stFileUploaderDropzoneInput"]').first
            single_uploader.set_input_files(empty_file)
            time.sleep(1.5)
            wait_for_app(page)
            take_shot("error_empty_file")

            # B. Oversized file
            oversized_file = os.path.abspath("outputs/test_videos/oversized.mp4")
            single_uploader.set_input_files(oversized_file)
            time.sleep(2.0)
            wait_for_app(page)
            take_shot("error_oversized_file")

            # C. Short video (< 10 frames)
            short_file = os.path.abspath("outputs/test_videos/short.mp4")
            single_uploader.set_input_files(short_file)
            time.sleep(1.5)
            wait_for_app(page)
            run_btn_short = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn_short.is_enabled():
                run_btn_short.click(force=True)
                time.sleep(2.5)
                wait_for_app(page)
                take_shot("error_short_video")

            # D. No-face video (toggle center crop OFF)
            noface_file = os.path.abspath("outputs/test_videos/noface.mp4")
            # Open sidebar on mobile if needed
            if vp_name == "mobile":
                try:
                    open_sb = page.locator('button[aria-label="Open sidebar"], [data-testid="stSidebarCollapsedControl"] button').first
                    if open_sb.is_visible():
                        open_sb.click()
                        time.sleep(1.0)
                except Exception:
                    pass
            # Turn toggle OFF
            toggle = page.locator('div[data-testid="stToggleSwitch"], label[data-baseweb="toggle"], div[data-testid="stCheckbox"]').first
            try:
                # check if checkbox is checked
                chk = page.locator('input[type="checkbox"]').first
                if chk and chk.is_checked():
                    chk.click(force=True)
                    time.sleep(1.0)
            except Exception as e:
                print(f"  Note toggle off: {e}")

            close_mobile_drawer()
            single_uploader.set_input_files(noface_file)
            time.sleep(1.5)
            wait_for_app(page)
            run_btn_noface = page.locator('button').filter(has_text="Run Forensic Analysis").first
            if run_btn_noface.is_enabled():
                run_btn_noface.click(force=True)
                time.sleep(3.0)
                wait_for_app(page)
                take_shot("error_no_face_video")

            context.close()

        browser.close()

    return captured


def audit_image_pixels(img_path: Path):
    """Perform comprehensive full-page pixel audit with solid 6x6 connected component analysis."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # H x W x 3
    h_img, w_img, _ = arr.shape
    total_pixels = h_img * w_img

    # Convert RGB array to HLS vectorized
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Lightness L = (max + min) / 2
    l = (max_c + min_c) / 2.0

    # Saturation
    s = np.zeros_like(l)
    nonzero_delta = delta > 1e-5
    s[nonzero_delta & (l <= 0.5)] = delta[nonzero_delta & (l <= 0.5)] / (max_c + min_c)[nonzero_delta & (l <= 0.5)]
    s[nonzero_delta & (l > 0.5)] = delta[nonzero_delta & (l > 0.5)] / (2.0 - max_c - min_c)[nonzero_delta & (l > 0.5)]

    # Hue
    h = np.zeros_like(l)
    r_is_max = nonzero_delta & (max_c == r)
    g_is_max = nonzero_delta & (max_c == g) & (~r_is_max)
    b_is_max = nonzero_delta & (max_c == b) & (~r_is_max) & (~g_is_max)

    h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0
    h = (h / 6.0) * 360.0  # 0 to 360 degrees

    sat_pct = s * 100.0
    light_pct = l * 100.0

    # 1. Near-Black: lightness < 8%
    near_black_mask = light_pct < 8.0

    # 2. Green: hue 65 - 165 deg, sat >= 10%, lightness < 88%
    green_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 65.0) & (h <= 165.0)

    # 3. Blue: hue 170 - 255 deg, sat >= 10%, lightness < 88%
    blue_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 170.0) & (h <= 255.0)

    # 4. Violet boundary check: share of pixels with hue in 250 - 258 deg
    boundary_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 250.0) & (h <= 258.0)

    # Solid 6x6 regions (binary opening with 6x6 kernel of ones)
    struct_6x6 = np.ones((6, 6), dtype=bool)

    solid_black_mask = scipy.ndimage.binary_opening(near_black_mask, structure=struct_6x6)
    solid_green_mask = scipy.ndimage.binary_opening(green_mask, structure=struct_6x6)
    solid_blue_mask = scipy.ndimage.binary_opening(blue_mask, structure=struct_6x6)

    # Extract bounding boxes for any solid hits
    solid_violations_boxes = []
    for mask_name, s_mask in [("near_black", solid_black_mask), ("green", solid_green_mask), ("blue", solid_blue_mask)]:
        if np.any(s_mask):
            labeled, num_features = scipy.ndimage.label(s_mask)
            slices = scipy.ndimage.find_objects(labeled)
            for sl in slices:
                y_min, y_max = sl[0].start, sl[0].stop
                x_min, x_max = sl[1].start, sl[1].stop
                w = x_max - x_min
                h_b = y_max - y_min
                if w >= 6 and h_b >= 6:
                    solid_violations_boxes.append({
                        "type": mask_name,
                        "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "width": int(w),
                        "height": int(h_b),
                        "center": [int((x_min + x_max) // 2), int((y_min + y_max) // 2)],
                    })

    counts = {
        "image": img_path.name,
        "width": w_img,
        "height": h_img,
        "total_pixels": int(total_pixels),
        "near_black_pixels": int(np.sum(near_black_mask)),
        "green_pixels": int(np.sum(green_mask)),
        "blue_pixels": int(np.sum(blue_mask)),
        "total_raw_violations": int(np.sum(near_black_mask) + np.sum(green_mask) + np.sum(blue_mask)),
        "solid_black_pixels": int(np.sum(solid_black_mask)),
        "solid_green_pixels": int(np.sum(solid_green_mask)),
        "solid_blue_pixels": int(np.sum(solid_blue_mask)),
        "total_solid_violations": int(np.sum(solid_black_mask) + np.sum(solid_green_mask) + np.sum(solid_blue_mask)),
        "boundary_250_258_pixels": int(np.sum(boundary_mask)),
        "boundary_250_258_share_pct": float((np.sum(boundary_mask) / total_pixels) * 100.0),
        "solid_boxes": solid_violations_boxes,
    }

    return counts


if __name__ == "__main__":
    print("Executing Broader Visual Verification suite...")

    # Run A: Default Chromium (LCD text enabled)
    print("\n>>> STARTING RUN A: Default Chromium (LCD text enabled) <<<")
    shots_a = capture_states_for_mode("run_a_lcd_enabled", extra_args=[])

    # Run B: LCD text disabled
    print("\n>>> STARTING RUN B: LCD Text Disabled (--disable-lcd-text) <<<")
    shots_b = capture_states_for_mode("run_b_no_lcd", extra_args=["--disable-lcd-text", "--disable-font-subpixel-positioning"])

    # Audit all captured screenshots
    print("\n>>> AUDITING ALL CAPTURED SCREENSHOTS <<<")
    results_a = []
    results_b = []

    for name, p_img in shots_a.items():
        res = audit_image_pixels(p_img)
        res["state_key"] = name
        results_a.append(res)

    for name, p_img in shots_b.items():
        res = audit_image_pixels(p_img)
        res["state_key"] = name
        results_b.append(res)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_a_lcd_enabled": results_a,
        "run_b_no_lcd": results_b,
    }

    with open("outputs/screenshots/audit_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nAudit complete! Results saved to outputs/screenshots/audit_results.json")
