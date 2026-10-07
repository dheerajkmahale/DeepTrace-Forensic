"""Comprehensive visual verification and pixel audit runner for Ultraviolet Forensics theme.

Executes:
1. State capture across desktop (1280x800) and phone (390x844) viewports for 28 states:
   - 6 tabs default (desktop + phone = 12)
   - Analyze synthetic fake preset (desktop + phone = 2)
   - Expanded expander interaction (desktop + phone = 2)
   - Analyze synthetic real preset (desktop + phone = 2)
   - History with 2 entries (desktop + phone = 2)
   - Batch evaluation with results (desktop + phone = 2)
   - File uploader dropzone (desktop + phone = 2)
   - 4 error states: empty, oversized, short, no-face (desktop = 4)
   Total: 16 desktop + 12 mobile = 28 screenshots per run.
2. Executes TWICE:
   - Run A: Default Chromium text rendering (LCD text enabled)
   - Run B: LCD text disabled (--disable-lcd-text, --disable-font-subpixel-positioning)
   Total: 56 screenshots audited.
3. Sanity check:
   - Rejects any screenshot with < 500 distinct colors or > 90% single-color coverage as blank/masked.
4. Audits:
   - Separates UI Chrome from Video Frame (user footage) content.
   - Near-black (lightness < 8%)
   - Green (hue 65-165, sat >= 10%, lightness < 88%)
   - Blue (hue 170-255, sat >= 10%, lightness < 88%)
   - Solid 6x6 regions (binary opening with 6x6 kernel)
   - Violet boundary 250-258 deg hue share
"""

import argparse
import colorsys
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("outputs/screenshots")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots")))
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


def wait_for_app(page, timeout=30000):
    """Wait until Streamlit elements are present and status widget is idle."""
    page.wait_for_selector(".stApp", timeout=timeout)
    page.wait_for_selector('[role="tab"]', timeout=timeout)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=8000)
    except Exception:
        pass
    time.sleep(1.2)


def check_sanity(filepath: Path):
    """Validate that screenshot is not a blank, broken, or masked capture."""
    im = Image.open(filepath).convert("RGB")
    arr = np.array(im)
    packed = (arr[:, :, 0].astype(np.uint32) << 16) | (arr[:, :, 1].astype(np.uint32) << 8) | (arr[:, :, 2].astype(np.uint32))
    unique_colors, counts = np.unique(packed, return_counts=True)
    num_distinct = len(unique_colors)
    max_share = np.max(counts) / counts.sum()

    if num_distinct < 500:
        raise RuntimeError(f"Sanity check FAILED for {filepath.name}: only {num_distinct} distinct colors (< 500). Blank/masked capture!")
    if max_share > 0.90:
        raise RuntimeError(f"Sanity check FAILED for {filepath.name}: single color covers {max_share:.1%} of pixels (> 90%). Blank/masked capture!")
    return num_distinct, max_share


def capture_states_for_mode(run_name: str, extra_args: list, reuse: bool = False):
    """Capture 28 desktop and mobile screenshots for a given Chromium configuration."""
    mode_dir = OUTPUT_DIR / run_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    captured = {}
    video_bboxes = {}
    dom_elements_map = {}

    if reuse:
        existing_files = list(mode_dir.glob("*.png"))
        if len(existing_files) >= 28:
            print(f"Mode {run_name} already has {len(existing_files)} screenshots. Reusing existing.")
            for f in existing_files:
                k = f.stem[len(run_name) + 1:]
                captured[k] = f
            return captured, video_bboxes, dom_elements_map

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=extra_args)

        for vp_name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
            print(f"\n=======================================================")
            print(f"[{run_name.upper()}] Viewport: {vp_name} ({width}x{height})")
            print(f"=======================================================")

            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

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
                full_key = f"{vp_name}_{shot_key}"
                filename = f"{run_name}_{full_key}.png"
                filepath = mode_dir / filename

                # Check if video frame element is present to record its bounding box
                try:
                    video_el = page.locator('#forensic-custom-video, video').first
                    if video_el.count() > 0 and video_el.is_visible():
                        box = video_el.bounding_box()
                        if box:
                            video_bboxes[full_key] = box
                except Exception:
                    pass

                page.screenshot(path=str(filepath), full_page=True)
                check_sanity(filepath)
                shutil.copy(filepath, ARTIFACT_DIR / filename)
                captured[full_key] = filepath
                print(f"  Captured: {filename}")
                return filepath

            # Load page
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60000)
            wait_for_app(page)
            close_mobile_drawer()

            # -------------------------------------------------------------
            # 1. Default Tabs 1 to 6 (6 shots)
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
            # 2. Analyze Tab: Run Synthetic Fake Preset (2 shots: result + expander)
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

                # Expanded Expander interaction
                expander = page.locator('div[data-testid="stExpander"]').first
                if expander.count() > 0:
                    try:
                        expander.locator("summary").click(force=True)
                        time.sleep(1.0)
                        take_shot("interaction_expanded_expander")
                        expander.locator("summary").click(force=True)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"  Note expander: {e}")

            # -------------------------------------------------------------
            # 3. Analyze Tab: Run Synthetic Real Preset (1 shot)
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
            # 4. History Tab with Entries (1 shot)
            # -------------------------------------------------------------
            print("Capturing History with entries...")
            page.locator('[role="tab"]').filter(has_text="History").first.click(force=True)
            time.sleep(1.2)
            wait_for_app(page)
            close_mobile_drawer()
            take_shot("history_with_entries")

            # -------------------------------------------------------------
            # 5. Batch Tab: Run Batch with Demo Files (1 shot)
            # -------------------------------------------------------------
            print("Running Batch evaluation...")
            page.locator('[role="tab"]').filter(has_text="Batch").first.click(force=True)
            time.sleep(1.2)
            wait_for_app(page)
            close_mobile_drawer()

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
                    page.wait_for_selector('button:has-text("Download Batch Results (CSV)")', timeout=45000)
                    time.sleep(2.0)
                    wait_for_app(page)
                    close_mobile_drawer()
                    take_shot("batch_with_results")

            # -------------------------------------------------------------
            # 6. Interaction: Dropzone (1 shot)
            # -------------------------------------------------------------
            page.locator('[role="tab"]').filter(has_text="Analyze").first.click(force=True)
            time.sleep(1.0)
            close_mobile_drawer()
            page.locator('label').filter(has_text="Upload a Video File").first.click(force=True)
            time.sleep(1.0)
            take_shot("interaction_file_uploader_dropzone")

            # -------------------------------------------------------------
            # 7. Error States (Desktop only: 4 shots)
            # -------------------------------------------------------------
            if vp_name == "desktop":
                empty_file = os.path.abspath("outputs/test_videos/empty.mp4")
                single_uploader = page.locator('input[type="file"][data-testid="stFileUploaderDropzoneInput"]').first
                if os.path.exists(empty_file):
                    single_uploader.set_input_files(empty_file)
                    time.sleep(1.5)
                    wait_for_app(page)
                    take_shot("error_empty_file")

                oversized_file = os.path.abspath("outputs/test_videos/oversized.mp4")
                if os.path.exists(oversized_file):
                    single_uploader.set_input_files(oversized_file)
                    time.sleep(2.0)
                    wait_for_app(page)
                    take_shot("error_oversized_file")

                short_file = os.path.abspath("outputs/test_videos/short.mp4")
                if os.path.exists(short_file):
                    single_uploader.set_input_files(short_file)
                    time.sleep(1.5)
                    wait_for_app(page)
                    run_btn_short = page.locator('button').filter(has_text="Run Forensic Analysis").first
                    if run_btn_short.is_enabled():
                        run_btn_short.click(force=True)
                        time.sleep(2.5)
                        wait_for_app(page)
                        take_shot("error_short_video")

                noface_file = os.path.abspath("outputs/test_videos/noface.mp4")
                if os.path.exists(noface_file):
                    # Turn off center-crop fallback toggle
                    chk = page.locator('input[type="checkbox"]').first
                    if chk and chk.is_checked():
                        chk.click(force=True)
                        time.sleep(1.0)

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

    return captured, video_bboxes, dom_elements_map


def audit_image_pixels(img_path: Path, video_bbox: Optional[dict] = None):
    """Perform comprehensive full-page pixel audit separating UI Chrome from Video Frame footage."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # H x W x 3
    h_img, w_img, _ = arr.shape
    total_pixels = h_img * w_img

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    l = (max_c + min_c) / 2.0

    s = np.zeros_like(l)
    nonzero_delta = delta > 1e-5
    s[nonzero_delta & (l <= 0.5)] = delta[nonzero_delta & (l <= 0.5)] / (max_c + min_c)[nonzero_delta & (l <= 0.5)]
    s[nonzero_delta & (l > 0.5)] = delta[nonzero_delta & (l > 0.5)] / (2.0 - max_c - min_c)[nonzero_delta & (l > 0.5)]

    h = np.zeros_like(l)
    r_is_max = nonzero_delta & (max_c == r)
    g_is_max = nonzero_delta & (max_c == g) & (~r_is_max)
    b_is_max = nonzero_delta & (max_c == b) & (~r_is_max) & (~g_is_max)

    h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0
    h = (h / 6.0) * 360.0

    sat_pct = s * 100.0
    light_pct = l * 100.0

    # 1. Near-Black: lightness < 8%
    near_black_mask = light_pct < 8.0

    # 2. Green: hue 65 - 165 deg, sat >= 10%, lightness < 88%
    green_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 65.0) & (h <= 165.0)

    # 3. Blue: hue 170 - 255 deg, sat >= 10%, lightness < 88%
    blue_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 170.0) & (h <= 255.0)

    # 4. Violet boundary: hue 250 - 258 deg
    boundary_mask = (sat_pct >= 10.0) & (light_pct < 88.0) & (h >= 250.0) & (h <= 258.0)

    # Solid 6x6 regions
    struct_6x6 = np.ones((6, 6), dtype=bool)
    solid_black_mask = scipy.ndimage.binary_opening(near_black_mask, structure=struct_6x6)
    solid_green_mask = scipy.ndimage.binary_opening(green_mask, structure=struct_6x6)
    solid_blue_mask = scipy.ndimage.binary_opening(blue_mask, structure=struct_6x6)

    # Region separation: UI Chrome vs Video Frame (user media footage)
    is_video_frame = np.zeros((h_img, w_img), dtype=bool)
    if video_bbox is not None:
        vx = int(max(0, video_bbox.get("x", 0)))
        vy = int(max(0, video_bbox.get("y", 0)))
        vw = int(video_bbox.get("width", 0))
        vh = int(video_bbox.get("height", 0))
        is_video_frame[vy : min(h_img, vy + vh), vx : min(w_img, vx + vw)] = True

    is_ui_chrome = ~is_video_frame

    # Counts
    ui_chrome_raw = int(np.sum((near_black_mask | green_mask | blue_mask) & is_ui_chrome))
    ui_chrome_solid = int(np.sum((solid_black_mask | solid_green_mask | solid_blue_mask) & is_ui_chrome))
    video_frame_raw = int(np.sum((near_black_mask | green_mask | blue_mask) & is_video_frame))
    video_frame_solid = int(np.sum((solid_black_mask | solid_green_mask | solid_blue_mask) & is_video_frame))

    # Extract bounding boxes for any solid UI chrome hits
    chrome_solid_boxes = []
    for mask_name, s_mask in [("near_black", solid_black_mask & is_ui_chrome),
                              ("green", solid_green_mask & is_ui_chrome),
                              ("blue", solid_blue_mask & is_ui_chrome)]:
        if np.any(s_mask):
            labeled, num_features = scipy.ndimage.label(s_mask)
            slices = scipy.ndimage.find_objects(labeled)
            for sl in slices:
                y_min, y_max = sl[0].start, sl[0].stop
                x_min, x_max = sl[1].start, sl[1].stop
                w = x_max - x_min
                h_b = y_max - y_min
                if w >= 6 and h_b >= 6:
                    chrome_solid_boxes.append({
                        "type": mask_name,
                        "region": "ui_chrome",
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
        "ui_chrome_raw_violations": ui_chrome_raw,
        "ui_chrome_solid_violations": ui_chrome_solid,
        "video_frame_raw_violations": video_frame_raw,
        "video_frame_solid_violations": video_frame_solid,
        "total_raw_violations": int(np.sum(near_black_mask) + np.sum(green_mask) + np.sum(blue_mask)),
        "total_solid_violations": int(np.sum(solid_black_mask) + np.sum(solid_green_mask) + np.sum(solid_blue_mask)),
        "boundary_250_258_share_pct": float((np.sum(boundary_mask) / total_pixels) * 100.0),
        "chrome_solid_boxes": chrome_solid_boxes,
        "has_video_frame": bool(video_bbox is not None),
    }
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Broader visual verification suite")
    parser.add_argument("--reuse", action="store_true", default=False, help="Reuse existing screenshots (default: False)")
    args = parser.parse_args()

    if not args.reuse and OUTPUT_DIR.exists():
        print(f"Deleting existing screenshot directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> EXECUTING BROADER VISUAL VERIFICATION SUITE FROM SCRATCH <<<")

    # Run A: Default Chromium (LCD text enabled)
    print("\n>>> STARTING RUN A: Default Chromium (LCD text enabled) <<<")
    shots_a, bboxes_a, dom_a = capture_states_for_mode("run_a_lcd_enabled", extra_args=[], reuse=args.reuse)

    # Run B: LCD text disabled
    print("\n>>> STARTING RUN B: LCD Text Disabled (--disable-lcd-text) <<<")
    shots_b, bboxes_b, dom_b = capture_states_for_mode(
        "run_b_no_lcd",
        extra_args=["--disable-lcd-text", "--disable-font-subpixel-positioning"],
        reuse=args.reuse,
    )

    # Audit all captured screenshots
    print("\n>>> AUDITING ALL CAPTURED SCREENSHOTS <<<")
    results_a = []
    results_b = []

    for name, p_img in sorted(shots_a.items()):
        bbox = bboxes_a.get(name)
        res = audit_image_pixels(p_img, video_bbox=bbox)
        res["state_key"] = name
        results_a.append(res)

    for name, p_img in sorted(shots_b.items()):
        bbox = bboxes_b.get(name)
        res = audit_image_pixels(p_img, video_bbox=bbox)
        res["state_key"] = name
        results_b.append(res)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_screenshots_run_a": len(results_a),
        "total_screenshots_run_b": len(results_b),
        "run_a_lcd_enabled": results_a,
        "run_b_no_lcd": results_b,
    }

    with open("outputs/screenshots/audit_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAudit complete! Captured {len(results_a)} Run A + {len(results_b)} Run B = {len(results_a)+len(results_b)} total screenshots.")
    print("Results saved to outputs/screenshots/audit_results.json")
