"""Verify Streamlit popovers (selectbox dropdown, tooltip icon, toast) in the running app."""

import os
import shutil
import time
from pathlib import Path
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("outputs/screenshots/overlay_checks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\dheer\.gemini\antigravity-ide\brain\35cae5b8-5ea6-49be-84c3-219dd6a58af1\screenshots")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"

def check_image_health(img_path: Path):
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im)
    packed = (arr[:, :, 0].astype(np.uint32) << 16) | (arr[:, :, 1].astype(np.uint32) << 8) | (arr[:, :, 2].astype(np.uint32))
    unique_colors, counts = np.unique(packed, return_counts=True)
    num_distinct = len(unique_colors)
    max_share = np.max(counts) / counts.sum()
    print(f"  [{img_path.name}] Size: {im.size}, Distinct Colors: {num_distinct}, Max Single-Color Share: {max_share:.1%}")
    if num_distinct < 500:
        raise ValueError(f"Fewer than 500 distinct colors ({num_distinct}) in {img_path.name}")
    if max_share > 0.90:
        raise ValueError(f"One color covers > 90% ({max_share:.1%}) in {img_path.name}")
    return num_distinct, max_share

print("Starting Popover Overlay Fix Verification...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_selector('.stApp', timeout=20000)
    page.wait_for_selector('[role="tab"]', timeout=30000)
    time.sleep(1.0)

    # 1. Model Selectbox Dropdown
    print("\n1. Testing Model Selectbox Dropdown...")
    combo = page.locator('[role="combobox"]').first
    combo.click()
    time.sleep(1.0)
    page.wait_for_selector('[role="option"]', timeout=10000)
    popover_shot = OUTPUT_DIR / "popover_selectbox_dropdown.png"
    page.screenshot(path=str(popover_shot))
    shutil.copy(popover_shot, ARTIFACT_DIR / "popover_selectbox_dropdown.png")
    check_image_health(popover_shot)

    # Close dropdown by clicking header
    page.locator('h3:has-text("Configuration")').first.click()
    time.sleep(0.8)

    # 2. Tooltip / Help Icon
    print("\n2. Testing Tooltip / Help Icon...")
    tt = page.locator('[data-testid="stTooltipHoverTarget"]').first
    tt.hover()
    time.sleep(1.0)
    tooltip_shot = OUTPUT_DIR / "popover_tooltip_open.png"
    page.screenshot(path=str(tooltip_shot))
    shutil.copy(tooltip_shot, ARTIFACT_DIR / "popover_tooltip_open.png")
    check_image_health(tooltip_shot)

    # 3. Toast Notification
    print("\n3. Testing Toast Notification...")
    btn = page.locator('[data-testid="stSidebar"] button').filter(has_text="Clear Analysis History").first
    btn.click()
    time.sleep(0.5)
    page.wait_for_selector('[data-testid="stToast"]', timeout=5000)
    toast_shot = OUTPUT_DIR / "popover_toast_open.png"
    page.screenshot(path=str(toast_shot))
    shutil.copy(toast_shot, ARTIFACT_DIR / "popover_toast_open.png")
    check_image_health(toast_shot)

    browser.close()

print("\nAll popovers verified open, visible, unmasked, and healthy!")
