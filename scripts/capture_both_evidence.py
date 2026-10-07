from playwright.sync_api import sync_playwright
import time
import os

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots"))

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    # Give plenty of viewport height to capture everything
    page = b.new_page(viewport={"width": 1400, "height": 2200})
    page.goto('http://localhost:8501')
    page.wait_for_selector('text=DEEPTRACE', timeout=20000)
    time.sleep(2)

    # Disable Center-crop fallback toggle in sidebar so real face detection runs!
    # Let's check sidebar toggle
    toggle = page.locator('div[data-testid="stCheckbox"]:has-text("Center-crop"), div[data-testid="stToggle"]:has-text("Center-crop")')
    if toggle.count() > 0:
        print("Turning off Center-crop fallback toggle so Haar face detection runs...")
        toggle.first.click()
        time.sleep(2)

    # Select authentic fake preset
    radio_labels = page.locator('div[data-testid="stRadio"] label').all()
    for r in radio_labels:
        if "Celeb-synthesis" in r.inner_text():
            r.click()
            print("Selected Celeb-synthesis preset")
            break
    time.sleep(2)

    # Click Analyze Video
    btn = page.locator('button:has-text("Analyze Video")')
    btn.first.click()
    print("Clicked Analyze Video. Waiting for analysis...")

    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(5)

    # Screenshot full page
    full_png = os.path.join(ARTIFACT_DIR, "deeptrace_fake_full_evidence.png")
    page.screenshot(path=full_png, full_page=True)
    print("Full fake evidence screenshot saved to:", full_png)

    # Now let's test authentic real video
    print("Selecting Authentic Real preset...")
    radio_labels = page.locator('div[data-testid="stRadio"] label').all()
    for r in radio_labels:
        if "Celeb-real" in r.inner_text():
            r.click()
            print("Selected Celeb-real preset")
            break
    time.sleep(2)

    btn = page.locator('button:has-text("Analyze Video")')
    btn.first.click()
    print("Clicked Analyze Video for Real video. Waiting for analysis...")

    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(5)

    real_png = os.path.join(ARTIFACT_DIR, "deeptrace_real_full_evidence.png")
    page.screenshot(path=real_png, full_page=True)
    print("Full real evidence screenshot saved to:", real_png)

    b.close()
    print("All captures completed successfully!")
