from playwright.sync_api import sync_playwright
import time
import os

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots"))

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1400, "height": 2600})
    page.goto('http://localhost:8501')
    page.wait_for_selector('div[data-testid="stRadio"]', timeout=30000)
    time.sleep(3)

    # 1. Select Celeb-synthesis fake preset
    labels = page.locator('div[data-testid="stRadio"] label').all()
    print("Labels found:", len(labels))
    clicked_fake = False
    for l in labels:
        txt = l.inner_text()
        if "Celeb-synthesis" in txt:
            l.click()
            clicked_fake = True
            print("Clicked:", txt)
            break
    assert clicked_fake, "Could not find Celeb-synthesis option!"

    # Wait for Analyze Video button to be enabled
    time.sleep(3)
    btn = page.locator('button:has-text("Analyze Video")')
    for _ in range(30):
        if btn.first.is_enabled():
            break
        time.sleep(0.5)
    print("Analyze Video button is enabled! Clicking...")
    btn.first.click()

    # Wait for completion
    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(5)

    fake_png = os.path.join(ARTIFACT_DIR, "deeptrace_verified_fake.png")
    page.screenshot(path=fake_png, full_page=True)
    print("Verified fake screenshot saved to:", fake_png)

    # Print verdict text
    hero = page.locator('.result-hero-card')
    print("FAKE RESULT HERO:")
    print(hero.inner_text().encode('ascii', errors='replace').decode('ascii'))

    # 2. Select Celeb-real preset
    labels = page.locator('div[data-testid="stRadio"] label').all()
    clicked_real = False
    for l in labels:
        txt = l.inner_text()
        if "Celeb-real" in txt:
            l.click()
            clicked_real = True
            print("Clicked:", txt)
            break
    assert clicked_real, "Could not find Celeb-real option!"

    time.sleep(3)
    for _ in range(30):
        if btn.first.is_enabled():
            break
        time.sleep(0.5)
    print("Analyze Video button is enabled for real video! Clicking...")
    btn.first.click()

    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(5)

    real_png = os.path.join(ARTIFACT_DIR, "deeptrace_verified_real.png")
    page.screenshot(path=real_png, full_page=True)
    print("Verified real screenshot saved to:", real_png)

    hero = page.locator('.result-hero-card')
    print("REAL RESULT HERO:")
    print(hero.inner_text().encode('ascii', errors='replace').decode('ascii'))

    b.close()
    print("All tests passed cleanly!")
