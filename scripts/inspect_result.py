from playwright.sync_api import sync_playwright
import time
import os

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots"))

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1400, "height": 1400})
    page.goto('http://localhost:8501')
    page.wait_for_selector('text=DEEPTRACE', timeout=20000)
    time.sleep(3)

    # Find radio labels
    radio_labels = page.locator('div[data-testid="stRadio"] label').all()
    print("Found radio labels count:", len(radio_labels))
    for r in radio_labels:
        print("  Label:", r.inner_text())

    # Click authentic fake preset
    clicked = False
    for r in radio_labels:
        if "Celeb-synthesis" in r.inner_text():
            r.click()
            clicked = True
            print("Successfully clicked Celeb-synthesis preset!")
            break

    assert clicked, "Failed to click Celeb-synthesis preset"
    time.sleep(3)

    # Verify Analyze Video button is now enabled
    btn = page.locator('button:has-text("Analyze Video")')
    assert btn.count() > 0
    print("Analyze Video button is present. Enabled status:", btn.first.is_enabled())
    btn.first.click()
    print("Clicked Analyze Video. Waiting for forensic analysis...")

    # Wait for completion
    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(4)

    # Capture screenshots of full page and result hero
    full_png = os.path.join(ARTIFACT_DIR, "deeptrace_full_analysis.png")
    page.screenshot(path=full_png, full_page=True)
    print("Full page screenshot saved to:", full_png)

    # Print Result Hero text
    print("\n=== RESULT HERO TEXT ===")
    print(page.locator('.result-hero-card').inner_text())

    # Print Temporal Forensic Evidence text if available
    print("\n=== EVIDENCE SECTION TEXT ===")
    cards = page.locator('.frame-evidence-card').all()
    print("Found frame evidence cards:", len(cards))
    for i, c in enumerate(cards):
        print(f"Card {i+1}:", c.inner_text())

    # Print Forensic Summary text
    print("\n=== FORENSIC SUMMARY TEXT ===")
    summary_grid = page.locator('.summary-block-grid')
    if summary_grid.count() > 0:
        print(summary_grid.first.inner_text())

    b.close()
    print("Script finished successfully!")
