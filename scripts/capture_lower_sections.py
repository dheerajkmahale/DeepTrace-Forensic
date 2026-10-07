from playwright.sync_api import sync_playwright
import time
import os

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots"))

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.goto('http://localhost:8501')
    page.wait_for_selector('div[data-testid="stRadio"]', timeout=30000)
    time.sleep(3)

    # Click fake preset
    labels = page.locator('div[data-testid="stRadio"] label').all()
    for l in labels:
        if "Celeb-synthesis" in l.inner_text():
            l.click()
            break
    time.sleep(3)

    btn = page.locator('button:has-text("Analyze Video")')
    for _ in range(30):
        if btn.first.is_enabled():
            break
        time.sleep(0.5)

    btn.first.click()
    page.wait_for_selector('.result-hero-card', timeout=90000)
    time.sleep(4)

    # Expand Technical Details if present
    details = page.locator('details:has-text("Technical Details")')
    if details.count() > 0:
        details.first.click()
        time.sleep(1)

    # Scroll down to reveal summary and details
    page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
    time.sleep(2)

    footer_png = os.path.join(ARTIFACT_DIR, "deeptrace_lower_sections.png")
    page.screenshot(path=footer_png)
    print("Lower sections screenshot saved to:", footer_png)

    # Print text in lower section
    summary_text = page.locator('.summary-block-grid').inner_text() if page.locator('.summary-block-grid').count() > 0 else "No summary grid"
    print("SUMMARY GRID:\n", summary_text.encode('ascii', errors='replace').decode('ascii'))
    b.close()
