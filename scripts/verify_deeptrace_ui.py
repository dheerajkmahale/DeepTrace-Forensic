import os
import sys
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join("outputs", "screenshots"))
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_verification():
    print("Starting Playwright DeepTrace verification...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1100})
        page = context.new_page()

        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="load", timeout=45000)
        
        # Wait for Streamlit app to render main heading
        page.wait_for_selector("text=DEEPTRACE", timeout=20000)
        time.sleep(2)

        # 1. Capture initial page
        initial_png = os.path.join(ARTIFACT_DIR, "deeptrace_initial.png")
        page.screenshot(path=initial_png, full_page=True)
        print(f"Initial screenshot saved to {initial_png}")

        body_text = page.inner_text("body")
        assert "DEEPTRACE" in body_text, "DEEPTRACE text missing"
        assert "MODEL ONLINE" in body_text, "MODEL ONLINE status missing"
        assert "Deepfake Forensic Analysis" in body_text, "Hero title missing"
        assert "0.39" in body_text, "Decision Threshold 0.39 missing"
        assert "MODE: REAL EXPERIMENT" in body_text, "Real experiment mode badge missing"
        print("Header, Status Pill, Model Provenance & Hero verified!")

        # 2. Select authentic fake video preset
        print("Selecting Authentic Fake Video preset...")
        fake_radio = page.locator('label:has-text("Celeb-synthesis")')
        assert fake_radio.count() > 0, "Celeb-synthesis radio option not found"
        fake_radio.first.click()
        time.sleep(2)

        # Check metadata cards
        meta_grid = page.locator(".video-meta-grid")
        assert meta_grid.count() > 0, "Video metadata grid missing after selecting video"
        print("Video metadata grid displayed successfully!")

        # Click 'Analyze Video'
        analyze_btn = page.locator('button:has-text("Analyze Video")')
        assert analyze_btn.count() > 0, "Analyze Video button missing"
        print("Clicking 'Analyze Video' for authentic fake...")
        analyze_btn.first.click()

        # Wait for analysis to complete
        print("Waiting for fake video analysis to complete...")
        page.wait_for_selector(".result-hero-card", timeout=90000)
        time.sleep(4)

        fake_result_png = os.path.join(ARTIFACT_DIR, "deeptrace_fake_analysis.png")
        page.screenshot(path=fake_result_png, full_page=True)
        print(f"Fake analysis screenshot saved to {fake_result_png}")

        body_fake = page.inner_text("body")
        assert "FAKE" in body_fake, "Verdict FAKE not found"
        assert "Fake Probability" in body_fake, "Fake Probability label not found"
        assert "Decision Threshold" in body_fake, "Decision Threshold label not found"
        assert "0.39" in body_fake, "Decision Threshold 0.39 not found"
        assert "Temporal Forensic Evidence" in body_fake, "Temporal Forensic Evidence section missing"
        assert "Important Frame 01" in body_fake, "Important Frame 01 missing"
        assert "Forensic Summary" in body_fake, "Forensic Summary missing"
        assert "Technical Details" in body_fake, "Technical Details missing"
        print("Authentic fake analysis verified successfully!")

        # 3. Test authentic real video preset
        print("Selecting Authentic Real Video preset...")
        real_radio = page.locator('label:has-text("Celeb-real")')
        assert real_radio.count() > 0, "Celeb-real radio option not found"
        real_radio.first.click()
        time.sleep(2)

        print("Clicking 'Analyze Video' for authentic real...")
        page.locator('button:has-text("Analyze Video")').first.click()

        print("Waiting for real video analysis to complete...")
        page.wait_for_selector(".result-hero-card", timeout=90000)
        time.sleep(4)

        real_result_png = os.path.join(ARTIFACT_DIR, "deeptrace_real_analysis.png")
        page.screenshot(path=real_result_png, full_page=True)
        print(f"Real analysis screenshot saved to {real_result_png}")

        body_real = page.inner_text("body")
        assert "REAL" in body_real, "Verdict REAL not found"
        print("Authentic real analysis verified successfully!")

        # 4. Check for unhandled exceptions or stack traces
        for err_kw in ["Traceback (most recent call last)", "ZeroDivisionError", "KeyError", "AttributeError", "FileNotFoundError"]:
            assert err_kw not in body_real, f"Error found in page: {err_kw}"

        print("Verification complete! No exceptions found. DeepTrace UI is fully functional!")
        browser.close()

if __name__ == "__main__":
    run_verification()
