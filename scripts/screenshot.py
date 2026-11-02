"""Take a screenshot of the running Streamlit app (used by GitHub Actions)."""

import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/app_screenshot.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url)
    # Wait until the matches have been calculated and drawn
    page.get_by_text("Your breakout room").wait_for(timeout=300_000)
    page.wait_for_timeout(3000)
    page.screenshot(path=out, full_page=True)
    browser.close()
print("Saved", out)
