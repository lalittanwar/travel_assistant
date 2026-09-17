"""
Scrapes the URLs listed in sources.json and saves cleaned article text as
.md files into knowledge_base/content/.

Strategy per URL:
  1. Try a fast plain-HTTP fetch (requests + BeautifulSoup).
  2. If the extracted text is too short (likely a JS-rendered page like
     visitsingapore.com), fall back to a real headless browser (Playwright)
     that waits for JavaScript to render before extracting text.

Setup:
    pip install requests beautifulsoup4 playwright
    playwright install chromium

Run:
    python scrape_sources.py
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent
SOURCES_FILE = BASE_DIR / "sources.json"
CONTENT_DIR = BASE_DIR / "content"

MIN_CHARS = 300  # below this, treat the fetch as "too thin" and try the browser fallback

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

TAGS_TO_STRIP = ["script", "style", "nav", "footer", "header", "form", "noscript", "iframe"]
CLASS_HINTS_TO_STRIP = [
    "navbox", "vector-header", "mw-editsection", "printfooter", "catlinks",
    "mw-references-wrap", "reflist", "footer", "site-footer", "cookie", "nav-",
    "breadcrumb", "sidebar", "advert", "newsletter", "mw-jump-link",
]


def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(TAGS_TO_STRIP):
        tag.decompose()

    for hint in CLASS_HINTS_TO_STRIP:
        for el in soup.find_all(class_=re.compile(hint, re.I)):
            el.decompose()

    container = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", id="mw-content-text")
        or soup.body
    )
    if container is None:
        return ""

    text = container.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n\n".join(lines)


def fetch_with_requests(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return clean_html_text(resp.text)


def fetch_with_playwright(url: str) -> str:
    """Headless-browser fallback for JS-rendered pages."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, timeout=30000, wait_until="networkidle")
        # Give client-side rendering a little extra time on slower pages
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()
    return clean_html_text(html)


def scrape_one(entry: dict) -> None:
    url = entry["url"]
    title = entry["title"]
    out_path = CONTENT_DIR / entry["file"]

    print(f"Fetching: {title} ({url})")

    body_text = ""
    method_used = "requests"
    try:
        body_text = fetch_with_requests(url)
    except requests.RequestException as e:
        print(f"  requests fetch failed: {e}")

    if len(body_text) < MIN_CHARS:
        print(f"  Plain fetch too thin ({len(body_text)} chars) — trying headless browser...")
        try:
            body_text = fetch_with_playwright(url)
            method_used = "playwright"
        except Exception as e:
            print(f"  Playwright fetch also failed: {e}")
            print("  -> Paste this page's content manually (see instructions.md).")

    if len(body_text) < MIN_CHARS:
        print(f"  WARNING: still short after fallback ({len(body_text)} chars).")
        print("  -> Paste this page's content manually (see instructions.md).")
    else:
        print(f"  OK via {method_used} ({len(body_text)} chars)")

    header = (
        f"TITLE: {title}\n"
        f"URL: {url}\n"
        f"RETRIEVED: {time.strftime('%Y-%m-%d')}\n"
        f"---\n\n"
    )

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body_text, encoding="utf-8")
    print(f"  Saved -> {out_path}")


def main():
    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    for entry in sources:
        scrape_one(entry)
        time.sleep(1.5)  # be polite between requests

    print("\nDone. Review each file in knowledge_base/content/ before running ingest.py.")
    print("Any file still under a few hundred characters needs a manual paste —")
    print("the header block (TITLE/URL/RETRIEVED) is already written for you.")


if __name__ == "__main__":
    main()
