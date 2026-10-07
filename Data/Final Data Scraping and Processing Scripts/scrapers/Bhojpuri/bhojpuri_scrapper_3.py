"""
BHOJPURI SCRAPER v3 - anjoria.com

FIXES vs v2:
  [CRIT] Thread-safety        - each worker gets its OWN Selenium driver
  [CRIT] URL mismatch         - permalink from entry-title/rel=bookmark, not first <a>
  [CRIT] Redirect URLs        - article_url = driver.current_url (real landed URL)
  [CRIT] Dead/expired pages   - 404 and redirect-to-home detected and skipped
  [CRIT] Pagination loop      - detects redirect-to-homepage on page 2+, stops after
                                3 consecutive pages with 0 new URLs
  [HIGH] # split bug removed  - no more content loss before hashtags
  [HIGH] # regex fixed        - handles "# word" (space after #) tokens too
  [HIGH] Length check         - moved AFTER clean_text so count is accurate
  [MED]  is_caption_like      - loosened, only drops zero-Devanagari short lines
  [MED]  Retry logic          - 3 retries with exponential back-off per article
  [LOW]  wait_for_content     - waits for ANY of the 5 content selectors
  [LOW]  lock_data removed    - unused variable deleted
  [ADD]  Deduplication        - identical texts written only once
  [ADD]  Boilerplate strip    - removes leaked nav/footer text
  [ADD]  Checkpoint saving    - saves to CSV every 50 articles (crash-safe)

Usage:
  python bhojpuri_scrapper_3.py --site https://anjoria.com --demo 100 --workers 8
  python bhojpuri_scrapper_3.py --site https://anjoria.com --workers 4
"""

import re
import time
import random
import argparse
import os
import threading
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

MIN_ARTICLE_LENGTH = 300
CHECKPOINT_EVERY   = 50
MAX_RETRIES        = 3
RETRY_BASE_DELAY   = 2
MAX_EMPTY_PAGES    = 3

CONTENT_SELECTORS = [
    ("class", "et_pb_post_content"),
    ("class", "td-post-content"),
    ("class", "entry-content"),
    ("class", "post-content"),
    ("class", "article-content"),
]

BOILERPLATE_RE = re.compile(
    r'(share\s+this|follow\s+us|subscribe|click\s+here'
    r'|facebook|instagram|twitter|youtube|whatsapp|telegram'
    r'|download\s+app|read\s+more|also\s+read|related\s+post'
    r'|tags?\s*:|posted\s+by|leave\s+a\s+comment|copyright'
    r'|all\s+rights\s+reserved|privacy\s+policy|terms\s+of\s+use)',
    re.IGNORECASE
)

_thread_local = threading.local()


def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Devanagari (0.0 – 1.0)."""
    if not text:
        return 0.0
    deva  = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    alpha = sum(1 for c in text if c.isalpha())
    return deva / alpha if alpha else 0.0


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--blink-settings=imagesEnabled=false")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )


def get_thread_driver():
    if not hasattr(_thread_local, "driver") or _thread_local.driver is None:
        _thread_local.driver = get_driver()
    return _thread_local.driver


def quit_thread_driver():
    if hasattr(_thread_local, "driver") and _thread_local.driver:
        try:
            _thread_local.driver.quit()
        except Exception:
            pass
        _thread_local.driver = None


def wait_for_content(driver, timeout=10):
    css = ", ".join(
        f".{sel[1]}" if sel[0] == "class" else f"#{sel[1]}"
        for sel in CONTENT_SELECTORS
    )
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
    except Exception:
        pass


def wait_for_page(driver, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
    except Exception:
        pass


def clean_text(text):
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'#\s*', '', text)
    lines = text.split('.')
    lines = [l for l in lines if not BOILERPLATE_RE.search(l)]
    text = '.'.join(lines)
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def is_caption_like(text):
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if deva >= 3:
        return False
    if len(text) < 80 and deva == 0:
        return True
    return False


def strip_boilerplate_paragraphs(paragraphs):
    return [p for p in paragraphs if not (BOILERPLATE_RE.search(p) and len(p) < 200)]


def is_dead_page(driver, base_url):
    current = driver.current_url.rstrip("/")
    base    = base_url.rstrip("/")
    if current == base or re.search(r"/page/\d+/?$", current):
        return True
    try:
        src = driver.page_source[:3000].lower()
        if ('class="error404"' in src or "<title>page not found" in src
                or "<title>404" in src):
            return True
    except Exception:
        pass
    return False


def extract_article(url, base_url="https://anjoria.com"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            driver = get_thread_driver()
            driver.get(url)
            wait_for_content(driver)
            time.sleep(random.uniform(0.4, 1.0))

            final_url = driver.current_url

            if is_dead_page(driver, base_url):
                print(f"  dead/expired: {url}")
                return None

            soup    = BeautifulSoup(driver.page_source, "lxml")
            content = None
            for attr, val in CONTENT_SELECTORS:
                content = soup.find("div", {attr: val})
                if content:
                    break
            if not content:
                return None

            for tag in content.find_all(
                ["div", "section", "aside"],
                class_=re.compile(
                    r"related|sharedaddy|jp-relatedposts|post-related|yarpp|wpcnt"
                    r"|share|social|subscribe|newsletter|comment"
                )
            ):
                tag.decompose()

            for tag in content.find_all(
                ["figure", "div"],
                class_=re.compile(
                    r"wp-caption|wp-block-image|et_pb_image"
                    r"|aligncenter|alignleft|alignright|gallery"
                )
            ):
                tag.decompose()
            for tag in content.find_all("figcaption"):
                tag.decompose()

            raw_paragraphs = []
            for p in content.find_all("p"):
                p_text = p.get_text().strip()
                if not p_text:
                    continue
                if re.search(
                    r'^In\s+["\u201c\u201d][^""\u201c\u201d]{0,60}["\u201c\u201d]\s*$',
                    p_text
                ):
                    continue
                if is_caption_like(p_text):
                    continue
                raw_paragraphs.append(p_text)

            raw_paragraphs = strip_boilerplate_paragraphs(raw_paragraphs)
            if not raw_paragraphs:
                return None

            text = clean_text(" ".join(raw_paragraphs))
            if len(text) < MIN_ARTICLE_LENGTH:
                return None

            return text, final_url

        except Exception:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"  retry {attempt}/{MAX_RETRIES} for {url} in {delay:.1f}s")
                time.sleep(delay)
                try:
                    quit_thread_driver()
                except Exception:
                    pass
            else:
                print(f"  failed after {MAX_RETRIES} attempts: {url}")
                return None
    return None


def extract_permalink(article_tag):
    # 1. rel=bookmark — canonical WordPress permalink marker
    a = article_tag.find("a", rel=lambda r: r and "bookmark" in r)
    if a and a.get("href"):
        return a["href"]

    # 2. Title heading link
    for cls in ("entry-title", "post-title", "article-title"):
        heading = article_tag.find(
            re.compile(r"^h[123456]$"),
            class_=re.compile(cls, re.IGNORECASE)
        )
        if heading:
            a = heading.find("a", href=True)
            if a:
                return a["href"]

    # 3. Divi / GeneratePress title link class
    a = article_tag.find(
        "a", class_=re.compile(r"entry-title-link|post-title-link", re.IGNORECASE)
    )
    if a and a.get("href"):
        return a["href"]

    # 4. First slug URL that is not a taxonomy/author link
    for a in article_tag.find_all("a", href=True):
        href = a["href"]
        if re.search(r"/(category|tag|author|page)/", href):
            continue
        if href.startswith("https://anjoria.com/") and len(href.split("/")) >= 4:
            return href

    return None


def get_urls_from_sitemap(base_url, max_articles=None):
    """
    Pull all article URLs from the WordPress XML sitemap — no browser needed.
    Tries multiple sitemap formats in order.
    """
    from xml.etree import ElementTree as ET

    base_clean = base_url.rstrip("/")
    headers    = {"User-Agent": "Mozilla/5.0 (compatible; SitemapBot/1.0)"}

    # Both namespaced and un-namespaced tag lookups
    SM_NS  = "http://www.sitemaps.org/schemas/sitemap/0.9"

    def find_text(el, tag):
        """
        FIX: ElementTree elements are NEVER falsy (even empty ones),
        so `el.find(x) or el.find(y)` always returns the first result
        regardless of whether it found anything.
        Correct approach: explicit `is not None` checks.
        """
        # Try namespaced first
        found = el.find(f"{{{SM_NS}}}{tag}")
        if found is not None and found.text:
            return found.text.strip()
        # Try un-namespaced
        found = el.find(tag)
        if found is not None and found.text:
            return found.text.strip()
        return None

    def fetch_xml(url):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                return None
            # FIX: don't gate on Content-Type — many servers return
            # text/html or application/octet-stream for XML sitemaps
            content = r.content.strip()
            if not content.startswith(b"<?xml") and not content.startswith(b"<"):
                return None
            return ET.fromstring(content)
        except Exception as e:
            print(f"    [sitemap] error fetching {url}: {e}")
        return None

    def is_post_sitemap(url):
        return bool(re.search(
            r"(post|article|news|bhojpuri|wp-sitemap-posts-post)",
            url, re.IGNORECASE
        ))

    def locs_from_urlset(url):
        """Yield all <loc> values from a single urlset sitemap."""
        root = fetch_xml(url)
        if root is None:
            return
        # Handle both namespaced <url> and plain <url>
        url_els = root.findall(f"{{{SM_NS}}}url") or root.findall("url")
        for el in url_els:
            loc = find_text(el, "loc")
            if loc:
                yield loc

    article_urls = set()

    candidates = [
        f"{base_clean}/wp-sitemap.xml",
        f"{base_clean}/sitemap.xml",
        f"{base_clean}/sitemap_index.xml",
        f"{base_clean}/post-sitemap.xml",
        f"{base_clean}/news-sitemap.xml",
    ]

    for candidate in candidates:
        print(f"  [sitemap] trying {candidate}")
        root = fetch_xml(candidate)
        if root is None:
            print(f"    → not found or invalid XML")
            continue

        tag = root.tag  # e.g. "{http://...}sitemapindex" or "sitemapindex"

        if "sitemapindex" in tag:
            # Collect all sub-sitemap <loc> values
            sm_els = root.findall(f"{{{SM_NS}}}sitemap") or root.findall("sitemap")
            sub_locs = []
            for el in sm_els:
                loc = find_text(el, "loc")
                if loc:
                    sub_locs.append(loc)

            print(f"    → index with {len(sub_locs)} sub-sitemaps")

            # Prefer post-specific sub-sitemaps; scan all if none match
            targets = [u for u in sub_locs if is_post_sitemap(u)] or sub_locs
            print(f"    → scanning {len(targets)} post/article sitemaps")

            for sub in targets:
                print(f"    → {sub}")
                for loc in locs_from_urlset(sub):
                    if loc.startswith(base_clean + "/"):
                        article_urls.add(loc)
                        if max_articles and len(article_urls) >= max_articles:
                            break
                if max_articles and len(article_urls) >= max_articles:
                    break

        elif "urlset" in tag:
            print(f"    → direct urlset sitemap")
            for loc in locs_from_urlset(candidate):
                if loc.startswith(base_clean + "/"):
                    article_urls.add(loc)
                    if max_articles and len(article_urls) >= max_articles:
                        break

        else:
            print(f"    → unrecognised root tag: {tag}")

        if article_urls:
            print(f"  [sitemap] ✅ {len(article_urls):,} URLs collected from {candidate}")
            break

    return article_urls


def get_article_urls(base_url, max_articles=None):
    """
    Collect article URLs: sitemap first (fast, no browser),
    falling back to Selenium pagination if sitemap yields nothing.
    """
    base_clean = base_url.rstrip("/")

    # ── Strategy 1: XML Sitemap (preferred) ────────────────────────────────
    print(f"\n  [URL collection] Trying XML sitemaps first...")
    article_urls = get_urls_from_sitemap(base_url, max_articles)
    if article_urls:
        urls = list(article_urls)
        if max_articles:
            urls = urls[:max_articles]
        print(f"  [URL collection] Sitemap yielded {len(urls):,} URLs.\n")
        return urls

    # ── Strategy 2: Selenium pagination fallback ────────────────────────────
    print(f"  [URL collection] Sitemap empty — falling back to Selenium pagination...\n")
    article_urls = set()
    page         = 1
    empty_streak = 0
    driver       = get_driver()

    try:
        while True:
            if max_articles and len(article_urls) >= max_articles:
                break

            target = base_clean if page == 1 else f"{base_clean}/page/{page}/"
            print(f"  Listing page {page}: {target}")
            driver.get(target)
            wait_for_page(driver)

            landed = driver.current_url.rstrip("/")
            # Detect infinite-scroll loop: page 2+ resolves back to page 1 / homepage
            if page > 1 and (landed == base_clean or re.search(r"/page/1/?$", landed)):
                print(f"  Page {page} loops back to homepage — site uses infinite scroll. Stopping.")
                break
            m = re.search(r"/page/(\d+)/?$", landed)
            if m and int(m.group(1)) < page:
                print(f"  Redirect loop detected (landed page {m.group(1)} < requested {page}). Stopping.")
                break

            soup     = BeautifulSoup(driver.page_source, "lxml")
            articles = soup.find_all("article")
            if not articles:
                print(f"  No <article> tags on page {page}. Stopping.")
                break

            found_this_page = 0
            for article in articles:
                link = extract_permalink(article)
                if not link:
                    continue
                if link.startswith(base_clean + "/") and link not in article_urls:
                    article_urls.add(link)
                    found_this_page += 1
                if max_articles and len(article_urls) >= max_articles:
                    break

            print(f"     -> {found_this_page} new URLs (total: {len(article_urls)})")

            if found_this_page == 0:
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_PAGES:
                    print(f"  {MAX_EMPTY_PAGES} empty pages in a row — infinite scroll site. Stopping.")
                    break
            else:
                empty_streak = 0

            page += 1

    finally:
        driver.quit()

    print(f"\n  Total URLs collected: {len(article_urls)}")
    return list(article_urls)


def save_checkpoint(data, output_path, written_count):
    new_records = data[written_count:]
    if not new_records:
        return written_count
    df_chunk = pd.DataFrame(new_records)
    write_header = not os.path.exists(output_path)
    df_chunk.to_csv(output_path, mode='a', index=False,
                    header=write_header, encoding="utf-8-sig")
    written_count += len(new_records)
    print(f"  [checkpoint] {written_count} articles saved")
    return written_count


def scrape_site(base_url, max_articles=None, num_workers=4, bhojpuri_only=False):
    # Minimum Devanagari ratio to keep an article when --bhojpuri-only is set
    DEVA_THRESHOLD = 0.40

    print(f"\n{'='*64}")
    print(f"  BHOJPURI SCRAPER v3  --  {base_url}")
    print(f"  Workers: {num_workers}  |  Max articles: {max_articles or 'all'}")
    if bhojpuri_only:
        print(f"  --bhojpuri-only ON  (dropping articles with < {DEVA_THRESHOLD:.0%} Devanagari)")
    print(f"{'='*64}\n")

    article_urls = get_article_urls(base_url, max_articles)

    os.makedirs("output", exist_ok=True)
    output_path = "output/scraped_articles_bhojpuri.csv"
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"\n  Starting parallel extraction with {num_workers} workers...\n")

    data              = []
    written_count     = 0
    seen_fingerprints = set()
    success           = 0
    skipped_dup       = 0
    skipped_english   = 0

    def worker(args):
        idx, url = args
        print(f"[{idx+1:>4}] {url}")
        result = extract_article(url, base_url)
        if result is None:
            return None
        text, final_url = result
        return {
            "website":      urlparse(base_url).netloc,
            "article_url":  final_url,
            "language":     "bhojpuri",
            "article_text": text
        }

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker, (i, url)): url
                for i, url in enumerate(article_urls)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # ── Bhojpuri-only filter ───────────────────────────────
                    if bhojpuri_only:
                        ratio = devanagari_ratio(result["article_text"])
                        if ratio < DEVA_THRESHOLD:
                            skipped_english += 1
                            print(f"  [bhojpuri-only] dropped ({ratio:.0%} Devanagari): "
                                  f"{result['article_url']}")
                            continue

                    # ── Deduplication ──────────────────────────────────────
                    fp = re.sub(r'\s+', ' ',
                                result["article_text"][:200].strip().lower())
                    if fp in seen_fingerprints:
                        skipped_dup += 1
                        continue
                    seen_fingerprints.add(fp)
                    data.append(result)
                    success += 1
                    if len(data) - written_count >= CHECKPOINT_EVERY:
                        written_count = save_checkpoint(data, output_path, written_count)
    finally:
        written_count = save_checkpoint(data, output_path, written_count)

    total = len(article_urls)
    print(f"\n{'='*64}")
    print(f"  COMPLETED")
    print(f"{'='*64}")
    print(f"  URLs attempted       : {total:,}")
    print(f"  Articles extracted   : {success:,}")
    print(f"  Duplicates skipped   : {skipped_dup:,}")
    if bhojpuri_only:
        print(f"  English dropped      : {skipped_english:,}  (--bhojpuri-only)")
    print(f"  Failed / empty       : {total - success - skipped_dup - skipped_english:,}")
    print(f"  Success rate         : {100 * success / max(total, 1):.1f}%")
    print(f"  Output               : {output_path}")
    print(f"{'='*64}\n")

    return data


def main():
    parser = argparse.ArgumentParser(description="Bhojpuri Scraper v3 for anjoria.com")
    parser.add_argument("--site",    type=str, required=True,
                        help="Base URL (e.g. https://anjoria.com)")
    parser.add_argument("--demo",    type=int, default=None,
                        help="Limit articles for a test run")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--bhojpuri-only", action="store_true", default=False,
                        help="Drop articles with < 40%% Devanagari script (filters out English articles)")
    args = parser.parse_args()
    scrape_site(args.site, max_articles=args.demo, num_workers=args.workers,
                bhojpuri_only=args.bhojpuri_only)


if __name__ == "__main__":
    main()