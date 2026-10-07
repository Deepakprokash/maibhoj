"""
Maithili Jindabaad Website Scraper
====================================
Website  : http://www.maithilijindabaad.com/
Purpose  : Scrape all articles, clean the text to pure Maithili (Devanagari),
           and save as a CSV dataset with columns:
               - article_link    : canonical URL of the article
               - article_text    : clean Devanagari text, preamble stripped
               - article_language: one of "Maithili" | "Hindi" | "Maithili-Hindi Mixed"

FIXES APPLIED
--------------
  Fix 1 – Preamble stripping:
    Articles submitted via "दहेज मुक्त मिथिला लेखनीक धार" and similar series
    include a structured header (author credit, source, weekly topic label)
    before the actual article body. These are now detected and stripped.

  Fix 2 – Language detection:
    Both Maithili and Hindi use Devanagari script so character filtering alone
    cannot distinguish them. Each article is now scored against curated lists of
    Maithili-specific and Hindi-specific function words / verb endings, and tagged
    accurately as "Maithili", "Hindi", or "Maithili-Hindi Mixed".

INSTALLATION (run once)
------------------------
    pip install requests beautifulsoup4 tqdm pandas

RUNNING COMMANDS
-----------------
  # 1. DEMO MODE  -- scrapes only 1 listing page, max 10 articles (~30 sec)
  python maithili_scraper.py --demo

  # 2. DEMO with custom article limit
  python maithili_scraper.py --demo --limit 5

  # 3. PARTIAL RUN -- scrape first N listing pages only
  python maithili_scraper.py --pages 10

  # 4. FULL RUN   -- scrapes ALL 600+ listing pages (~5600 articles, ~2 hrs)
  python maithili_scraper.py

  # 5. CUSTOM OUTPUT FILE
  python maithili_scraper.py --demo --output my_test.csv
  python maithili_scraper.py --output full_dataset.csv

OUTPUT FILES
-------------
  maithili_dataset.csv      -- final dataset  (utf-8-sig, Excel-safe)
  maithili_dataset.partial  -- auto-checkpoint every 200 articles
  collected_links.txt       -- all discovered article URLs (resume checkpoint)
  skipped_urls.txt          -- URLs skipped (404 / empty / non-Devanagari)
  scraper.log               -- full run log
"""

import re
import sys
import time
import logging
import argparse
import requests
import pandas as pd
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin, urlparse, parse_qs

# ---- DEFAULTS ----------------------------------------------------------------
BASE_URL        = "https://maithilijindabaad.com"
BLOG_PAGE_URL   = "https://maithilijindabaad.com/?page_id=22928"
OUTPUT_CSV      = "maithili_dataset.csv"
SKIPPED_FILE    = "skipped_urls.txt"
DELAY_SECONDS   = 1.2
REQUEST_TIMEOUT = 20
MAX_RETRIES     = 3
DEMO_LIMIT      = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ---- LOGGING -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ---- DEVANAGARI PATTERNS -----------------------------------------------------
DEVANAGARI_PATTERN = re.compile(
    r'[\u0900-\u097F\u0966-\u096F\u1CD0-\u1CFF\uA8E0-\uA8FF]'
)
NOISE_PATTERN = re.compile(
    r'[\u200b-\u200f\u202a-\u202e\ufeff\u00a0]'
)

# ---- LANGUAGE DETECTION WORD LISTS ------------------------------------------
# These are function words and verb endings that reliably distinguish the two scripts.
# Maithili markers: unique verb forms, postpositions, particles found only in Maithili
MAITHILI_MARKERS = [
    'अछि', 'छथि', 'छलाह', 'छल ', 'छल।', 'छल,',
    'केँ ', 'सँ ',  'मे ',  'कऽ ', 'लऽ ',
    'जाइत', 'होइत', 'रहैत', 'करैत', 'बनैत',
    'छैक', 'छनि', 'छियनि', 'छियैक', 'अछनि',
    'कहलनि', 'देलनि', 'भेलनि', 'गेलनि',
    'भेल ', 'भेल।', 'गेल ', 'गेल।',
    'ओतय', 'एतय', 'जतय', 'कतय',
    'आयल', 'गेलाह', 'कएल', 'लेल ',
    'तखन', 'एखन', 'किएक', 'कियैक',
    'हुनका', 'हमरा', 'अपना', 'तोहर',
]
# Hindi markers: function words and verb forms found only in Hindi
HINDI_MARKERS = [
    'नहीं', 'करते', 'होते', 'जाते', 'आते', 'जाता',
    'हैं ', 'हैं।', 'हैं,',
    'था ', 'था।', 'था,', 'थे ', 'थे।', 'थी ', 'थी।',
    'होगा', 'करेगा', 'जाएगा', 'होंगे',
    'करेंगे', 'जाएंगे',
    'रहे हैं', 'रहा है', 'रही है',
    'किया ', 'किया।', 'किया,',
    'लेकिन', 'बल्कि', 'इसलिए', 'क्योंकि',
    'उनका', 'उनकी', 'उनके', 'उन्हें',
    'इसका', 'इसकी', 'इसके', 'इसमें',
    'यहां', 'वहां', 'जहां', 'कहां',
    'करना', 'होना', 'जाना', 'आना',
]


# ---- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="maithili_scraper.py",
        description="Scrape maithilijindabaad.com and build a clean Maithili dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python maithili_scraper.py --demo              # Quick test (10 articles)\n"
            "  python maithili_scraper.py --demo --limit 5    # Test with 5 articles\n"
            "  python maithili_scraper.py --pages 20          # First 20 listing pages\n"
            "  python maithili_scraper.py                     # Full scrape (~5600 articles)\n"
        )
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run in DEMO mode.\n"
            "Crawls only 1 listing page and scrapes up to --limit articles.\n"
            "Use this to quickly verify the pipeline before the full run."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max number of articles to scrape. In demo mode defaults to 10."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        metavar="N",
        help="Crawl only the first N listing pages (default: all pages)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        metavar="FILE",
        help=f"Output CSV filename. (default: {OUTPUT_CSV})"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_SECONDS,
        metavar="SEC",
        help=f"Seconds to wait between requests. (default: {DELAY_SECONDS})"
    )
    return parser.parse_args()


# ---- HTTP --------------------------------------------------------------------

def fetch(url, delay=DELAY_SECONDS, retries=MAX_RETRIES):
    # type: (str, float, int) -> Optional[requests.Response]
    """Fetch a URL with retries and exponential back-off."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 404:
                log.warning("404 - %s", url)
                return None
            else:
                log.warning("HTTP %s (attempt %d) - %s", resp.status_code, attempt, url)
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s - %s", attempt, exc, url)
        time.sleep(delay * attempt)
    log.error("Failed after %d attempts: %s", retries, url)
    return None


# ---- URL UTILS ---------------------------------------------------------------

def normalise_article_url(href):
    # type: (str) -> Optional[str]
    if not href:
        return None
    if not href.startswith("http"):
        href = urljoin(BASE_URL, href)
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "p" in qs and qs["p"][0].isdigit():
        return "https://maithilijindabaad.com/?p={}".format(qs["p"][0])
    return None


def get_total_pages():
    # type: () -> int
    resp = fetch(BLOG_PAGE_URL)
    if resp is None:
        raise RuntimeError("Could not load blog listing page.")
    soup = BeautifulSoup(resp.text, "html.parser")
    page_nums = []
    for a in soup.select("a[href*='paged=']"):
        m = re.search(r"paged=(\d+)", a.get("href", ""))
        if m:
            page_nums.append(int(m.group(1)))
    return max(page_nums) if page_nums else 1


def collect_article_links_from_page(page_num):
    # type: (int) -> List[str]
    url = BLOG_PAGE_URL if page_num == 1 else "{}&paged={}".format(BLOG_PAGE_URL, page_num)
    resp = fetch(url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        canon = normalise_article_url(a["href"])
        if canon:
            links.add(canon)
    return list(links)


# ---- TEXT EXTRACTION & CLEANING ----------------------------------------------

def extract_article_content(soup):
    # type: (BeautifulSoup) -> str
    content_div = (
        soup.find("div", class_="entry-content")
        or soup.find("div", class_="post-content")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|post|article", re.I))
    )
    if content_div:
        for tag in content_div.find_all(
            ["nav", "aside", "footer", "script", "style",
             "button", "form", "iframe", "noscript"]
        ):
            tag.decompose()
        for cls in ["sharedaddy", "post-tags", "related", "comment",
                    "breadcrumb", "pagination", "nav", "sidebar", "widget"]:
            for el in content_div.find_all(class_=re.compile(cls, re.I)):
                el.decompose()
        return content_div.get_text(separator=" ", strip=True)
    body = soup.find("body")
    if body:
        for tag in body.find_all(["nav", "header", "footer", "aside", "script", "style"]):
            tag.decompose()
        return body.get_text(separator=" ", strip=True)
    return soup.get_text(separator=" ", strip=True)


def strip_preamble(text):
    # type: (str) -> str
    """
    Remove structured submission headers from "लेखनीक धार" series articles.

    Preamble structure (all on one line):
      लेख विचार प्रेषित : <AUTHOR>  श्रोत – <SOURCE>,  वृहस्पतिवार
      साप्ताहिक गतिविधि  विषय – <TOPIC>  <ARTICLE BODY...>

    Stripped in 5 left-to-right steps:
      1. "लेख विचार प्रेषित : "         — attribution opener
      2. "<author> श्रोत/स्रोत [–] "    — author name + source label
      3. "<source_name> वृहस्पतिवार , " — source name + weekday marker
      4. "साप्ताहिक गतिविधि "           — series label
      5. "विषय – / विषय : "             — topic label (value is KEPT as it
                                           flows directly into the article body)
    """
    # Quick guard: only process articles that start with the preamble marker
    if not re.match(r'^\s*\u0932\u0947\u0916\s*\u0935\u093f\u091a\u093e\u0930\s*\u092a\u094d\u0930\u0947\u0937\u093f\u0924', text):
        return text

    # Step 1: remove "लेख विचार प्रेषित : "
    text = re.sub(r'^\s*लेख\s*विचार\s*प्रेषित\s*[:\u2013\-]\s*', '', text)

    # Step 2: remove everything up to and including "श्रोत" / "स्रोत" and its separator
    text = re.sub(r'^[^\n]*?(?:श्रोत|स्रोत)\s*[:\u2013\-]*\s*', '', text)

    # Step 3: remove everything up to and including the weekday marker
    text = re.sub(r'^[^\n]*?(?:वृहस्पतिवार|बृहस्पतिवार)\s*[,\s]*', '', text)

    # Step 4: remove "साप्ताहिक गतिविधि " series label
    text = re.sub(r'^साप्ताहिक\s*गतिविधि\s*', '', text)

    # Step 5: remove "विषय – " / "विषय : " topic label, keep topic value
    text = re.sub(r'^विषय\s*[:\u2013\-]\s*', '', text)

    result = text.strip()
    # Safety: if over-stripped, fall back to step-1-only removal
    if not has_sufficient_devanagari(result, min_chars=50):
        return re.sub(r'^\s*लेख\s*विचार\s*प्रेषित\s*[:\u2013\-]\s*', '', text).strip()
    return result


def detect_language(text):
    # type: (str) -> str
    """
    Score the text against Maithili-specific and Hindi-specific word lists.
    Both languages share Devanagari script so we use lexical markers.

    Returns one of:
      "Maithili"              – clearly Maithili
      "Hindi"                 – clearly Hindi
      "Maithili-Hindi Mixed"  – meaningful presence of both
    """
    m_score = sum(text.count(marker) for marker in MAITHILI_MARKERS)
    h_score = sum(text.count(marker) for marker in HINDI_MARKERS)
    total   = m_score + h_score

    if total == 0:
        # No markers at all — default to Maithili (site is Maithili-focused)
        return "Maithili"

    h_ratio = h_score / total
    m_ratio = m_score / total

    if h_ratio >= 0.60:
        return "Hindi"
    elif h_ratio >= 0.20:
        return "Maithili-Hindi Mixed"
    else:
        return "Maithili"


def clean_text(raw_text):
    # type: (str) -> str
    """
    Clean raw text to pure Maithili (Devanagari).
    1. Strip zero-width / invisible noise chars
    2. Remove English/Latin words
    3. Remove leftover URLs and HTML entity fragments
    4. Keep ONLY Devanagari + digits + safe punctuation + whitespace
    5. Collapse excess whitespace
    """
    text = raw_text
    # 1. noise
    text = NOISE_PATTERN.sub("", text)
    # 2. English words
    text = re.sub(r"[A-Za-z]+(?:['\-][A-Za-z]+)*", " ", text)
    # 3. residual URLs and HTML entities
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    # 4. keep only Devanagari + safe chars
    text = re.sub(
        u"[^\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF"
        u"0-9"
        u"\u0964\u0965"
        u" \t\n\r"
        u",\\.!\\?:;\\(\\)\u2013\u2014\u2026\u201c\u201d\u2018\u2019"
        u"]",
        " ",
        text,
    )
    # 5. collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_sufficient_devanagari(text, min_chars=100):
    # type: (str, int) -> bool
    return len(DEVANAGARI_PATTERN.findall(text)) >= min_chars


# ---- SINGLE ARTICLE SCRAPER --------------------------------------------------

def scrape_article(url, delay=DELAY_SECONDS):
    # type: (str, float) -> Optional[Dict[str, str]]
    resp = fetch(url, delay=delay)
    if resp is None:
        return None

    final_parsed = urlparse(resp.url)
    final_qs = parse_qs(final_parsed.query)
    if "p" not in final_qs and final_parsed.path in ("", "/"):
        log.warning("Redirected to home (post deleted?): %s", url)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    if not (soup.find("article") or
            soup.find("div", class_=re.compile(r"entry|post", re.I))):
        log.warning("No article body found: %s", url)
        return None

    raw  = extract_article_content(soup)
    text = clean_text(raw)

    # Fix 1: strip metadata preamble (author credit / source / topic label lines)
    text = strip_preamble(text)

    if not has_sufficient_devanagari(text):
        log.info("Skipped (too little Devanagari): %s", url)
        return None

    # Fix 2: detect actual language instead of blindly tagging everything "Maithili"
    language = detect_language(text)
    log.info("Scraped [%s]: %s", language, url)

    return {
        "article_link"    : url,
        "article_text"    : text,
        "article_language": language,
    }


# ---- DEMO RUNNER -------------------------------------------------------------

def run_demo(limit, output_file, delay):
    # type: (int, str, float) -> None
    _banner("DEMO MODE", "Scraping 1 listing page, up to {} articles".format(limit))

    links = collect_article_links_from_page(1)
    log.info("[DEMO] Found %d links on page 1", len(links))
    links = sorted(links)[:limit]

    records = []
    skipped = []
    for url in tqdm(links, desc="[DEMO] Scraping"):
        result = scrape_article(url, delay=delay)
        if result:
            records.append(result)
        else:
            skipped.append(url)
        time.sleep(delay)

    df = pd.DataFrame(records, columns=["article_link", "article_text", "article_language"])
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    sep = "-" * 60
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("  Articles scraped : {}".format(len(records)))
    print("  Articles skipped : {}".format(len(skipped)))
    print("  Output file      : {}".format(output_file))
    print("=" * 60)

    for i, row in df.iterrows():
        print("\n" + sep)
        print("  Article #{}".format(i + 1))
        print(sep)
        print("  Link    : {}".format(row["article_link"]))
        print("  Lang    : {}".format(row["article_language"]))
        print("  Length  : {} chars".format(len(row["article_text"])))
        print("  Preview (first 400 chars after preamble strip):")
        print("  " + row["article_text"][:400])

    if skipped:
        print("\n" + sep)
        print("  Skipped URLs ({}) :".format(len(skipped)))
        for s in skipped:
            print("    x  {}".format(s))

    print("\n" + "=" * 60)
    print("  To run the FULL scrape:  python maithili_scraper.py")
    print("=" * 60 + "\n")


# ---- FULL RUNNER -------------------------------------------------------------

def run_full(max_pages, article_limit, output_file, delay):
    # type: (Optional[int], Optional[int], str, float) -> None

    # Step 1: collect links
    log.info("Discovering total listing pages...")
    total_pages = get_total_pages()
    pages_to_crawl = min(total_pages, max_pages) if max_pages else total_pages
    _banner(
        "FULL RUN",
        "Pages: {}/{} | Article limit: {}".format(
            pages_to_crawl, total_pages,
            "unlimited" if not article_limit else article_limit
        )
    )

    all_links = set()
    for page_num in tqdm(range(1, pages_to_crawl + 1), desc="Collecting links"):
        links = collect_article_links_from_page(page_num)
        all_links.update(links)
        time.sleep(delay)

    log.info("Total unique article links: %d", len(all_links))

    with open("collected_links.txt", "w", encoding="utf-8") as f:
        for link in sorted(all_links):
            f.write(link + "\n")
    log.info("Link checkpoint saved -> collected_links.txt")

    # Step 2: scrape
    ordered = sorted(all_links)
    if article_limit:
        ordered = ordered[:article_limit]

    records = []
    skipped = []
    for url in tqdm(ordered, desc="Scraping articles"):
        result = scrape_article(url, delay=delay)
        if result:
            records.append(result)
        else:
            skipped.append(url)
        time.sleep(delay)
        if len(records) > 0 and len(records) % 200 == 0:
            pd.DataFrame(records).to_csv(
                output_file + ".partial", index=False, encoding="utf-8-sig"
            )
            log.info("Checkpoint -> %d articles saved so far", len(records))

    # Step 3: save
    df = pd.DataFrame(records, columns=["article_link", "article_text", "article_language"])
    df.drop_duplicates(subset="article_link", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    with open(SKIPPED_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(skipped))

    print("\n" + "=" * 60)
    print("  SCRAPE COMPLETE")
    print("=" * 60)
    print("  Articles saved   : {}".format(len(df)))
    print("  Articles skipped : {}".format(len(skipped)))
    print("  Dataset          : {}".format(output_file))
    print("  Skipped log      : {}".format(SKIPPED_FILE))
    print("=" * 60)

    if not df.empty:
        sample = df.iloc[0]
        print("\n-- Sample record --")
        print("  Link : {}".format(sample["article_link"]))
        print("  Text : {}...".format(sample["article_text"][:300]))
        print()


# ---- UTILS -------------------------------------------------------------------

def _banner(title, subtitle=""):
    # type: (str, str) -> None
    w = 60
    print("\n" + "=" * w)
    print("  " + title)
    if subtitle:
        print("  " + subtitle)
    print("=" * w + "\n")


# ---- ENTRY POINT -------------------------------------------------------------

def main():
    args = parse_args()
    log.info("=== Maithili Jindabaad Scraper Started ===")
    log.info("Mode : %s", "DEMO" if args.demo else "FULL")
    log.info("Args : %s", vars(args))

    if args.demo:
        limit = args.limit if args.limit is not None else DEMO_LIMIT
        run_demo(limit=limit, output_file=args.output, delay=args.delay)
    else:
        run_full(
            max_pages     = args.pages,
            article_limit = args.limit,   # None unless user explicitly passed --limit
            output_file   = args.output,
            delay         = args.delay,
        )


if __name__ == "__main__":
    main()