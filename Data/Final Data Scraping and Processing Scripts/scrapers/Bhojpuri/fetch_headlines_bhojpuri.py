"""
fetch_headlines_bhojpuri.py
===========================
Reads your scraped CSV, visits each article URL in the `article_url`
column, extracts the article headline (the <h1> tag on the page), and
writes the result into a new `article_headline` column.

Website  : khabarbhojpuri.com  (WordPress, Bhojpuri news portal)
Headline : <h1> tag on each article page

Features
--------
- Resumes from where it left off (progress saved to a JSON checkpoint file)
- Polite crawling: randomised delay between requests (1–3 s by default)
- Retries failed requests up to 3 times with exponential back-off
- Skips rows whose `article_url` is empty / NaN
- Saves the enriched CSV to a new file so the original is never touched
- Prints a live progress bar using tqdm

Usage
-----
    python fetch_headlines_bhojpuri.py                          # uses defaults below
    python fetch_headlines_bhojpuri.py --input my_data.csv
    python fetch_headlines_bhojpuri.py --input my_data.csv --output enriched.csv
    python fetch_headlines_bhojpuri.py --input my_data.csv --delay 2 --workers 1

Dependencies
------------
    pip install requests beautifulsoup4 tqdm pandas
"""

import argparse
import json
import os
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── defaults (override via CLI flags) ────────────────────────────────────────
DEFAULT_INPUT      = "bhojpuri_articles_4982_final.csv"
DEFAULT_OUTPUT     = "scraped_data_with_headlines.csv"
DEFAULT_DELAY      = 1.5        # seconds between requests (avg)
DEFAULT_JITTER     = 1.0        # ± seconds of randomness added to delay
DEFAULT_RETRIES    = 3          # max retries per URL
DEFAULT_CHECKPOINT = "bhojpuri_headline_checkpoint.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; HeadlineFetcher/1.0; "
        "+https://github.com/your-repo)"
    ),
    "Accept-Language": "hi,en;q=0.9",
}
# ─────────────────────────────────────────────────────────────────────────────


def extract_headline(html: str) -> str:
    """
    Extract the article headline from the page HTML.

    Strategy (in order of preference):
    1. First <h1> tag on the page — the post title on khabarbhojpuri.com article pages
    2. <title> tag, stripping the site-name suffix " - खबर भोजपुरी"
    3. og:title meta tag
    4. Empty string if nothing found
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. <h1> — most reliable for WordPress single-post pages
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    # 2. <title> tag fallback — strip site name suffix
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        for suffix in [
            " - खबर भोजपुरी",
            " – खबर भोजपुरी",
            "- खबर भोजपुरी",
            "– खबर भोजपुरी",
            " - Khabar Bhojpuri",
            " – Khabar Bhojpuri",
        ]:
            if suffix in title_text:
                return title_text.replace(suffix, "").strip()
        return title_text

    # 3. og:title meta fallback
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    return ""


def fetch_with_retry(
    url: str,
    retries: int,
    delay: float,
    jitter: float,
    session: requests.Session,
) -> str | None:
    """Fetch a URL, retrying on failure. Returns HTML string or None."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding   # handle Devanagari correctly
            return resp.text
        except requests.RequestException as exc:
            wait = delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            if attempt < retries:
                tqdm.write(
                    f"  ⚠  Attempt {attempt} failed for {url}: {exc}. "
                    f"Retrying in {wait:.1f}s…"
                )
                time.sleep(wait)
            else:
                tqdm.write(f"  ✗  All {retries} attempts failed for {url}: {exc}")
    return None


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch headlines for khabarbhojpuri.com articles"
    )
    parser.add_argument("--input",      default=DEFAULT_INPUT,      help="Input CSV file path")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,     help="Output CSV file path")
    parser.add_argument("--url-col",    default="article_url",      help="Name of the URL column")
    parser.add_argument("--delay",      type=float, default=DEFAULT_DELAY,   help="Base delay in seconds")
    parser.add_argument("--jitter",     type=float, default=DEFAULT_JITTER,  help="Random jitter in seconds")
    parser.add_argument("--retries",    type=int,   default=DEFAULT_RETRIES, help="Max retries per URL")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint file for resuming")
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\n📂 Loading '{args.input}' …")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"   {len(df):,} rows × {len(df.columns)} columns loaded.")

    if args.url_col not in df.columns:
        raise ValueError(
            f"Column '{args.url_col}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Ensure output column exists
    if "article_headline" not in df.columns:
        df["article_headline"] = ""
    else:
        df["article_headline"] = df["article_headline"].fillna("")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = load_checkpoint(args.checkpoint)
    already_done = sum(1 for v in checkpoint.values() if v != "")
    print(f"   Checkpoint loaded: {already_done:,} URLs already fetched.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    session = requests.Session()
    save_every = 50
    failed_urls = []

    urls = df[args.url_col].tolist()

    with tqdm(total=len(urls), desc="Fetching headlines", unit="article") as pbar:
        for idx, url in enumerate(urls):
            pbar.update(1)

            # Skip empty / NaN links
            if not isinstance(url, str) or not url.strip():
                df.at[idx, "article_headline"] = ""
                continue

            url = url.strip()

            # Use cached result if available
            if url in checkpoint and checkpoint[url]:
                df.at[idx, "article_headline"] = checkpoint[url]
                continue

            # Fetch page
            html = fetch_with_retry(url, args.retries, args.delay, args.jitter, session)
            if html is None:
                headline = "FETCH_ERROR"
                failed_urls.append(url)
            else:
                headline = extract_headline(html)

            df.at[idx, "article_headline"] = headline
            checkpoint[url] = headline

            # Polite delay
            sleep_time = args.delay + random.uniform(-args.jitter / 2, args.jitter)
            time.sleep(max(0.5, sleep_time))

            # Periodic save
            if (idx + 1) % save_every == 0:
                save_checkpoint(args.checkpoint, checkpoint)
                df.to_csv(args.output, index=False, encoding="utf-8-sig")
                tqdm.write(f"   💾 Progress saved at row {idx + 1}")

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(args.checkpoint, checkpoint)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    total_fetched   = df["article_headline"].ne("").ne("FETCH_ERROR").sum()
    total_errors    = (df["article_headline"] == "FETCH_ERROR").sum()
    total_empty_url = df[args.url_col].isna().sum() + (df[args.url_col] == "").sum()

    print(f"\n✅ Done!")
    print(f"   Output saved to  : {args.output}")
    print(f"   Headlines fetched: {total_fetched:,}")
    print(f"   Fetch errors     : {total_errors:,}")
    print(f"   Empty URL rows   : {total_empty_url:,}")
    if failed_urls:
        print(f"\n   Failed URLs ({len(failed_urls)}):")
        for u in failed_urls[:20]:
            print(f"     {u}")
        if len(failed_urls) > 20:
            print(f"     … and {len(failed_urls) - 20} more (see checkpoint file)")


if __name__ == "__main__":
    main()
