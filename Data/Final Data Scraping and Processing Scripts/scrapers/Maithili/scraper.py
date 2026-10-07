import requests
import time
import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

HEADERS = {"User-Agent": "Mozilla/5.0"}
MIN_ARTICLE_LENGTH = 300


# Get Article URLs from Post Sitemaps


def get_article_urls_from_sitemap(base_url):
    base = base_url.replace("http://", "https://").rstrip("/")
    sitemap_index = base + "/sitemap.xml"
    article_urls = []

    try:
        response = requests.get(sitemap_index, headers=HEADERS, timeout=15)
        root = ET.fromstring(response.content)

        for loc in root.findall(".//{*}loc"):
            sitemap_link = loc.text

            if "post-sitemap" in sitemap_link:
                child_resp = requests.get(sitemap_link, headers=HEADERS, timeout=15)
                child_root = ET.fromstring(child_resp.content)

                for url_tag in child_root.findall(".//{*}loc"):
                    url = url_tag.text

                    if not url.endswith((".jpg", ".png", ".jpeg", ".gif", ".webp", ".pdf")):
                        article_urls.append(url)

        article_urls = list(set(article_urls))
        print(f"Total article URLs found: {len(article_urls)}")

        return article_urls

    except Exception as e:
        print("Sitemap error:", e)
        return []


# -----------------------------------
# Extract Clean Article Text
# -----------------------------------

def extract_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "lxml")

        content = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="post-content") or
            soup.find("div", class_="td-post-content") or
            soup.find("article")
        )

        if not content:
            return None

        # 🔥 REMOVE comment forms completely
        for tag in content.find_all(["form", "textarea", "input"]):
            tag.decompose()

        # 🔥 REMOVE any div/section containing "comment" in class name
        for tag in content.find_all(["div", "section"]):
            classes = tag.get("class")
            if classes and any("comment" in c.lower() for c in classes):
                tag.decompose()

        paragraphs = content.find_all("p")

        clean_paragraphs = [
            p.get_text().strip()
            for p in paragraphs
            if p.get_text().strip()
        ]

        text = " ".join(clean_paragraphs).strip()

        if len(text) < MIN_ARTICLE_LENGTH:
            return None

        return text

    except:
        return None


# -----------------------------------
# Main Scraper
# -----------------------------------

def scrape_site(base_url, max_articles=None):
    print(f"\nScraping: {base_url}")

    article_urls = get_article_urls_from_sitemap(base_url)

    data = []
    count = 0

    for url in article_urls:

        if max_articles and count >= max_articles:
            break

        if "/images/" in url:
            continue

        print(f"[{count+1}] Processing: {url}")

        text = extract_article(url)

        if text:
            if "esamaad" in base_url:
                language = "maithili"
            else:
                language = "bhojpuri"

            data.append({
                "website": urlparse(base_url).netloc,
                "article_url": url,
                "language": language,
                "article_text": text
            })

            count += 1

        time.sleep(0.3)

    return data


# -----------------------------------
# Entry Point
# -----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--demo", type=int, help="Limit number of articles (optional)")
    args = parser.parse_args()

    data = scrape_site(args.site, max_articles=args.demo)

    if not os.path.exists("output"):
        os.makedirs("output")

    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["article_url"], inplace=True)

    output_file = "output/scraped_articles.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nCompleted.")
    print("Total Articles Scraped:", len(df))


if __name__ == "__main__":
    main()