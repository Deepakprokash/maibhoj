"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BHOJPURI SCRAPER — DIAGNOSTIC & REPORT TOOL                        ║
║         Analyzes: scraped_articles_bhojpuri.csv (or any output CSV)        ║
║         Usage: python diagnose_bhojpuri.py --file your_output.csv          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import sys
import argparse
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import urlparse

try:
    import pandas as pd
except ImportError:
    sys.exit("❌  pandas not found. Run: pip install pandas")

# ── Unicode range helpers ──────────────────────────────────────────────────────
def devanagari_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Devanagari."""
    if not text:
        return 0.0
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    alpha = sum(1 for c in text if c.isalpha())
    return deva / alpha if alpha else 0.0

def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    latin = sum(1 for c in text if '\u0041' <= c <= '\u007A' or '\u00C0' <= c <= '\u024F')
    alpha = sum(1 for c in text if c.isalpha())
    return latin / alpha if alpha else 0.0

# ── Pattern detectors ──────────────────────────────────────────────────────────
SOCIAL_BOILERPLATE = [
    r'share\s+this', r'follow\s+us', r'subscribe', r'click\s+here',
    r'facebook', r'instagram', r'twitter', r'youtube', r'whatsapp',
    r'telegram', r'download\s+app', r'read\s+more', r'also\s+read',
    r'related\s+post', r'tags?\s*:', r'category\s*:', r'posted\s+by',
    r'leave\s+a\s+comment', r'copyright', r'all\s+rights\s+reserved',
    r'privacy\s+policy', r'terms\s+of\s+use',
]

HTML_ARTIFACTS = [
    r'<[a-zA-Z]', r'&amp;', r'&nbsp;', r'&lt;', r'&gt;', r'&quot;',
    r'\[caption', r'\[/caption\]', r'\[gallery', r'<!--', r'-->',
]

TRUNCATION_PATTERNS = [
    r'\.\.\.?\s*$',          # ends with ellipsis
    r'\bपढ़ें\s*$',           # "read" at end (common "read more" fragment)
    r'\bक्लिक\s*$',           # "click" fragment
    r'In\s+["\u201c\u201d]', # "In quote" that the scraper uses as a break-signal
]

def check_flags(text: str) -> list:
    """Return a list of issue flags for a single article text."""
    flags = []

    if devanagari_ratio(text) < 0.30:
        flags.append("LOW_DEVANAGARI")
    if latin_ratio(text) > 0.60:
        flags.append("HIGH_ENGLISH")
    if '#' in text:
        flags.append("LEFTOVER_HASHTAG")
    if re.search(r'#\S+', text):
        flags.append("HASHTAG_TOKEN")
    for pat in SOCIAL_BOILERPLATE:
        if re.search(pat, text, re.IGNORECASE):
            flags.append("SOCIAL_BOILERPLATE")
            break
    for pat in HTML_ARTIFACTS:
        if re.search(pat, text):
            flags.append("HTML_ARTIFACT")
            break
    for pat in TRUNCATION_PATTERNS:
        if re.search(pat, text):
            flags.append("POSSIBLE_TRUNCATION")
            break
    if len(text) < 300:
        flags.append("BELOW_MIN_LENGTH")
    if len(text) < 500:
        flags.append("SHORT_ARTICLE")
    if len(set(text.split())) < 20:
        flags.append("LOW_UNIQUE_WORDS")

    return flags


# ── URL analysis ───────────────────────────────────────────────────────────────
def extract_category(url: str) -> str:
    """Try to parse a category slug from the URL path."""
    try:
        path = urlparse(url).path.strip("/")
        parts = path.split("/")
        # Usually: /category/slug/ or /slug/article-name/
        if len(parts) >= 2:
            return parts[0]
        return parts[0] if parts else "unknown"
    except Exception:
        return "unknown"


# ── Duplicate detection ────────────────────────────────────────────────────────
def fingerprint(text: str) -> str:
    """Cheap text fingerprint: first 200 normalised chars."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return normalized[:200]


# ── Main diagnostic ────────────────────────────────────────────────────────────
def run_diagnostics(filepath: str):
    print("\n" + "═" * 72)
    print("  BHOJPURI SCRAPER — DIAGNOSTIC REPORT")
    print("  File:", filepath)
    print("═" * 72)

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    except FileNotFoundError:
        sys.exit(f"❌  File not found: {filepath}")

    total = len(df)
    print(f"\n📂  Rows loaded          : {total:,}")

    # ── Column check ─────────────────────────────────────────────────────────
    expected_cols = {"website", "article_url", "language", "article_text"}
    actual_cols   = set(df.columns)
    missing_cols  = expected_cols - actual_cols
    extra_cols    = actual_cols  - expected_cols
    print(f"📋  Columns present      : {list(df.columns)}")
    if missing_cols:
        print(f"⚠️   Missing columns      : {missing_cols}")
    if extra_cols:
        print(f"ℹ️   Extra columns        : {extra_cols}")

    # ── Null / empty ─────────────────────────────────────────────────────────
    null_text = df["article_text"].isna().sum() if "article_text" in df.columns else 0
    empty_text = (df["article_text"].fillna("").str.strip() == "").sum() \
                 if "article_text" in df.columns else 0
    null_url  = df["article_url"].isna().sum() if "article_url" in df.columns else 0

    print(f"\n── NULL / EMPTY ───────────────────────────────────────────────────")
    print(f"   Null article_text     : {null_text:,}")
    print(f"   Empty article_text    : {empty_text:,}")
    print(f"   Null article_url      : {null_url:,}")

    # Drop rows with no text for further analysis
    df = df[df["article_text"].notna() & (df["article_text"].str.strip() != "")].copy()
    usable = len(df)
    print(f"   Usable rows (non-empty): {usable:,}  ({100*usable/total:.1f}% of total)")

    if usable == 0:
        print("\n❌  No usable rows — nothing to analyse further.")
        return

    # ── Duplicates ────────────────────────────────────────────────────────────
    dup_urls  = df["article_url"].duplicated().sum() if "article_url" in df.columns else 0
    df["_fp"] = df["article_text"].apply(fingerprint)
    dup_text  = df["_fp"].duplicated().sum()

    print(f"\n── DUPLICATES ─────────────────────────────────────────────────────")
    print(f"   Duplicate URLs        : {dup_urls:,}")
    print(f"   Duplicate texts (≈)   : {dup_text:,}")

    # ── Length distribution ───────────────────────────────────────────────────
    df["_len"] = df["article_text"].str.len()
    pct = [10, 25, 50, 75, 90, 95, 99]
    quantiles = df["_len"].quantile([p/100 for p in pct])

    print(f"\n── ARTICLE LENGTH (characters) ────────────────────────────────────")
    print(f"   Min    : {df['_len'].min():,}")
    print(f"   Max    : {df['_len'].max():,}")
    print(f"   Mean   : {df['_len'].mean():,.0f}")
    print(f"   Median : {df['_len'].median():,.0f}")
    for p, q in zip(pct, quantiles):
        print(f"   p{p:02d}    : {q:,.0f}")

    buckets = [
        ("< 300   (below MIN)", (df["_len"] < 300).sum()),
        ("300–500 (borderline)", ((df["_len"] >= 300) & (df["_len"] < 500)).sum()),
        ("500–1000",             ((df["_len"] >= 500) & (df["_len"] < 1000)).sum()),
        ("1000–2000",            ((df["_len"] >= 1000) & (df["_len"] < 2000)).sum()),
        ("2000–5000",            ((df["_len"] >= 2000) & (df["_len"] < 5000)).sum()),
        ("> 5000  (long)",       (df["_len"] >= 5000).sum()),
    ]
    print(f"\n   Length buckets:")
    for label, count in buckets:
        bar = "█" * min(40, int(40 * count / usable))
        print(f"     {label:<28} {count:>5,}  {bar}")

    # ── Language / Script quality ─────────────────────────────────────────────
    df["_deva_ratio"]  = df["article_text"].apply(devanagari_ratio)
    df["_latin_ratio"] = df["article_text"].apply(latin_ratio)

    low_deva   = (df["_deva_ratio"] < 0.30).sum()
    mid_deva   = ((df["_deva_ratio"] >= 0.30) & (df["_deva_ratio"] < 0.60)).sum()
    high_deva  = (df["_deva_ratio"] >= 0.60).sum()
    high_latin = (df["_latin_ratio"] > 0.60).sum()

    print(f"\n── SCRIPT QUALITY ─────────────────────────────────────────────────")
    print(f"   Devanagari ratio (mean): {df['_deva_ratio'].mean():.2%}")
    print(f"   Articles with < 30% Devanagari  : {low_deva:,}  ← likely English/mixed")
    print(f"   Articles with 30–60% Devanagari : {mid_deva:,}  ← mixed script")
    print(f"   Articles with ≥ 60% Devanagari  : {high_deva:,}  ← good Bhojpuri")
    print(f"   Articles with > 60% Latin       : {high_latin:,}  ← mostly English")

    # ── Per-article flag analysis ─────────────────────────────────────────────
    all_flags = []
    flagged_rows = []
    for idx, row in df.iterrows():
        flags = check_flags(row["article_text"])
        all_flags.extend(flags)
        if flags:
            flagged_rows.append({
                "article_url": row.get("article_url", ""),
                "length": row["_len"],
                "deva_ratio": f"{row['_deva_ratio']:.2%}",
                "flags": ", ".join(flags),
                "snippet": row["article_text"][:120].replace("\n", " ")
            })

    flag_counts = Counter(all_flags)
    print(f"\n── CONTENT FLAGS ──────────────────────────────────────────────────")
    print(f"   Total flagged articles: {len(flagged_rows):,} / {usable:,}  "
          f"({100*len(flagged_rows)/usable:.1f}%)")
    print(f"\n   Flag breakdown:")
    flag_descriptions = {
        "LOW_DEVANAGARI":      "< 30% Devanagari script",
        "HIGH_ENGLISH":        "> 60% Latin/English characters",
        "LEFTOVER_HASHTAG":    "Contains '#' character",
        "HASHTAG_TOKEN":       "Contains #word token",
        "SOCIAL_BOILERPLATE":  "Social media / nav boilerplate leaked",
        "HTML_ARTIFACT":       "Raw HTML or HTML entity present",
        "POSSIBLE_TRUNCATION": "May be cut short (ellipsis / break pattern)",
        "BELOW_MIN_LENGTH":    "Under 300 characters (scraper should have dropped)",
        "SHORT_ARTICLE":       "Under 500 characters",
        "LOW_UNIQUE_WORDS":    "Fewer than 20 unique words",
    }
    for flag, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
        desc = flag_descriptions.get(flag, "")
        print(f"     {flag:<30} {cnt:>5,}   — {desc}")

    # ── '#' split bug analysis ────────────────────────────────────────────────
    # The scraper does: text.split('#', 1)[-1]  →  keeps AFTER the first '#'
    # This means any article whose text had a '#' before real content is chopped.
    hashtag_articles = df[df["article_text"].str.contains(r'#', na=False)]
    print(f"\n── SCRAPER BUG: '#' SPLIT ─────────────────────────────────────────")
    print(f"   Articles still containing '#' after clean_text: {len(hashtag_articles):,}")
    print(f"   ⚠  The scraper splits on '#' keeping only text AFTER the first '#'.")
    print(f"      If '#' appears mid-article, all preceding content is LOST silently.")

    # ── 'In quote' truncation ─────────────────────────────────────────────────
    in_quote_pattern = r'In\s+["\u201c\u201d]'
    truncated = df[df["article_text"].str.contains(in_quote_pattern, na=False)]
    print(f"\n── SCRAPER BUG: 'In QUOTE' EARLY BREAK ───────────────────────────")
    print(f"   Articles containing 'In \"…\"' pattern (scraper breaks here): {len(truncated):,}")
    print(f"   ⚠  The scraper stops collecting paragraphs at this pattern.")
    print(f"      Articles ending at this signal may have substantial missing content.")

    # ── is_caption_like false positives ──────────────────────────────────────
    # Paragraphs < 80 chars with < 5 Devanagari chars are dropped.
    # In Bhojpuri, short sentences are valid — this filter may be too aggressive.
    short_deva_ok = df[
        (df["_len"] < 500) & (df["_deva_ratio"] > 0.50)
    ]
    print(f"\n── POTENTIAL OVER-FILTERING: is_caption_like ──────────────────────")
    print(f"   Short (< 500 ch) but high-Devanagari (> 50%) articles: {len(short_deva_ok):,}")
    print(f"   ⚠  The scraper skips paragraphs < 80 chars with < 5 Devanagari chars.")
    print(f"      Valid short Bhojpuri sentences may be silently dropped per paragraph.")

    # ── Thread-safety warning (structural bug) ───────────────────────────────
    print(f"\n── CRITICAL BUG: SELENIUM THREAD SAFETY ──────────────────────────")
    print(f"   The scraper shares Selenium drivers across threads:")
    print(f"   drivers[idx % num_workers] inside ThreadPoolExecutor")
    print(f"   ⚠  Selenium WebDriver is NOT thread-safe. Multiple threads hitting")
    print(f"      the same driver simultaneously corrupt page loads, cause blank")
    print(f"      pages, and silently return None — explaining many of the 1775")
    print(f"      missing articles (5000 attempted − 3225 saved).")

    # ── URL / category breakdown ──────────────────────────────────────────────
    if "article_url" in df.columns:
        df["_category"] = df["article_url"].apply(extract_category)
        cat_counts = df["_category"].value_counts().head(20)
        print(f"\n── URL CATEGORY DISTRIBUTION (top 20) ─────────────────────────────")
        for cat, cnt in cat_counts.items():
            bar = "█" * min(40, int(40 * cnt / usable))
            print(f"   {cat:<35} {cnt:>5,}  {bar}")

    # ── Samples of flagged articles ───────────────────────────────────────────
    print(f"\n── SAMPLE FLAGGED ARTICLES (up to 10 per flag type) ───────────────")
    for flag in sorted(flag_counts.keys(), key=lambda x: -flag_counts[x]):
        samples = [r for r in flagged_rows if flag in r["flags"]][:5]
        print(f"\n  [{flag}]  ({flag_counts[flag]:,} articles)")
        for s in samples:
            print(f"    URL     : {s['article_url']}")
            print(f"    Length  : {s['length']:,}  Deva: {s['deva_ratio']}")
            print(f"    Snippet : {s['snippet']!r}")
            print()

    # ── Summary of scraper issues found ──────────────────────────────────────
    print("═" * 72)
    print("  SUMMARY OF ISSUES FOUND IN THE SCRAPER")
    print("═" * 72)

    issues = [
        ("CRITICAL", "Thread-safety race condition",
         "ThreadPoolExecutor shares Selenium drivers (drivers[idx % num_workers]).\n"
         "    Multiple threads hit the same driver → corrupted page loads → silent None returns.\n"
         "    FIX: Give each thread its own driver, or use a queue/pool pattern."),

        ("HIGH",     "# split destroys content before first hashtag",
         "Line: text = text.split('#', 1)[-1].strip()\n"
         "    Keeps only text AFTER the first '#'. If real content precedes a hashtag\n"
         "    (common in Bhojpuri articles), all of it is silently discarded.\n"
         "    FIX: Use re.sub(r'#\\S+', '', text) (which already exists in clean_text)."),

        ("HIGH",     "'In QUOTE' paragraph break is too aggressive",
         "The scraper stops collecting paragraphs when it sees: In \"...\"\n"
         "    This heuristic was meant to drop song lyrics, but also truncates\n"
         "    any genuine article paragraph that ends with a quoted source.\n"
         "    FIX: Only break if the pattern appears in the last paragraph, or\n"
         "    remove it and instead strip full quoted blocks."),

        ("MEDIUM",   "is_caption_like filter may drop valid Bhojpuri sentences",
         "Condition: len(text) < 80 and devanagari < 5\n"
         "    Short sentences with mixed script (English names/numbers in Bhojpuri)\n"
         "    are silently dropped paragraph-by-paragraph.\n"
         "    FIX: Raise the Devanagari threshold check or remove the length cap."),

        ("MEDIUM",   "No retry logic for failed article fetches",
         "extract_article() catches ALL exceptions and returns None.\n"
         "    Network timeouts, bot detection, or Selenium crashes are swallowed.\n"
         "    FIX: Add retry with exponential back-off (e.g. 3 retries)."),

        ("LOW",      "Unused variable lock_data",
         "lock_data = [] is defined but never used. Minor code smell."),

        ("LOW",      "wait_for_page only waits for entry-content, not all selectors",
         "The scraper waits for .entry-content but the content div might be\n"
         "    .et_pb_post_content or .td-post-content — it then falls through\n"
         "    immediately and may parse a partially loaded page.\n"
         "    FIX: Wait for any of the 5 content selector classes."),
    ]

    for severity, title, detail in issues:
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "•")
        print(f"\n  {icon}  [{severity}]  {title}")
        print(f"    {detail}")

    # ── Final numbers recap ───────────────────────────────────────────────────
    clean_articles = usable - len(flagged_rows)
    print(f"\n{'═'*72}")
    print(f"  FINAL NUMBERS")
    print(f"{'═'*72}")
    print(f"  Total attempted (approx)     : 5,000")
    print(f"  Saved in CSV                 : {total:,}  ({100*total/5000:.1f}% success rate)")
    print(f"  Usable (non-empty)           : {usable:,}")
    print(f"  Flagged / suspect            : {len(flagged_rows):,}  ({100*len(flagged_rows)/usable:.1f}%)")
    print(f"  Clean (no flags)             : {clean_articles:,}  ({100*clean_articles/usable:.1f}%)")
    print(f"  Duplicate texts              : {dup_text:,}")
    print(f"  Missing (not in CSV at all)  : {5000 - total:,}  ← primary cause: thread-safety bug")
    print()

    # ── Optionally export flagged CSV ─────────────────────────────────────────
    if flagged_rows:
        out = "flagged_articles_report.csv"
        pd.DataFrame(flagged_rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  📄  Flagged articles saved to: {out}")

    print(f"\n  ✅  Diagnostic complete.\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose a Bhojpuri scraper output CSV"
    )
    parser.add_argument(
        "--file", "-f",
        default="output/scraped_articles_bhojpuri.csv",
        help="Path to the scraped CSV file (default: output/scraped_articles_bhojpuri.csv)"
    )
    args = parser.parse_args()
    run_diagnostics(args.file)