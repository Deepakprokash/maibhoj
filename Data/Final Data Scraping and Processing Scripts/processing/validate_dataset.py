"""
Maithili Dataset Validator
============================
Run this on your full scraped CSV to get a complete quality report.

Usage:
    python validate_dataset.py                          # checks maithili_dataset.csv
    python validate_dataset.py --file my_dataset.csv   # checks a specific file
    python validate_dataset.py --sample 20             # also prints N random rows
"""

import re
import csv
import sys
import random
import argparse
from collections import Counter
from pathlib import Path

# Python's csv module has a default field size limit of 131,072 bytes.
# Long Maithili articles easily exceed this — raise it to the system maximum.
csv.field_size_limit(sys.maxsize)

# ── SAME MARKER LISTS AS SCRAPER ─────────────────────────────────────────────
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
PREAMBLE_KW = [
    'लेख विचार प्रेषित', 'साप्ताहिक गतिविधि',
    'वृहस्पतिवार', 'बृहस्पतिवार',
]

DEVANAGARI_RE  = re.compile(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]')
ENGLISH_RE     = re.compile(r'[A-Za-z]')
HTML_TAG_RE    = re.compile(r'<[^>]+>')
NOISE_RE       = re.compile(
    r'[^\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF'
    r'0-9\u0964\u0965 \t\n\r,\.!\?:;\(\)'
    r'\u2013\u2014\u2026\u201c\u201d\u2018\u2019\-"]'
)
VALID_URL_RE   = re.compile(r'^https://maithilijindabaad\.com/\?p=\d+$')

# ── HELPERS ───────────────────────────────────────────────────────────────────

def detect_language(text):
    m = sum(text.count(w) for w in MAITHILI_MARKERS)
    h = sum(text.count(w) for w in HINDI_MARKERS)
    total = m + h
    if total == 0:
        return "Maithili"
    ratio = h / total
    if ratio >= 0.60:
        return "Hindi"
    elif ratio >= 0.20:
        return "Maithili-Hindi Mixed"
    return "Maithili"

def hindi_ratio(text):
    m = sum(text.count(w) for w in MAITHILI_MARKERS)
    h = sum(text.count(w) for w in HINDI_MARKERS)
    total = m + h
    return round(h / total * 100, 1) if total else 0.0

def bar(value, total, width=30):
    filled = int(width * value / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def sep(char="─", width=68):
    return char * width

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate Maithili scraper output CSV.")
    parser.add_argument("--file",   default="maithili_dataset.csv", help="Path to CSV file")
    parser.add_argument("--sample", type=int, default=0,            help="Print N random sample rows")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n{sep('═')}")
    print(f"  MAITHILI DATASET VALIDATOR")
    print(f"  File : {csv_path}")
    print(f"{sep('═')}\n")

    rows = []
    try:
        with open(csv_path, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)

    total = len(rows)
    print(f"  Total rows loaded : {total:,}")
    print(f"  Columns           : {list(rows[0].keys()) if rows else '(empty)'}\n")

    if total == 0:
        print("ERROR: CSV is empty.")
        sys.exit(1)

    issues = []   # collect (check_name, count, examples)

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 1 — Required columns
    # ═══════════════════════════════════════════════════════════════════════
    required = {"article_link", "article_text", "article_language"}
    missing_cols = required - set(rows[0].keys())
    print(sep())
    print("  CHECK 1 — Required columns")
    print(sep())
    if missing_cols:
        print(f"  ❌  Missing columns: {missing_cols}")
        issues.append(("Missing columns", len(missing_cols), list(missing_cols)))
    else:
        print("  ✅  All 3 required columns present")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 2 — Duplicate article links
    # ═══════════════════════════════════════════════════════════════════════
    link_counts = Counter(r['article_link'] for r in rows)
    dupes = {k: v for k, v in link_counts.items() if v > 1}
    print(f"\n{sep()}")
    print("  CHECK 2 — Duplicate article links")
    print(sep())
    if dupes:
        print(f"  ❌  {len(dupes)} duplicate links found")
        for link, cnt in list(dupes.items())[:5]:
            print(f"       ({cnt}x) {link}")
        issues.append(("Duplicate links", len(dupes), list(dupes.keys())[:5]))
    else:
        print(f"  ✅  All {total:,} links are unique")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 3 — URL format validity
    # ═══════════════════════════════════════════════════════════════════════
    bad_urls = [r['article_link'] for r in rows if not VALID_URL_RE.match(r['article_link'])]
    print(f"\n{sep()}")
    print("  CHECK 3 — URL format  (must be ?p=NNN)")
    print(sep())
    if bad_urls:
        print(f"  ❌  {len(bad_urls)} malformed URLs")
        for u in bad_urls[:5]:
            print(f"       {u}")
        issues.append(("Malformed URLs", len(bad_urls), bad_urls[:5]))
    else:
        print(f"  ✅  All URLs follow correct ?p=NNN format")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 4 — Empty or near-empty articles
    # ═══════════════════════════════════════════════════════════════════════
    empty = [(r['article_link'], len(r['article_text'])) for r in rows
             if len(r['article_text'].strip()) < 100]
    print(f"\n{sep()}")
    print("  CHECK 4 — Empty / near-empty text  (< 100 chars)")
    print(sep())
    if empty:
        print(f"  ❌  {len(empty)} articles too short")
        for link, l in empty[:5]:
            print(f"       {link}  →  {l} chars")
        issues.append(("Empty articles", len(empty), [x[0] for x in empty[:5]]))
    else:
        print(f"  ✅  All articles have ≥ 100 chars of text")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 5 — Devanagari content ratio
    # ═══════════════════════════════════════════════════════════════════════
    low_deva = []
    for r in rows:
        t = r['article_text']
        if not t:
            continue
        deva = len(DEVANAGARI_RE.findall(t))
        ratio = deva / len(t)
        if ratio < 0.50:
            low_deva.append((r['article_link'], round(ratio*100, 1)))
    print(f"\n{sep()}")
    print("  CHECK 5 — Devanagari ratio  (must be ≥ 50% of chars)")
    print(sep())
    if low_deva:
        print(f"  ❌  {len(low_deva)} articles have < 50% Devanagari")
        for link, pct in low_deva[:5]:
            print(f"       {link}  →  {pct}% Devanagari")
        issues.append(("Low Devanagari ratio", len(low_deva), [x[0] for x in low_deva[:5]]))
    else:
        print(f"  ✅  All articles are ≥ 50% Devanagari characters")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 6 — English / Latin characters remaining
    # ═══════════════════════════════════════════════════════════════════════
    has_english = [(r['article_link'], re.findall(r'[A-Za-z]{2,}', r['article_text'])[:5])
                   for r in rows if ENGLISH_RE.search(r['article_text'])]
    print(f"\n{sep()}")
    print("  CHECK 6 — English / Latin characters remaining")
    print(sep())
    if has_english:
        print(f"  ❌  {len(has_english)} articles still contain English")
        for link, words in has_english[:5]:
            print(f"       {link}  →  {words}")
        issues.append(("English remaining", len(has_english), [x[0] for x in has_english[:5]]))
    else:
        print(f"  ✅  Zero English/Latin characters in any article")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 7 — HTML tags remaining
    # ═══════════════════════════════════════════════════════════════════════
    has_html = [(r['article_link'], HTML_TAG_RE.findall(r['article_text'])[:3])
                for r in rows if HTML_TAG_RE.search(r['article_text'])]
    print(f"\n{sep()}")
    print("  CHECK 7 — HTML tags remaining")
    print(sep())
    if has_html:
        print(f"  ❌  {len(has_html)} articles still contain HTML tags")
        for link, tags in has_html[:5]:
            print(f"       {link}  →  {tags}")
        issues.append(("HTML tags", len(has_html), [x[0] for x in has_html[:5]]))
    else:
        print(f"  ✅  No HTML tags found in any article")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 8 — Stray / unexpected symbols
    # ═══════════════════════════════════════════════════════════════════════
    has_noise = []
    for r in rows:
        hits = list(set(NOISE_RE.findall(r['article_text'])))
        if hits:
            has_noise.append((r['article_link'], hits[:6]))
    print(f"\n{sep()}")
    print("  CHECK 8 — Stray / unexpected non-Devanagari symbols")
    print(sep())
    if has_noise:
        print(f"  ❌  {len(has_noise)} articles have unexpected symbols")
        for link, chars in has_noise[:5]:
            display = [(hex(ord(c)), c) for c in chars]
            print(f"       {link}  →  {display}")
        issues.append(("Stray symbols", len(has_noise), [x[0] for x in has_noise[:5]]))
    else:
        print(f"  ✅  No unexpected symbols in any article")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 9 — Preamble remnants
    # ═══════════════════════════════════════════════════════════════════════
    has_preamble = []
    for r in rows:
        first300 = r['article_text'][:300]
        found = [kw for kw in PREAMBLE_KW if kw in first300]
        if found:
            has_preamble.append((r['article_link'], found))
    print(f"\n{sep()}")
    print("  CHECK 9 — Preamble / metadata remnants in article text")
    print(sep())
    if has_preamble:
        print(f"  ❌  {len(has_preamble)} articles still have preamble metadata")
        for link, kws in has_preamble[:5]:
            print(f"       {link}  →  {kws}")
        issues.append(("Preamble remnants", len(has_preamble), [x[0] for x in has_preamble[:5]]))
    else:
        print(f"  ✅  No preamble metadata found in any article")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 10 — Duplicate article texts
    # ═══════════════════════════════════════════════════════════════════════
    text_counts = Counter(r['article_text'][:300] for r in rows)
    dupe_texts  = {k: v for k, v in text_counts.items() if v > 1}
    print(f"\n{sep()}")
    print("  CHECK 10 — Duplicate article texts")
    print(sep())
    if dupe_texts:
        print(f"  ❌  {len(dupe_texts)} duplicate texts found")
        for snippet, cnt in list(dupe_texts.items())[:3]:
            print(f"       ({cnt}x) {snippet[:80]}...")
        issues.append(("Duplicate texts", len(dupe_texts), []))
    else:
        print(f"  ✅  All article texts are unique")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 11 — Language label distribution
    # ═══════════════════════════════════════════════════════════════════════
    lang_dist   = Counter(r['article_language'] for r in rows)
    # Re-detect language on each article and compare to stored label
    mismatches  = []
    for r in rows:
        detected = detect_language(r['article_text'])
        stored   = r['article_language']
        if detected != stored:
            mismatches.append((r['article_link'], stored, detected))
    print(f"\n{sep()}")
    print("  CHECK 11 — Language label distribution & accuracy")
    print(sep())
    print(f"  Distribution:")
    for lang, cnt in sorted(lang_dist.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"    {lang:<30} {cnt:>6,}  ({pct:.1f}%)  {bar(cnt, total, 20)}")
    if mismatches:
        print(f"\n  ⚠️   {len(mismatches)} label mismatches (stored vs re-detected):")
        for link, stored, detected in mismatches[:5]:
            print(f"       {link}  stored={stored}  detected={detected}")
    else:
        print(f"\n  ✅  All language labels match re-detection")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 12 — Text length distribution
    # ═══════════════════════════════════════════════════════════════════════
    lengths = sorted(len(r['article_text']) for r in rows)
    n = len(lengths)
    mean_l   = sum(lengths) // n
    median_l = lengths[n // 2]
    p10      = lengths[n // 10]
    p90      = lengths[int(n * 0.9)]
    buckets  = Counter()
    for l in lengths:
        if   l < 200:   buckets["< 200"]    += 1
        elif l < 500:   buckets["200–499"]  += 1
        elif l < 1000:  buckets["500–999"]  += 1
        elif l < 3000:  buckets["1k–2999"]  += 1
        elif l < 6000:  buckets["3k–5999"]  += 1
        else:           buckets["6k+"]      += 1
    print(f"\n{sep()}")
    print("  CHECK 12 — Text length distribution")
    print(sep())
    print(f"  Min    : {lengths[0]:,}     Max : {lengths[-1]:,}")
    print(f"  Mean   : {mean_l:,}     Median: {median_l:,}")
    print(f"  P10    : {p10:,}       P90   : {p90:,}")
    print(f"\n  Bucket breakdown:")
    for bucket, cnt in sorted(buckets.items()):
        pct = cnt / total * 100
        print(f"    {bucket:<12} {cnt:>6,}  ({pct:.1f}%)  {bar(cnt, total, 20)}")
    very_short = [r['article_link'] for r in rows if len(r['article_text']) < 200]
    if very_short:
        print(f"\n  ⚠️   {len(very_short)} very short articles (< 200 chars):")
        for link in very_short[:5]:
            print(f"       {link}")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 13 — Hindi leakage stats (informational)
    # ═══════════════════════════════════════════════════════════════════════
    hindi_ratios = [hindi_ratio(r['article_text']) for r in rows]
    hi_buckets   = Counter()
    for hr in hindi_ratios:
        if   hr == 0:    hi_buckets["0% (pure Maithili)"]   += 1
        elif hr < 10:    hi_buckets["1–9% (trace Hindi)"]   += 1
        elif hr < 20:    hi_buckets["10–19% (light mix)"]   += 1
        elif hr < 60:    hi_buckets["20–59% (heavy mix)"]   += 1
        else:            hi_buckets["60%+ (mostly Hindi)"]  += 1
    print(f"\n{sep()}")
    print("  CHECK 13 — Hindi leakage (informational)")
    print(sep())
    for bucket, cnt in sorted(hi_buckets.items()):
        pct = cnt / total * 100
        print(f"    {bucket:<30} {cnt:>6,}  ({pct:.1f}%)  {bar(cnt, total, 20)}")
    heavy = [r['article_link'] for r in rows if hindi_ratio(r['article_text']) >= 20]
    if heavy:
        print(f"\n  ⚠️   {len(heavy)} articles with ≥ 20% Hindi (labelled 'Mixed' or 'Hindi'):")
        for link in heavy[:10]:
            print(f"       {link}")

    # ═══════════════════════════════════════════════════════════════════════
    # RANDOM SAMPLE PREVIEW
    # ═══════════════════════════════════════════════════════════════════════
    if args.sample > 0:
        sample_rows = random.sample(rows, min(args.sample, total))
        print(f"\n{sep('═')}")
        print(f"  RANDOM SAMPLE — {len(sample_rows)} articles")
        print(sep('═'))
        for r in sample_rows:
            print(f"\n  Link : {r['article_link']}")
            print(f"  Lang : {r['article_language']}  |  Length: {len(r['article_text']):,} chars")
            print(f"  Text : {r['article_text'][:250]}...")
            print(f"  {sep('-', 60)}")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{sep('═')}")
    print(f"  FINAL SUMMARY")
    print(sep('═'))
    print(f"  Total articles : {total:,}")
    print()
    if not issues:
        print("  🎉  ALL CHECKS PASSED — Dataset is clean and ready to use!")
    else:
        print(f"  ⚠️   {len(issues)} check(s) need attention:\n")
        for name, count, examples in issues:
            print(f"    ❌  {name:<30} → {count:,} affected")
            for ex in examples[:3]:
                print(f"         {ex}")
    print(sep('═'))
    print()

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())