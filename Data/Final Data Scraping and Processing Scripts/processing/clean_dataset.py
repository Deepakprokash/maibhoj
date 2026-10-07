"""
Maithili Dataset Cleaner
==========================
Fixes the 3 issues found by validate_dataset.py on the full scraped CSV:

  Issue 1 — 4 articles with < 50% Devanagari     → DROP them
  Issue 2 — 15 articles with preamble remnants    → RE-STRIP preamble
  Issue 3 — 18 duplicate article texts            → DEDUPLICATE (keep first)
  Bonus   — 4 very short articles (< 200 chars)   → DROP them

Usage:
    python clean_dataset.py                                    # default filenames
    python clean_dataset.py --input my.csv --output clean.csv  # custom filenames
    python clean_dataset.py --dry-run                          # report only, no write
"""

import re
import csv
import sys
import argparse
from collections import OrderedDict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# ── SAME CONSTANTS AS SCRAPER / VALIDATOR ────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]')
NOISE_RE      = re.compile(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00a0]')

PREAMBLE_KW   = ['लेख विचार प्रेषित', 'साप्ताहिक गतिविधि',
                 'वृहस्पतिवार', 'बृहस्पतिवार']

# Minimum chars to keep an article
MIN_CHARS = 200
# Minimum Devanagari ratio to keep an article
MIN_DEVA_RATIO = 0.50


# ── ENHANCED PREAMBLE STRIPPER ────────────────────────────────────────────────
# The scraper's strip_preamble only triggers on "लेख विचार प्रेषित".
# Some articles have a shorter preamble that skips the attribution line and
# starts directly with साप्ताहिक गतिविधि or have वृहस्पतिवार in a header.
# This enhanced version handles all observed patterns.

def strip_preamble(text):
    """
    Remove submission preamble from article text.
    Handles all variants observed in the full dataset:

    Variant A (full):
      लेख विचार प्रेषित : <author>  श्रोत/स्रोत – <source>  वृहस्पतिवार
      साप्ताहिक गतिविधि  विषय – <TOPIC>  <body>

    Variant B (no attribution line):
      <source name>  वृहस्पतिवार / बृहस्पतिवार  साप्ताहिक गतिविधि  विषय – <TOPIC>  <body>

    Variant C (partial — only weekday label + topic remain):
      वृहस्पतिवार  साप्ताहिक गतिविधि  विषय – <TOPIC>  <body>

    Variant D (only साप्ताहिक गतिविधि + topic remain):
      साप्ताहिक गतिविधि  विषय – <TOPIC>  <body>
    """
    original = text

    # ── Variant A: starts with "लेख विचार प्रेषित"
    if re.match(r'^\s*लेख\s*विचार\s*प्रेषित', text):
        # Step 1: remove "लेख विचार प्रेषित : "
        text = re.sub(r'^\s*लेख\s*विचार\s*प्रेषित\s*[:\u2013\-]\s*', '', text)
        # Step 2: remove author + श्रोत/स्रोत label
        text = re.sub(r'^[^\n]*?(?:श्रोत|स्रोत)\s*[:\u2013\-]*\s*', '', text)
        # Step 3: remove source name + weekday
        text = re.sub(r'^[^\n]*?(?:वृहस्पतिवार|बृहस्पतिवार)\s*[,\s]*', '', text)
        # Step 4 & 5: साप्ताहिक गतिविधि + विषय label
        text = re.sub(r'^साप्ताहिक\s*गतिविधि\s*', '', text)
        text = re.sub(r'^विषय\s*[:\u2013\-]\s*', '', text)

    # ── Variant B: starts with source name then weekday (no attribution)
    elif re.search(r'(?:वृहस्पतिवार|बृहस्पतिवार)', text[:200]):
        # Remove everything up to and including the weekday marker
        text = re.sub(r'^[^\n]*?(?:वृहस्पतिवार|बृहस्पतिवार)\s*[,\s]*', '', text)
        # Then साप्ताहिक गतिविधि + विषय
        text = re.sub(r'^साप्ताहिक\s*गतिविधि\s*', '', text)
        text = re.sub(r'^विषय\s*[:\u2013\-]\s*', '', text)

    # ── Variant C/D: starts directly with साप्ताहिक गतिविधि
    elif re.match(r'^\s*साप्ताहिक\s*गतिविधि', text):
        text = re.sub(r'^साप्ताहिक\s*गतिविधि\s*', '', text)
        text = re.sub(r'^विषय\s*[:\u2013\-]\s*', '', text)

    else:
        return text   # no preamble detected

    result = text.strip()
    # Safety: if we stripped too much, return the original
    deva = len(DEVANAGARI_RE.findall(result))
    return result if deva >= 50 else original


def has_preamble_remnant(text):
    """Returns True if any preamble keyword appears in the first 300 chars."""
    first = text[:300]
    return any(kw in first for kw in PREAMBLE_KW)


def devanagari_ratio(text):
    if not text:
        return 0.0
    return len(DEVANAGARI_RE.findall(text)) / len(text)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Clean the full Maithili dataset CSV.")
    p.add_argument("--input",   default="maithili_full_dataset.csv")
    p.add_argument("--output",  default="maithili_full_dataset_clean.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be changed without writing output.")
    return p.parse_args()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    infile  = Path(args.input)
    outfile = Path(args.output)

    if not infile.exists():
        print(f"ERROR: Input file not found: {infile}")
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  MAITHILI DATASET CLEANER")
    print(f"  Input  : {infile}")
    print(f"  Output : {outfile}  {'(DRY RUN — no file written)' if args.dry_run else ''}")
    print(f"{'='*68}\n")

    rows = []
    with open(infile, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    original_count = len(rows)
    print(f"  Loaded {original_count:,} articles\n")

    stats = {
        'dropped_low_deva'   : [],
        'dropped_too_short'  : [],
        'preamble_fixed'     : [],
        'preamble_unfixable' : [],
        'dupes_dropped'      : [],
    }

    # ── Pass 1: Fix preamble remnants ────────────────────────────────────────
    print(f"{'─'*68}")
    print("  PASS 1 — Fixing preamble remnants")
    print(f"{'─'*68}")
    for row in rows:
        if has_preamble_remnant(row['article_text']):
            fixed = strip_preamble(row['article_text'])
            if has_preamble_remnant(fixed):
                # Still has remnant after stripping — mark for manual review
                stats['preamble_unfixable'].append(row['article_link'])
                print(f"  ⚠️  Could not fully fix: {row['article_link']}")
                print(f"     Remaining: {fixed[:120]}")
            else:
                stats['preamble_fixed'].append(row['article_link'])
                row['article_text'] = fixed
                print(f"  ✅  Fixed: {row['article_link'].split('=')[-1]}")

    print(f"\n  Fixed   : {len(stats['preamble_fixed'])}")
    print(f"  Unfixable: {len(stats['preamble_unfixable'])}")

    # ── Pass 2: Drop low Devanagari articles ─────────────────────────────────
    print(f"\n{'─'*68}")
    print("  PASS 2 — Dropping low Devanagari ratio articles (< 50%)")
    print(f"{'─'*68}")
    before = len(rows)
    kept = []
    for row in rows:
        ratio = devanagari_ratio(row['article_text'])
        if ratio < MIN_DEVA_RATIO:
            stats['dropped_low_deva'].append(row['article_link'])
            print(f"  ❌  Dropped ?p={row['article_link'].split('=')[-1]}"
                  f"  ({ratio*100:.1f}% Devanagari)")
        else:
            kept.append(row)
    rows = kept
    print(f"\n  Dropped : {before - len(rows)}")

    # ── Pass 3: Drop very short articles ─────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  PASS 3 — Dropping very short articles (< {MIN_CHARS} chars)")
    print(f"{'─'*68}")
    before = len(rows)
    kept = []
    for row in rows:
        if len(row['article_text']) < MIN_CHARS:
            stats['dropped_too_short'].append(row['article_link'])
            print(f"  ❌  Dropped ?p={row['article_link'].split('=')[-1]}"
                  f"  ({len(row['article_text'])} chars)")
        else:
            kept.append(row)
    rows = kept
    print(f"\n  Dropped : {before - len(rows)}")

    # ── Pass 4: Deduplicate by text content ───────────────────────────────────
    print(f"\n{'─'*68}")
    print("  PASS 4 — Deduplicating by article text")
    print(f"{'─'*68}")
    before = len(rows)
    seen_texts = OrderedDict()
    kept = []
    for row in rows:
        # Use first 500 chars as the deduplication key
        key = row['article_text'][:500].strip()
        if key in seen_texts:
            stats['dupes_dropped'].append(row['article_link'])
            print(f"  ❌  Dropped duplicate: ?p={row['article_link'].split('=')[-1]}"
                  f"  (same text as ?p={seen_texts[key].split('=')[-1]})")
        else:
            seen_texts[key] = row['article_link']
            kept.append(row)
    rows = kept
    print(f"\n  Dropped : {before - len(rows)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_count = len(rows)
    total_dropped = original_count - final_count

    print(f"\n{'='*68}")
    print("  CLEANING SUMMARY")
    print(f"{'='*68}")
    print(f"  Original articles     : {original_count:,}")
    print(f"  Preambles fixed       : {len(stats['preamble_fixed'])}")
    print(f"  Dropped (low Deva)    : {len(stats['dropped_low_deva'])}")
    print(f"  Dropped (too short)   : {len(stats['dropped_too_short'])}")
    print(f"  Dropped (duplicates)  : {len(stats['dupes_dropped'])}")
    print(f"  ─────────────────────────────────")
    print(f"  Final articles        : {final_count:,}  (-{total_dropped} removed)")
    print(f"{'='*68}")

    if stats['preamble_unfixable']:
        print(f"\n  ⚠️  {len(stats['preamble_unfixable'])} articles could not be auto-fixed:")
        for link in stats['preamble_unfixable']:
            print(f"     {link}")
        print("     These have been KEPT in the output but flagged above.")
        print("     They contain पृहस्पतिवार/साप्ताहिक गतिविधि"
              " within the article body itself (not as preamble).")

    # ── Write output ──────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n  DRY RUN — no file written.")
    else:
        with open(outfile, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  ✅  Saved to: {outfile}")
        print(f"  Run validate_dataset.py --file {outfile} to verify.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
