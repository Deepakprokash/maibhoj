"""
patch_sentiment.py  —  Patch missing sentiment for already-annotated rows
=========================================================================
Only calls the sentiment API for rows that have good summary + emotion
but are missing the sentiment label. Saves API cost by NOT redoing
summary or emotion calls.

Usage:
  export OPENAI_API_KEY="sk-..."
  python patch_sentiment.py \
      --checkpoint ./checkpoints/maithili_1_checkpoint.json \
      --input "/path/to/maithili-1.csv" \
      --concurrency 5
"""

import os, json, asyncio, argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

MODEL            = "gpt-4o-mini"
TEMPERATURE      = 0.1
RETRY_LIMIT      = 4
RETRY_BASE_DELAY = 5
CHECKPOINT_EVERY = 50
MAX_ARTICLE_CHARS = 6000

SENTIMENT_SYSTEM = (
    "You are an expert sentiment analyst specializing in multilingual Indian "
    "news content (Bhojpuri, Maithili, Hindi).\n\n"
    "Task: Classify the overall sentiment of the given news article.\n\n"
    "Rules:\n"
    "- Choose exactly one label: Positive, Negative, or Neutral\n"
    "- Positive  : article conveys good news, achievement, celebration, hope, or progress\n"
    "- Negative  : article conveys bad news, crime, disaster, conflict, grief, or criticism\n"
    "- Neutral   : article is factual/informational with no clear positive or negative tone\n"
    "- Respond with ONLY valid JSON, no markdown, no explanation\n\n"
    'Required format:\n{\n  "sentiment": "Positive"\n}'
)
SENTIMENT_USER     = "Classify the sentiment of this news article:\n\n{article_text}"
SENTIMENT_FALLBACK = "Neutral"

def is_sentiment_missing(v):
    s = v.get("sentiment", "").strip()
    return s == "" or s not in ("Positive", "Negative", "Neutral")

async def call_sentiment(client, semaphore, article_text: str) -> str:
    text = article_text[:MAX_ARTICLE_CHARS]
    delay = RETRY_BASE_DELAY
    async with semaphore:
        for attempt in range(RETRY_LIMIT):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": SENTIMENT_SYSTEM},
                        {"role": "user",   "content": SENTIMENT_USER.format(article_text=text)},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                s = parsed.get("sentiment", "").strip()
                if s in ("Positive", "Negative", "Neutral"):
                    return s
                print(f"  ⚠️  Unexpected sentiment value: {s!r}, using fallback")
                return SENTIMENT_FALLBACK
            except Exception as e:
                if attempt < RETRY_LIMIT - 1:
                    print(f"  Retry {attempt+1}/{RETRY_LIMIT} — {e}")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    print(f"  ❌ Permanent failure after {RETRY_LIMIT} attempts: {e}")
                    return SENTIMENT_FALLBACK

async def main_async(checkpoint_path: Path, csv_path: str, api_key: str, concurrency: int):
    # Load checkpoint
    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)

    # Load CSV for article text
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")

    # Detect article_text column
    text_col = None
    for candidate in ["article_text", "Article_Text", "text"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        print(f"ERROR: Could not find article_text column. Columns: {list(df.columns)}")
        return

    # Find rows needing sentiment patch
    to_patch = sorted(int(k) for k, v in data.items() if is_sentiment_missing(v))
    print(f"\n  Rows needing sentiment patch : {len(to_patch)}")
    print(f"  Concurrency                 : {concurrency}")

    if not to_patch:
        print("  ✅ Nothing to patch!")
        return

    client    = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    pbar      = tqdm(total=len(to_patch), desc="  Patching sentiment", unit="row")
    patched   = 0

    async def worker(idx: int):
        nonlocal patched
        if idx >= len(df):
            pbar.update(1)
            return
        article_text = str(df.at[idx, text_col])
        sentiment    = await call_sentiment(client, semaphore, article_text)
        data[str(idx)]["sentiment"] = sentiment
        patched += 1
        pbar.update(1)

        # Save checkpoint periodically
        if patched % CHECKPOINT_EVERY == 0:
            tmp = checkpoint_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(checkpoint_path)

    await asyncio.gather(*[worker(i) for i in to_patch])
    pbar.close()

    # Final save
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Patched {patched} rows")
    print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # Sentiment distribution
    sentiments = [data[str(i)].get("sentiment", "") for i in to_patch]
    from collections import Counter
    dist = Counter(sentiments)
    print(f"\n  Sentiment distribution (patched rows):")
    for label, count in dist.most_common():
        bar = "█" * int(count / len(sentiments) * 40)
        print(f"    {label:<10} {count:>5,}  ({count/len(sentiments)*100:5.1f}%)  {bar}")

def main():
    parser = argparse.ArgumentParser(description="Patch missing sentiment in checkpoint")
    parser.add_argument("--checkpoint",   required=True, help="Path to checkpoint JSON")
    parser.add_argument("--input",        required=True, help="Path to original CSV")
    parser.add_argument("--concurrency",  type=int, default=5)
    args    = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY environment variable first.")
        return
    asyncio.run(main_async(Path(args.checkpoint), args.input, api_key, args.concurrency))

if __name__ == "__main__":
    main()
