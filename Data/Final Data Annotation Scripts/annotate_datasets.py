"""
annotate_datasets.py  —  Maithra Annotation Pipeline
=====================================================
Two SEPARATE API calls per article:
  Call 1 -> Summarization  (plain text)
  Call 2 -> Ekman emotions (structured JSON)

Usage:
  export OPENAI_API_KEY="sk-..."
  python annotate_datasets.py --estimate --data-dir ./data
  python annotate_datasets.py --all --data-dir ./data
  python annotate_datasets.py --dataset maithili_1 --input /path/to/file.csv

Requirements:
  pip install openai pandas tqdm
"""

import os, json, asyncio, argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "bhojpuri_1": {
        "description": "Bhojpuri-1 (~4,982 articles)",
        "col_map": {
            "website": "website", "article_url": "article_url",
            "language": "language", "headline": "headline",
            "article_text": "article_text",
        },
    },
    "bhojpuri_2": {
        "description": "Bhojpuri-2 (~10,073 articles)",
        "col_map": {
            "website": "website", "article_url": "article_url",
            "language": "language", "article_text": "article_text",
        },
    },
    "maithili_1": {
        "description": "Maithili-1 (~5,574 articles)",
        "col_map": {
            "article_link": "article_url", "article_text": "article_text",
            "article_language": "language", "article_headline": "headline",
        },
    },
    "maithili_2": {
        "description": "Maithili-2 (~2,998 articles)",
        "col_map": {
            "website": "website", "article_url": "article_url",
            "language": "language", "article_text": "article_text",
        },
    },
}

STANDARD_COLS   = ["website", "article_url", "language", "headline", "article_text"]
EMOTION_KEYS    = ["ekman_joy","ekman_sadness","ekman_anger","ekman_fear",
                   "ekman_disgust","ekman_surprise","ekman_neutral"]
ANNOTATION_COLS = ["summary"] + EMOTION_KEYS + ["dominant_emotion", "sentiment"]

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL             = "gpt-4o-mini"
TEMPERATURE       = 0.1
CONCURRENCY       = 30
RETRY_LIMIT       = 4
RETRY_BASE_DELAY  = 5
CHECKPOINT_EVERY  = 50
MAX_ARTICLE_CHARS = 6000

OUTPUT_DIR     = Path("./output")
CHECKPOINT_DIR = Path("./checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  —  Call 1: Summarization
# ─────────────────────────────────────────────────────────────────────────────
SUMMARIZATION_SYSTEM = (
    "You are an expert multilingual summarizer specializing in Maithili and Bhojpuri.\n\n"
    "CRITICAL LANGUAGE RULES:\n"
    "- Maithili is NOT Hindi. They are different languages. "
    "Maithili uses words like: अछि, छथि, छलाह, केँ, सँ, मे, थिक, कयलनि, होएत, रहल अछि.\n"
    "- Bhojpuri is NOT Hindi. "
    "Bhojpuri uses words like: बा, बाड़े, रहल बा, कइलस, गइल, बतवलें, होई, रहल बाड़ें.\n"
    "- NEVER write the summary in Hindi. "
    "Hindi words like है, हैं, था, थे, किया, करेगा are FORBIDDEN.\n\n"
    "Task: Write a 3 to 5 sentence summary strictly in the same language as the article. "
    "Preserve key facts, actors, and events. "
    "Do not add information not present in the article.\n\n"
    "Respond with ONLY the summary text. No preamble, no labels, no explanation."
)
SUMMARIZATION_USER = (
    "Language: {language}\n"
    "Instruction: Summarize the article below in {language} ONLY. "
    "DO NOT use Hindi (है/हैं/था/किया are Hindi — do not use them). "
    "Write exactly as the article is written.\n\n"
    "Article:\n{article_text}\n\n"
    "Summary in {language}:"
)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  —  Call 2: Ekman Emotion Classification
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_SYSTEM = (
    "You are an expert in affective computing and emotion analysis for "
    "multilingual Indian news content (Bhojpuri, Maithili, Hindi).\n\n"
    "Task: Analyze the emotional tone of the given news article using "
    "Ekman's 7 basic emotion categories.\n\n"
    "Rules:\n"
    "- All scores must be floats between 0.0 and 1.0\n"
    "- All 7 scores must sum to exactly 1.0\n"
    "- dominant_emotion must match the category with the highest score\n"
    "- Respond with ONLY valid JSON, no markdown, no explanation\n\n"
    'Required format:\n'
    '{\n'
    '  "ekman_joy":      0.00,\n'
    '  "ekman_sadness":  0.00,\n'
    '  "ekman_anger":    0.00,\n'
    '  "ekman_fear":     0.00,\n'
    '  "ekman_disgust":  0.00,\n'
    '  "ekman_surprise": 0.00,\n'
    '  "ekman_neutral":  0.00,\n'
    '  "dominant_emotion": "neutral"\n'
    '}'
)
EMOTION_USER = "Analyze the emotional tone of this news article:\n\n{article_text}"

EMOTION_FALLBACK = {
    "ekman_joy": 0.0, "ekman_sadness": 0.0, "ekman_anger": 0.0,
    "ekman_fear": 0.0, "ekman_disgust": 0.0, "ekman_surprise": 0.0,
    "ekman_neutral": 1.0, "dominant_emotion": "neutral",
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  —  Call 3: Sentiment Classification
# ─────────────────────────────────────────────────────────────────────────────
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
    'Required format:\n'
    '{\n'
    '  "sentiment": "Positive"\n'
    '}'
)
SENTIMENT_USER = "Classify the sentiment of this news article:\n\n{article_text}"

SENTIMENT_FALLBACK = "Neutral"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_normalize(csv_path: str, dataset_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    df = df.rename(columns=DATASETS[dataset_key]["col_map"])
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[STANDARD_COLS].copy()
    before = len(df)
    df = df[df["article_text"].str.strip().astype(bool)].reset_index(drop=True)
    if len(df) < before:
        print(f"    Dropped {before - len(df)} rows with empty article_text")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(key: str) -> dict:
    p = CHECKPOINT_DIR / f"{key}_checkpoint.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        print(f"    Checkpoint: {len(data)} articles already done")
        return data
    return {}

def save_checkpoint(key: str, data: dict):
    p   = CHECKPOINT_DIR / f"{key}_checkpoint.json"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

# ─────────────────────────────────────────────────────────────────────────────
# API CALL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
async def api_call(client, system, user, row_idx, label):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=400,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            msg  = str(exc)
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < RETRY_LIMIT:
                tag = "Rate limit" if ("rate_limit" in msg.lower() or "429" in msg) else "Error"
                print(f"\n    [{label}] row {row_idx} {tag} — retry {attempt} in {wait}s")
                await asyncio.sleep(wait)
            else:
                print(f"\n    [{label}] row {row_idx} FAILED: {msg[:100]}")
                return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# CALL 1 — SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────────────
async def call_summarization(client, article_text: str, row_idx: int, language: str = "Maithili") -> str:
    user   = SUMMARIZATION_USER.format(
        article_text=article_text[:MAX_ARTICLE_CHARS],
        language=language,
    )
    result = await api_call(client, SUMMARIZATION_SYSTEM, user, row_idx, "SUMMARY")
    return result if result else ""

# ─────────────────────────────────────────────────────────────────────────────
# CALL 2 — EKMAN EMOTION
# ─────────────────────────────────────────────────────────────────────────────
def parse_emotion(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        end     = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end]).strip()
    try:
        parsed = json.loads(cleaned)
        if not all(k in parsed for k in EMOTION_KEYS + ["dominant_emotion"]):
            return EMOTION_FALLBACK
        total = sum(float(parsed[k]) for k in EMOTION_KEYS)
        if total <= 0:
            return EMOTION_FALLBACK
        for k in EMOTION_KEYS:
            parsed[k] = round(float(parsed[k]) / total, 4)
        parsed["dominant_emotion"] = max(
            EMOTION_KEYS, key=lambda k: parsed[k]
        ).replace("ekman_", "")
        return parsed
    except (json.JSONDecodeError, ValueError, KeyError):
        return EMOTION_FALLBACK

async def call_emotion(client, article_text: str, row_idx: int) -> dict:
    user   = EMOTION_USER.format(article_text=article_text[:MAX_ARTICLE_CHARS])
    result = await api_call(client, EMOTION_SYSTEM, user, row_idx, "EMOTION")
    if not result:
        return EMOTION_FALLBACK
    return parse_emotion(result)

# ─────────────────────────────────────────────────────────────────────────────
# CALL 3 — SENTIMENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def parse_sentiment(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        end     = -1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[1:end]).strip()
    try:
        parsed = json.loads(cleaned)
        label  = str(parsed.get("sentiment", "")).strip().capitalize()
        if label in ("Positive", "Negative", "Neutral"):
            return label
        return SENTIMENT_FALLBACK
    except (json.JSONDecodeError, ValueError, KeyError):
        return SENTIMENT_FALLBACK

async def call_sentiment(client, article_text: str, row_idx: int) -> str:
    user   = SENTIMENT_USER.format(article_text=article_text[:MAX_ARTICLE_CHARS])
    result = await api_call(client, SENTIMENT_SYSTEM, user, row_idx, "SENTIMENT")
    if not result:
        return SENTIMENT_FALLBACK
    return parse_sentiment(result)

# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE WORKER
# ─────────────────────────────────────────────────────────────────────────────
async def process_article(client, row_idx: int, article_text: str,
                           semaphore: asyncio.Semaphore, language: str = "Maithili"):
    async with semaphore:
        summary   = await call_summarization(client, article_text, row_idx, language)
        await asyncio.sleep(0.05)
        emotion   = await call_emotion(client, article_text, row_idx)
        await asyncio.sleep(0.05)
        sentiment = await call_sentiment(client, article_text, row_idx)
    # Return None on permanent failure so the row is NOT saved to checkpoint
    # and will be retried on the next run
    if summary == "" and emotion["ekman_neutral"] == 1.0 and emotion["dominant_emotion"] == "neutral":
        return None
    return {"summary": summary, **emotion, "sentiment": sentiment}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# INTERMEDIATE CSV WRITER
# Called every CHECKPOINT_EVERY articles so you always have a usable CSV
# even if the run is interrupted or credit runs out mid-way.
# File is written to output/<dataset_key>_annotated.csv (same as final output).
# ─────────────────────────────────────────────────────────────────────────────
def save_intermediate_csv(df: pd.DataFrame, checkpoint: dict, dataset_key: str):
    out_df = df.copy()
    for col in ANNOTATION_COLS:
        out_df[col] = "" if col in ("summary", "dominant_emotion", "sentiment") else 0.0

    for i in range(len(out_df)):
        res = checkpoint.get(str(i))
        if res:
            out_df.at[i, "summary"]          = res.get("summary", "")
            out_df.at[i, "ekman_joy"]        = res.get("ekman_joy", 0.0)
            out_df.at[i, "ekman_sadness"]    = res.get("ekman_sadness", 0.0)
            out_df.at[i, "ekman_anger"]      = res.get("ekman_anger", 0.0)
            out_df.at[i, "ekman_fear"]       = res.get("ekman_fear", 0.0)
            out_df.at[i, "ekman_disgust"]    = res.get("ekman_disgust", 0.0)
            out_df.at[i, "ekman_surprise"]   = res.get("ekman_surprise", 0.0)
            out_df.at[i, "ekman_neutral"]    = res.get("ekman_neutral", 0.0)
            out_df.at[i, "dominant_emotion"] = res.get("dominant_emotion", "neutral")
            out_df.at[i, "sentiment"]        = res.get("sentiment", SENTIMENT_FALLBACK)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_key}_annotated.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    done = sum(1 for i in range(len(out_df)) if str(i) in checkpoint)
    print(f"\n    [CSV] Intermediate save: {done:,}/{len(out_df):,} rows → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
async def process_dataset(dataset_key: str, csv_path: str, api_key: str):
    print(f"\n{'='*65}")
    print(f"  {DATASETS[dataset_key]['description']}")
    print(f"  Input : {csv_path}")
    print(f"{'='*65}")

    df         = load_and_normalize(csv_path, dataset_key)
    checkpoint = load_checkpoint(dataset_key)
    pending    = [i for i in range(len(df)) if str(i) not in checkpoint]

    print(f"    Total    : {len(df):,}")
    print(f"    Done     : {len(checkpoint):,}")
    print(f"    Pending  : {len(pending):,}")

    if pending:
        client    = AsyncOpenAI(api_key=api_key)
        semaphore = asyncio.Semaphore(CONCURRENCY)
        pbar      = tqdm(total=len(pending), desc=f"  {dataset_key}", unit="art")
        batch     = {}

        async def worker(idx: int):
            result = await process_article(client, idx,
                                           df.at[idx, "article_text"],
                                           semaphore,
                                           language=df.at[idx, "language"])
            if result is None:
                # Permanent API failure — do NOT checkpoint so it retries next run
                pbar.update(1)
                return
            batch[str(idx)] = result
            pbar.update(1)
            if len(batch) % CHECKPOINT_EVERY == 0:
                merged = {**checkpoint, **batch}
                save_checkpoint(dataset_key, merged)
                save_intermediate_csv(df, merged, dataset_key)

        await asyncio.gather(*[worker(i) for i in pending])
        pbar.close()
        checkpoint.update(batch)
        save_checkpoint(dataset_key, checkpoint)
        print(f"    Done. Checkpoint saved.")

    # Build output CSV
    for col in ANNOTATION_COLS:
        df[col] = "" if col in ("summary", "dominant_emotion", "sentiment") else 0.0

    missing = 0
    for i in range(len(df)):
        res = checkpoint.get(str(i))
        if res:
            df.at[i, "summary"]          = res.get("summary", "")
            df.at[i, "ekman_joy"]        = res.get("ekman_joy", 0.0)
            df.at[i, "ekman_sadness"]    = res.get("ekman_sadness", 0.0)
            df.at[i, "ekman_anger"]      = res.get("ekman_anger", 0.0)
            df.at[i, "ekman_fear"]       = res.get("ekman_fear", 0.0)
            df.at[i, "ekman_disgust"]    = res.get("ekman_disgust", 0.0)
            df.at[i, "ekman_surprise"]   = res.get("ekman_surprise", 0.0)
            df.at[i, "ekman_neutral"]    = res.get("ekman_neutral", 0.0)
            df.at[i, "dominant_emotion"] = res.get("dominant_emotion", "neutral")
            df.at[i, "sentiment"]        = res.get("sentiment", SENTIMENT_FALLBACK)
        else:
            missing += 1

    if missing:
        print(f"    {missing} articles missing from checkpoint (left blank)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_key}_annotated.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    Saved  : {out_path}  ({len(df):,} rows)")

    dist = df["dominant_emotion"].value_counts()
    print(f"\n    Emotion distribution:")
    for emotion, count in dist.items():
        bar = "█" * int(count / len(df) * 40)
        print(f"      {emotion:<12} {count:>6,}  ({count/len(df)*100:5.1f}%)  {bar}")

    sdist = df["sentiment"].value_counts()
    print(f"\n    Sentiment distribution:")
    for label, count in sdist.items():
        bar = "█" * int(count / len(df) * 40)
        print(f"      {label:<12} {count:>6,}  ({count/len(df)*100:5.1f}%)  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# COST ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def print_cost_estimate(dataset_paths: dict):
    INPUT_PRICE, OUTPUT_PRICE = 0.15, 0.60   # per 1M tokens
    PROMPT_OH = 200   # system+user template overhead per call (tokens)

    print(f"\n{'='*65}")
    print(f"  COST ESTIMATE  |  {MODEL}")
    print(f"  Input ${INPUT_PRICE}/1M tokens  |  Output ${OUTPUT_PRICE}/1M tokens")
    print(f"  Three separate calls per article (summarization + emotion + sentiment)")
    print(f"{'='*65}")
    print(f"  {'Dataset':<14} {'Articles':>8} {'Input M':>9} {'Output M':>9} {'Cost $':>8}")
    print(f"  {'-'*53}")

    grand, grand_n = 0.0, 0
    for key, path in dataset_paths.items():
        if not Path(path).exists():
            print(f"  {key:<14}  not found: {path}")
            continue
        df  = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lstrip("\ufeff")
        df  = df.rename(columns=DATASETS[key]["col_map"])
        if "article_text" not in df.columns:
            continue
        n         = len(df)
        avg_tok   = df["article_text"].dropna().str.len().mean() / 4
        in_per    = (avg_tok + PROMPT_OH) * 3   # three calls
        out_per   = 150 + 80 + 10               # summary + emotion JSON + sentiment JSON
        in_M      = n * in_per  / 1_000_000
        out_M     = n * out_per / 1_000_000
        cost      = in_M * INPUT_PRICE + out_M * OUTPUT_PRICE
        grand    += cost
        grand_n  += n
        print(f"  {key:<14} {n:>8,} {in_M:>9.3f} {out_M:>9.3f} {cost:>8.3f}")

    print(f"  {'-'*53}")
    print(f"  {'TOTAL':<14} {grand_n:>8,} {'':>9} {'':>9} {grand:>8.3f}")
    print(f"\n  Batch API (50% off, 24h delay) : ${grand/2:.3f}")
    print(f"  Your $5 OpenAI credit covers the full run.\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global CONCURRENCY
    parser = argparse.ArgumentParser(
        description="Maithra annotation pipeline: summarization + Ekman emotions"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",      action="store_true",
        help="Process all 4 datasets from --data-dir")
    mode.add_argument("--dataset",  choices=list(DATASETS.keys()),
        help="Process one dataset")
    mode.add_argument("--estimate", action="store_true",
        help="Print cost estimate and exit")
    parser.add_argument("--input",    type=str, help="CSV path (with --dataset)")
    parser.add_argument("--data-dir", type=str, default="./data",
        help="Folder with CSVs (with --all, default: ./data)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args    = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    data_dir = Path(args.data_dir)

    if args.estimate:
        print_cost_estimate({k: str(data_dir / f"{k.replace('_','-')}.csv") for k in DATASETS})
        return

    if not api_key:
        print("ERROR: set OPENAI_API_KEY environment variable first.")
        return

    if args.all:
        jobs = [(k, str(data_dir / f"{k.replace('_','-')}.csv")) for k in DATASETS
                if (data_dir / f"{k.replace('_','-')}.csv").exists()]
        missing = [k for k in DATASETS if not (data_dir / f"{k.replace('_','-')}.csv").exists()]
        if missing:
            print(f"  Skipping (not found): {missing}")
    else:
        if not args.input:
            print("ERROR: --input required with --dataset")
            return
        jobs = [(args.dataset, args.input)]

    if not jobs:
        print("No datasets found. Check --data-dir.")
        return

    CONCURRENCY = args.concurrency
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print(f"\n  Model       : {MODEL}")
    print(f"  Concurrency : {CONCURRENCY}")
    print(f"  Datasets    : {len(jobs)}")

    for dataset_key, csv_path in jobs:
        asyncio.run(process_dataset(dataset_key, csv_path, api_key))

    print(f"\nAll done. Outputs in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()