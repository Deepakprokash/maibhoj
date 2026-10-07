"""
source_affect_classification.py — Zero-shot SOURCE-ARTICLE affect classification
================================================================================
Reviewer 3 asked for a conventional benchmark table: how well do the five
baseline models classify the SOURCE ARTICLE's emotion (7-class Ekman) and
sentiment (3-class) *directly* — not the generated headline/summary.

This is the sibling of affective_preservation_groq.py. The ONLY change in the
task is the input text: here the model classifies `article_text` (the source
article) and we score the prediction against the gold article-level labels
(`dominant_emotion`, `sentiment`, `ekman_*`). Same models, same prompts, same
metrics — so the numbers are directly comparable to the preservation tables.

Providers are MIXED per model. Each --models entry may carry a provider prefix:
    groq:openai/gpt-oss-20b
    openrouter:meta-llama/llama-3.3-70b-instruct
An entry with no prefix uses --provider (default: groq). This lets you serve
whatever each provider still offers — e.g. gpt-oss free on Groq, the retired
Llama/Qwen models on OpenRouter.

Keys (set whichever providers you use):
    export GROQ_API_KEY="gsk_..."
    export OPENROUTER_API_KEY="sk-or-..."

Install:
    pip install openai pandas tqdm scikit-learn numpy

Examples:
    # everything on OpenRouter
    python source_affect_classification.py --all \\
        --bhojpuri gold_bhojpuri.csv --maithili gold_maithili.csv \\
        --provider openrouter \\
        --models meta-llama/llama-4-scout-17b-16e-instruct \\
                 meta-llama/llama-3.3-70b-instruct \\
                 qwen/qwen3-32b openai/gpt-oss-20b \\
                 meta-llama/llama-3.1-8b-instruct

    # mixed: gpt-oss free on Groq, the rest on OpenRouter
    python source_affect_classification.py --all \\
        --bhojpuri gold_bhojpuri.csv --maithili gold_maithili.csv \\
        --models groq:openai/gpt-oss-20b \\
                 openrouter:meta-llama/llama-3.3-70b-instruct \\
                 openrouter:qwen/qwen3-32b

    # one model, one language
    python source_affect_classification.py --name bhojpuri --input gold_bhojpuri.csv \\
        --models openrouter:meta-llama/llama-3.3-70b-instruct
"""

import os, json, asyncio, argparse, math, re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDERS  (OpenAI-compatible endpoints)
# ─────────────────────────────────────────────────────────────────────────────
PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1",       "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",         "key_env": "OPENROUTER_API_KEY"},
    "nvidia":     {"base_url": "https://integrate.api.nvidia.com/v1",  "key_env": "NVIDIA_API_KEY"},
    "sambanova":  {"base_url": "https://api.sambanova.ai/v1",          "key_env": "SAMBANOVA_API_KEY"},
}
DEFAULT_PROVIDER = "groq"

# Default model list is provider-agnostic (no prefix -> uses --provider).
# NOTE: on Groq free tier, 4 of these 5 retired in Jul-Aug 2026 — serve those on
# OpenRouter. IDs differ slightly across providers (see README).
DEFAULT_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

TEMPERATURE      = 0.1
MAX_TOKENS       = 2048     # headroom so reasoning models don't truncate before JSON
CONCURRENCY      = 3
RETRY_LIMIT      = 5
RETRY_BASE_DELAY = 5
CHECKPOINT_EVERY = 25
MAX_TEXT_CHARS   = 4000
MAX_SECONDS      = 0        # >0: stop launching new rows after this many seconds

EMOTION_KEYS = ["ekman_joy", "ekman_sadness", "ekman_anger", "ekman_fear",
                "ekman_disgust", "ekman_surprise", "ekman_neutral"]
EMOTIONS     = [k.replace("ekman_", "") for k in EMOTION_KEYS]
SENTIMENTS   = ["positive", "negative", "neutral"]

TEXT_COL      = "article_text"     # source text to classify
GOLD_EMO_COL  = "dominant_emotion" # gold article emotion
GOLD_SENT_COL = "sentiment"        # gold article sentiment

OUTPUT_DIR     = Path("./source_classification_results")
CHECKPOINT_DIR = Path("./source_classification_checkpoints")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  — identical to the gold-annotation / preservation prompts, targeting
# the news article (system prompts already say "news article"; user line too).
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
EMOTION_USER = "Analyze the emotional tone of this news {noun}:\n\n{text}"

SENTIMENT_SYSTEM = (
    "You are an expert sentiment analyst specializing in multilingual Indian "
    "news content (Bhojpuri, Maithili, Hindi).\n\n"
    "Task: Classify the overall sentiment of the given news article.\n\n"
    "Rules:\n"
    "- Choose exactly one label: Positive, Negative, or Neutral\n"
    "- Positive  : good news, achievement, celebration, hope, or progress\n"
    "- Negative  : bad news, crime, disaster, conflict, grief, or criticism\n"
    "- Neutral   : factual/informational with no clear positive or negative tone\n"
    "- Respond with ONLY valid JSON, no markdown, no explanation\n\n"
    'Required format:\n{\n  "sentiment": "Positive"\n}'
)
SENTIMENT_USER = "Classify the sentiment of this news {noun}:\n\n{text}"

# task -> the noun used in the USER prompt (system prompt stays "news article",
# matching the original APE pipeline where only the user line's noun changed).
TASK_NOUN = "article"   # set from --task: article | headline | summary

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER / MODEL SPEC
# ─────────────────────────────────────────────────────────────────────────────
def parse_model_spec(spec: str, default_provider: str):
    """'openrouter:meta-llama/llama-3.3-70b-instruct' -> ('openrouter', 'meta-llama/...').
    A bare id (no known provider prefix) -> (default_provider, spec)."""
    if ":" in spec:
        head, rest = spec.split(":", 1)
        if head in PROVIDERS:
            return head, rest
    return default_provider, spec


def make_client(provider: str) -> AsyncOpenAI:
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"], "")
    if not key:
        raise SystemExit(f"ERROR: set {cfg['key_env']} to use provider '{provider}'.")
    headers = {}
    if provider == "openrouter":
        # optional but recommended by OpenRouter; harmless elsewhere
        headers = {"HTTP-Referer": "https://maibhoj-nlp.local",
                   "X-Title": "MaiBhoj-NLP source affect classification"}
    return AsyncOpenAI(api_key=key, base_url=cfg["base_url"], default_headers=headers)


def _provider_params(provider: str, model: str):
    """Return (extra_kwargs, extra_body) to keep reasoning models terse enough to
    emit JSON. Tolerant: if a provider ignores these, _extract_json still copes."""
    m = model.lower()
    if provider == "groq":
        if "qwen3" in m:   return {"reasoning_effort": "none"}, {}
        if "gpt-oss" in m: return {"reasoning_effort": "low"}, {}
        return {}, {}
    if provider == "openrouter":
        if "gpt-oss" in m: return {}, {"reasoning": {"effort": "low"}}
        if "qwen3" in m:   return {}, {"reasoning": {"enabled": False}}
        return {}, {}
    return {}, {}

# ─────────────────────────────────────────────────────────────────────────────
# API + PARSERS
# ─────────────────────────────────────────────────────────────────────────────
async def api_call(client, provider, model, system, user, row_idx):
    extra, extra_body = _provider_params(provider, model)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = await client.chat.completions.create(
                model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
                extra_body=extra_body or None, **extra,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception as exc:
            msg  = str(exc)
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            if attempt < RETRY_LIMIT:
                tag = "rate-limit" if ("429" in msg or "rate_limit" in msg.lower()) else "error"
                print(f"\n    [{provider}:{model}] row {row_idx} {tag} — retry {attempt} in {wait}s")
                await asyncio.sleep(wait)
            else:
                print(f"\n    [{provider}:{model}] row {row_idx} FAILED: {msg[:90]}")
                return None
    return None


def _extract_json(raw: str, hint: str = "") -> str:
    s = raw.strip()
    s = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"```(?:json)?", " ", s)
    candidates = re.findall(r"\{[^{}]*\}", s, re.DOTALL)
    if hint:
        for c in candidates:
            if hint in c:
                return c
    if candidates:
        return candidates[-1]
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group(0) if m else s


def parse_emotion(raw):
    if not raw:
        return None
    try:
        parsed = json.loads(_extract_json(raw, hint="ekman_joy"))
        if not all(k in parsed for k in EMOTION_KEYS):
            return None
        total = sum(float(parsed[k]) for k in EMOTION_KEYS)
        if total <= 0:
            dom = str(parsed.get("dominant_emotion", "")).strip().lower()
            if dom in EMOTIONS:
                oh = {k: 0.0 for k in EMOTION_KEYS}
                oh[f"ekman_{dom}"] = 1.0
                oh["dominant_emotion"] = dom
                return oh
            return None
        for k in EMOTION_KEYS:
            parsed[k] = round(float(parsed[k]) / total, 4)
        parsed["dominant_emotion"] = max(EMOTION_KEYS, key=lambda k: parsed[k]).replace("ekman_", "")
        return {k: parsed[k] for k in EMOTION_KEYS + ["dominant_emotion"]}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


def parse_sentiment(raw):
    if not raw:
        return None
    try:
        label = str(json.loads(_extract_json(raw, hint="sentiment")).get("sentiment", "")).strip().lower()
        return label if label in SENTIMENTS else None
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None


_DEADLINE = 0.0

async def classify_article(client, provider, model, text, row_idx, semaphore):
    snippet = str(text)[:MAX_TEXT_CHARS]
    async with semaphore:
        if _DEADLINE and asyncio.get_event_loop().time() > _DEADLINE:
            return None
        emo_raw  = await api_call(client, provider, model, EMOTION_SYSTEM,
                                  EMOTION_USER.format(noun=TASK_NOUN, text=snippet), row_idx)
        await asyncio.sleep(0.05)
        sent_raw = await api_call(client, provider, model, SENTIMENT_SYSTEM,
                                  SENTIMENT_USER.format(noun=TASK_NOUN, text=snippet), row_idx)
    emo  = parse_emotion(emo_raw)
    sent = parse_sentiment(sent_raw)
    if emo is None or sent is None:
        return None
    emo["pred_sentiment"] = sent
    return emo

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def js_divergence(p, q):
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    q = np.clip(q, 1e-12, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def slice_tag(provider, model):
    return f"{provider}__{safe_name(model)}"


def ckpt_path(lang, provider, model):
    return CHECKPOINT_DIR / f"{lang}__{slice_tag(provider, model)}.json"


def load_checkpoint(lang, provider, model):
    p = ckpt_path(lang, provider, model)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(lang, provider, model, data):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    p   = ckpt_path(lang, provider, model)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)

# ─────────────────────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────────────────────
async def classify_all(client, lang, provider, model, df):
    checkpoint = load_checkpoint(lang, provider, model)
    pending    = [i for i in range(len(df)) if str(i) not in checkpoint]
    print(f"    [{provider}:{model}] total {len(df)} | done {len(checkpoint)} | pending {len(pending)}")
    if not pending:
        return checkpoint

    semaphore = asyncio.Semaphore(CONCURRENCY)
    pbar      = tqdm(total=len(pending), desc=f"  {lang}/{safe_name(model)[:20]}", unit="art")
    batch     = {}
    global _DEADLINE
    _DEADLINE = (asyncio.get_event_loop().time() + MAX_SECONDS) if MAX_SECONDS else 0.0

    async def worker(idx):
        pred = await classify_article(client, provider, model, df.at[idx, TEXT_COL], idx, semaphore)
        pbar.update(1)
        if pred is None:
            return
        batch[str(idx)] = pred
        if len(batch) % CHECKPOINT_EVERY == 0:
            save_checkpoint(lang, provider, model, {**checkpoint, **batch})

    await asyncio.gather(*[worker(i) for i in pending])
    pbar.close()
    checkpoint.update(batch)
    save_checkpoint(lang, provider, model, checkpoint)
    failed = len(df) - len(checkpoint)
    if failed:
        print(f"    [{provider}:{model}] {failed} rows still failed/pending — rerun to retry them")
    return checkpoint


def build_report(lang, provider, model, df, preds):
    from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                                 precision_score, recall_score,
                                 confusion_matrix, classification_report)
    out = df.copy()
    g_emo, p_emo, js_list = [], [], []
    g_sent, p_sent = [], []

    n_missing = 0
    for i in range(len(out)):
        if str(i) not in preds:
            n_missing += 1
            continue
        pr   = preds[str(i)]
        ge   = str(out.at[i, GOLD_EMO_COL]).strip().lower()
        pe   = pr["dominant_emotion"]
        gvec = np.array([float(out.at[i, k]) if k in out.columns else 0.0 for k in EMOTION_KEYS])
        pvec = np.array([pr[k] for k in EMOTION_KEYS])
        js   = js_divergence(gvec, pvec) if gvec.sum() > 0 else float("nan")

        gs = str(out.at[i, GOLD_SENT_COL]).strip().lower() if GOLD_SENT_COL in out.columns else ""
        ps = pr.get("pred_sentiment", "")

        for k in EMOTION_KEYS:
            out.at[i, f"pred_{k}"] = pr[k]
        out.at[i, "pred_emotion"]      = pe
        out.at[i, "emotion_correct"]   = (pe == ge)
        out.at[i, "js_divergence"]     = round(js, 4) if not math.isnan(js) else ""
        out.at[i, "pred_sentiment"]    = ps
        out.at[i, "sentiment_correct"] = (ps == gs)

        g_emo.append(ge); p_emo.append(pe)
        if not math.isnan(js): js_list.append(js)
        if gs in SENTIMENTS:
            g_sent.append(gs); p_sent.append(ps)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_DIR / f"{lang}__{slice_tag(provider, model)}__rows.csv",
               index=False, encoding="utf-8-sig")

    if not g_emo:
        print(f"    [{provider}:{model}] {lang}: 0 scored rows ({n_missing} missing) — skipping report")
        return {"language": lang, "provider": provider, "model": model,
                "n_scored": 0, "n_missing": n_missing}, None, None

    emo_labels = sorted(set(g_emo) | set(p_emo))
    row = {
        "language": lang, "provider": provider, "model": model,
        "n_scored": len(g_emo), "n_missing": n_missing,
        "emo_acc": accuracy_score(g_emo, p_emo),
        "emo_macro_precision": precision_score(g_emo, p_emo, average="macro", labels=emo_labels, zero_division=0),
        "emo_macro_recall": recall_score(g_emo, p_emo, average="macro", labels=emo_labels, zero_division=0),
        "emo_macro_f1": f1_score(g_emo, p_emo, average="macro", labels=emo_labels, zero_division=0),
        "emo_weighted_f1": f1_score(g_emo, p_emo, average="weighted", labels=emo_labels, zero_division=0),
        "emo_kappa": cohen_kappa_score(g_emo, p_emo),
        "emo_mean_js": float(np.mean(js_list)) if js_list else None,
    }
    if g_sent:
        row.update({
            "sent_acc": accuracy_score(g_sent, p_sent),
            "sent_macro_precision": precision_score(g_sent, p_sent, average="macro", labels=SENTIMENTS, zero_division=0),
            "sent_macro_recall": recall_score(g_sent, p_sent, average="macro", labels=SENTIMENTS, zero_division=0),
            "sent_macro_f1": f1_score(g_sent, p_sent, average="macro", labels=SENTIMENTS, zero_division=0),
            "sent_weighted_f1": f1_score(g_sent, p_sent, average="weighted", labels=SENTIMENTS, zero_division=0),
            "sent_kappa": cohen_kappa_score(g_sent, p_sent),
        })

    detail = dict(row)
    detail["emotion_confusion_matrix"] = {
        "labels": emo_labels,
        "matrix": confusion_matrix(g_emo, p_emo, labels=emo_labels).tolist(),
    }
    emo_cls = classification_report(g_emo, p_emo, labels=emo_labels, zero_division=0, output_dict=True)
    detail["emotion_per_class"] = emo_cls
    sent_cls = None
    if g_sent:
        detail["sentiment_confusion_matrix"] = {
            "labels": SENTIMENTS,
            "matrix": confusion_matrix(g_sent, p_sent, labels=SENTIMENTS).tolist(),
        }
        sent_cls = classification_report(g_sent, p_sent, labels=SENTIMENTS, zero_division=0, output_dict=True)
        detail["sentiment_per_class"] = sent_cls
    with open(OUTPUT_DIR / f"{lang}__{slice_tag(provider, model)}__metrics.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    print(f"    [{provider}:{model}]  emo_acc={row['emo_acc']:.3f} "
          f"emo_F1={row['emo_macro_f1']:.3f} "
          f"sent_acc={row.get('sent_acc', float('nan')):.3f} "
          f"sent_F1={row.get('sent_macro_f1', float('nan')):.3f}")

    per_class_rows = _per_class_rows(lang, provider, model, emo_cls, sent_cls)
    return row, detail, per_class_rows


def _per_class_rows(lang, provider, model, emo_cls, sent_cls):
    rows = []
    for axis, rep in (("emotion", emo_cls), ("sentiment", sent_cls)):
        if not rep:
            continue
        for label, m in rep.items():
            if not isinstance(m, dict):   # skip 'accuracy' scalar
                continue
            rows.append({
                "language": lang, "provider": provider, "model": model,
                "axis": axis, "class": label,
                "precision": m.get("precision"), "recall": m.get("recall"),
                "f1": m.get("f1-score"), "support": m.get("support"),
            })
    return rows


async def run(jobs, model_specs, default_provider):
    clients = {}   # provider -> client (lazily created)
    summary, per_class_all = [], []
    for lang, path in jobs:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lstrip("﻿")
        for need in (TEXT_COL, GOLD_EMO_COL):
            if need not in df.columns:
                raise SystemExit(f"ERROR: '{need}' missing in {path}. Found: {list(df.columns)}")
        print(f"\n{'='*66}\n  {lang.upper()}  |  {path}  ({len(df)} rows)\n{'='*66}")
        for spec in model_specs:
            provider, model = parse_model_spec(spec, default_provider)
            if provider not in clients:
                clients[provider] = make_client(provider)
            preds = await classify_all(clients[provider], lang, provider, model, df)
            row, _detail, per_class = build_report(lang, provider, model, df, preds)
            summary.append(row)
            if per_class:
                per_class_all.extend(per_class)

    OUTPUT_DIR.mkdir(exist_ok=True)
    comp = pd.DataFrame(summary)
    comp.to_csv(OUTPUT_DIR / "comparison_full.csv", index=False, encoding="utf-8-sig")
    if per_class_all:
        pd.DataFrame(per_class_all).to_csv(
            OUTPUT_DIR / "per_class_full.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n{'='*66}\n  SOURCE-ARTICLE CLASSIFICATION (model vs gold article label)\n{'='*66}")
    cols = [c for c in ["language", "provider", "model", "n_scored", "n_missing",
                        "emo_acc", "emo_macro_f1", "emo_kappa", "emo_mean_js",
                        "sent_acc", "sent_macro_f1", "sent_kappa"]
            if c in comp.columns]
    print(comp[cols].round(4).to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR/'comparison_full.csv'} + per_class_full.csv "
          f"+ per-slice rows/metrics in {OUTPUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global CONCURRENCY, TEXT_COL, CHECKPOINT_DIR, OUTPUT_DIR, CHECKPOINT_EVERY, MAX_SECONDS, MAX_TOKENS, TASK_NOUN
    ap = argparse.ArgumentParser(description="Zero-shot source-article affect classification (Groq/OpenRouter, mixed)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="run both languages")
    mode.add_argument("--name", help="single-run label, e.g. bhojpuri")
    ap.add_argument("--input",    help="gold csv with article_text + labels (with --name)")
    ap.add_argument("--bhojpuri", help="Bhojpuri gold csv")
    ap.add_argument("--maithili", help="Maithili gold csv")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="model ids, optionally prefixed 'groq:' / 'openrouter:'")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=list(PROVIDERS),
                    help="default provider for unprefixed model ids")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--text-col", default=TEXT_COL, help="source text column (default article_text)")
    ap.add_argument("--task", default="article", choices=["article", "headline", "summary"],
                    help="noun in the USER prompt: article (source classification) | headline | summary (APE)")
    ap.add_argument("--ckpt-dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--max-seconds", type=int, default=MAX_SECONDS, help=">0 self-timeout per invocation")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if args.all:
        jobs = [(l, p) for l, p in [("bhojpuri", args.bhojpuri), ("maithili", args.maithili)] if p]
        if not jobs:
            raise SystemExit("Provide --bhojpuri and/or --maithili with --all.")
    else:
        if not args.input:
            raise SystemExit("--input required with --name")
        jobs = [(args.name, args.input)]

    CONCURRENCY = args.concurrency
    TEXT_COL = args.text_col
    TASK_NOUN = args.task
    CHECKPOINT_DIR = Path(args.ckpt_dir)
    OUTPUT_DIR = Path(args.out_dir)
    CHECKPOINT_EVERY = args.checkpoint_every
    MAX_SECONDS = args.max_seconds
    MAX_TOKENS = args.max_tokens
    OUTPUT_DIR.mkdir(exist_ok=True); CHECKPOINT_DIR.mkdir(exist_ok=True)

    resolved = [parse_model_spec(s, args.provider) for s in args.models]
    print(f"  Default provider : {args.provider}")
    print(f"  Models           : " + ", ".join(f"{p}:{m}" for p, m in resolved))
    print(f"  Concurrency      : {CONCURRENCY} | text-col: {TEXT_COL} | max_tokens: {MAX_TOKENS}")
    print(f"  Checkpoints      : {CHECKPOINT_DIR} | Output: {OUTPUT_DIR}")
    asyncio.run(run(jobs, args.models, args.provider))


if __name__ == "__main__":
    main()
