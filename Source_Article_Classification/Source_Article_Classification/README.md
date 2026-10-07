# Source-Article Affective Classification

Zero-shot **emotion (7-class Ekman)** and **sentiment (3-class)** classification of the
**source news article** (Bhojpuri + Maithili), scored against the gold article labels.
Five classifiers: GPT-OSS-20B, GPT-OSS-120B (Groq); Qwen3-32B, Gemma-3-27B, LLaMA-3.1-8B (local GPU).

## Files
- `source_affect_classification.py` — classifier engine (OpenAI-compatible: Groq / OpenRouter)
- `run_model_rotate.sh` — Groq runner; **insert your own GROQ keys** (placeholders inside)
- `aggregate_all.py` — rebuilds `comparison_full.csv` + `per_class_full.csv`
- `gold_bhojpuri.csv`, `gold_maithili.csv` — inputs: `article_text` + gold labels
- `*_local_gpu.ipynb`, `SOURCE_gemma3_27b.ipynb` — executed local-GPU notebooks (one per model)
- `source_classification_results/` — per-model metrics, model bundles, and the combined tables

## Reproduce
Groq models: `./run_model_rotate.sh "openai/gpt-oss-20b"` (and `openai/gpt-oss-120b`).
Local-GPU models: run the matching notebook (4-bit, 24 GB), producing a `*_bundle.json`.
Combine everything (no API calls): `python3 aggregate_all.py`.

## Install
`pip install openai pandas tqdm scikit-learn numpy`
(notebooks also need `transformers accelerate bitsandbytes torch`)
