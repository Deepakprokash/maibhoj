# Affective Preservation Evaluation (APE)

How faithfully the **emotion (7-class Ekman)** and **sentiment (3-class)** of a source
article are retained in a model-**generated headline** and **generated summary**.
Preservation = agreement between the classifier's prediction on the generated output and
the gold **article-level** label. Two tasks (headline, summary) × two languages
(218 Bhojpuri, 201 Maithili), five classifiers.

## Files
- `source_affect_classification.py` — classifier engine; `--task headline|summary` sets the prompt noun
- `run_ape_rotate.sh` — Groq runner: `./run_ape_rotate.sh <model> <headline|summary>`; **insert your own GROQ keys**
- `aggregate_ape.py` — rebuilds `ape_comparison_full.csv` + `ape_per_class_full.csv`
- `ape_headline_{bho,mai}.csv`, `ape_summary_{bho,mai}.csv` — inputs: generated text + gold labels
- `APE_gemma3_27b.ipynb` — executed local-GPU notebook (runs both tasks)
- `ape_headline_results/`, `ape_summary_results/` — per-model metrics for all 5 classifiers
- `ape_comparison_full.csv`, `ape_per_class_full.csv` — combined tables

## Reproduce
Groq model: `./run_ape_rotate.sh "openai/gpt-oss-120b" headline` (and `summary`).
Local-GPU model (Gemma): run `APE_gemma3_27b.ipynb` (4-bit, 24 GB) → `ape_bundle_*.json`.
Combine: `python3 aggregate_ape.py`.

The APE prompt is the source/gold-annotation prompt with the noun changed from *article* to
*headline*/*summary*, so any divergence reflects affective drift from generation, not prompt changes.

## Install
`pip install openai pandas tqdm scikit-learn numpy`
(notebook also needs `transformers accelerate bitsandbytes torch`)
