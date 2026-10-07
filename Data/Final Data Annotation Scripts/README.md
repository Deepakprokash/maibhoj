# Final Data Annotation Scripts

Annotation pipeline for the **Maithili & Bhojpuri news corpus**. Takes the scraped article
CSVs and adds, per article: a native-language `summary`, seven **Ekman** emotion scores
(`ekman_joy … ekman_neutral`), a `dominant_emotion`, and a `sentiment`
(Positive / Negative / Neutral).

## Scripts
- **`annotate_datasets.py`** — main pipeline (OpenAI `gpt-4o-mini`). Three API calls per
  article (summarize → Ekman emotions → sentiment), run async with checkpointing so runs
  are resumable. Normalizes each dataset's columns to a standard schema before annotating.
- **`patch_sentiment.py`** — cost-saver that fills in only the missing `sentiment` label on
  rows that already have a good summary + emotion (avoids re-running the other two calls).

## Annotated datasets (`annotated_datasets/`)

| File | Contents | Rows |
|---|---|---|
| `bhojpuri_1_annotated.csv` | full khabarbhojpuri set with annotation columns (blanks where not yet annotated) | 4,982 |
| `maithili_1_annotated.csv` | full maithilijindabaad set with annotation columns (blanks where not yet annotated) | 5,573 |
| `good_rows_bhojpuri_1.csv` | only fully-clean annotated rows (deliverable) | 2,178 |
| `good_rows_maithili_1.csv` | only fully-clean annotated rows (deliverable) | 2,022 |

The `good_rows_*` files are the export of fully-complete rows (valid summary + non-fallback
emotion + valid sentiment); the `*_annotated` files keep every source row and leave
annotation cells blank where a row has not been processed.

## Usage
```bash
export OPENAI_API_KEY="sk-..."
python annotate_datasets.py --estimate --data-dir /path/to/csvs   # cost estimate
python annotate_datasets.py --dataset bhojpuri_1 --input bhojpuri-1.csv
```
Input CSVs (bhojpuri-1/2, maithili-1/2) come from the scraping repo.
