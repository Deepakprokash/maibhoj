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

| File | Contents | Rows | link |
|---|---|---|---|
|bhojpuri_2_annotated.csv| (full anjoria bhojpuri set with annotation| 10052 | https://docs.google.com/spreadsheets/d/10Q-CuYcFbeIOWIxxiaHSj1qII0WPdiW09dQssCzj4FY/edit?usp=sharing |
| `bhojpuri_1_annotated.csv` | full set with annotation from khabarbhojpuri (blanks where not yet annotated) | 4,982 | https://drive.google.com/file/d/1u71ImbHg6QNwdB1f8Bjf98q-pj9BWPoj/view?usp=sharing |
| `maithili_1_annotated.csv` | full set with annotation from maithilijindabaad (blanks where not yet annotated) | 5,573 | https://docs.google.com/spreadsheets/d/16Hj37XCSfXEltQAJ-4ApUY4GbfDWBUC2lTOlcGDePbU/edit?usp=sharing |
|maithili_2_annotated.csv| full set with annotation from esamaad | 2997 | https://drive.google.com/file/d/1bkmQ-XqAqJfvFGE1nxUVw9I7OCmiETXF/view?usp=sharing |
| `test_rows_bhojpuri_1.csv` | test data bhojpuri  | 2,178 | https://drive.google.com/file/d/1J9fNPk-gZSE9TpIvdy_ih6yPwbgkFOFM/view?usp=sharing |
| `test_rows_maithili_1.csv` | test data maithili | 2,022 | https://drive.google.com/file/d/1gsHJgHL2HbHNnEk6lWNBz1_kVtFW0zLg/view?usp=drive_link |

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
