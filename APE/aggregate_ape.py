"""
aggregate_ape.py — build combined APE (preservation) tables from the per-slice
metrics.json in ape_headline_results/ and ape_summary_results/ (both the Groq-run
new-classifier files and the local-GPU bundle splits). No API calls.

Outputs:
  ape_comparison_full.csv  — one row per task x language x model
  ape_per_class_full.csv   — per-class precision/recall/f1/support
"""
import json, csv
from pathlib import Path

TASKS = {"headline": Path("ape_headline_results"), "summary": Path("ape_summary_results")}
COLS = ["task", "language", "provider", "model", "n_scored", "n_missing",
        "emo_acc", "emo_macro_f1", "emo_weighted_f1", "emo_kappa", "emo_mean_js",
        "sent_acc", "sent_macro_f1", "sent_weighted_f1", "sent_kappa"]

def main():
    summary, per_class = [], []
    for task, d in TASKS.items():
        if not d.exists():
            continue
        for f in sorted(d.glob("*__metrics.json")):
            m = json.load(open(f, encoding="utf-8"))
            row = {c: m.get(c) for c in COLS}
            row["task"] = task
            row["provider"] = m.get("provider", "groq")
            summary.append(row)
            for axis, key in (("emotion", "emotion_per_class"), ("sentiment", "sentiment_per_class")):
                rep = m.get(key)
                if not rep:
                    continue
                for label, mm in rep.items():
                    if not isinstance(mm, dict):
                        continue
                    per_class.append({"task": task, "language": m.get("language"),
                                      "provider": row["provider"], "model": m.get("model"),
                                      "axis": axis, "class": label,
                                      "precision": mm.get("precision"), "recall": mm.get("recall"),
                                      "f1": mm.get("f1-score"), "support": mm.get("support")})
    summary.sort(key=lambda r: (r["task"], str(r["model"]), r["language"]))
    with open("ape_comparison_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(summary)
    if per_class:
        pc_cols = ["task", "language", "provider", "model", "axis", "class", "precision", "recall", "f1", "support"]
        with open("ape_per_class_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=pc_cols); w.writeheader(); w.writerows(per_class)

    print(f"APE tables rebuilt from {len(summary)} slices:")
    for r in summary:
        ea = r.get("emo_acc"); sa = r.get("sent_acc")
        ea = f"{ea:.3f}" if isinstance(ea, (int, float)) else "  -  "
        sa = f"{sa:.3f}" if isinstance(sa, (int, float)) else "  -  "
        print(f"  {r['task']:8s} {r['language']:9s} {str(r['model']):32s} emo={ea} sent={sa}")
    print("\nSaved: ape_comparison_full.csv + ape_per_class_full.csv")

if __name__ == "__main__":
    main()
