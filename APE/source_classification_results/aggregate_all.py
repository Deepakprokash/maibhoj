"""
aggregate_all.py — rebuild the combined reviewer tables from every per-slice
metrics.json in source_classification_results/. Safe to run anytime; makes NO
API calls. Each model run overwrites comparison_full.csv with only its own rows,
so this reconstructs the full multi-model table from the durable metrics files.

Outputs:
  comparison_full.csv  — one row per language x model (headline metrics)
  per_class_full.csv   — per-class precision/recall/f1/support
"""
import json, glob, csv
from pathlib import Path

RES = Path("source_classification_results")

SUMMARY_COLS = ["language", "provider", "model", "n_scored", "n_missing",
                "emo_acc", "emo_macro_precision", "emo_macro_recall", "emo_macro_f1",
                "emo_weighted_f1", "emo_kappa", "emo_mean_js",
                "sent_acc", "sent_macro_precision", "sent_macro_recall", "sent_macro_f1",
                "sent_weighted_f1", "sent_kappa"]

def _ingest(d, summary, per_class):
    """Fold one metrics-detail dict (a slice) into the running tables."""
    summary.append({c: d.get(c) for c in SUMMARY_COLS})
    for axis, key in (("emotion", "emotion_per_class"), ("sentiment", "sentiment_per_class")):
        rep = d.get(key)
        if not rep:
            continue
        for label, m in rep.items():
            if not isinstance(m, dict):
                continue
            per_class.append({
                "language": d.get("language"), "provider": d.get("provider"),
                "model": d.get("model"), "axis": axis, "class": label,
                "precision": m.get("precision"), "recall": m.get("recall"),
                "f1": m.get("f1-score"), "support": m.get("support"),
            })


def _slices_from_bundle(obj):
    """A portable bundle is either a list of slice-dicts, or {'slices': [...]}."""
    if isinstance(obj, dict) and "slices" in obj:
        return obj["slices"]
    if isinstance(obj, list):
        return obj
    return [obj]   # a single slice dict


def main():
    summary, per_class = [], []
    seen = set()   # (language, model) de-dupe so a bundle + its metrics.json don't double-count
    # per-slice metrics.json (native pipeline output)
    for f in sorted(RES.glob("*__metrics.json")):
        d = json.load(open(f, encoding="utf-8"))
        key = (d.get("language"), d.get("model"))
        if key in seen:
            continue
        seen.add(key); _ingest(d, summary, per_class)
    # portable bundles handed back from the standalone notebook (e.g. local GPU run)
    for f in sorted(RES.glob("*_bundle.json")):
        for d in _slices_from_bundle(json.load(open(f, encoding="utf-8"))):
            key = (d.get("language"), d.get("model"))
            if key in seen:
                continue
            seen.add(key); _ingest(d, summary, per_class)

    # sort for stable, readable tables
    summary.sort(key=lambda r: (r["model"] or "", r["language"] or ""))
    with open(RES / "comparison_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS); w.writeheader(); w.writerows(summary)
    if per_class:
        pc_cols = ["language", "provider", "model", "axis", "class", "precision", "recall", "f1", "support"]
        with open(RES / "per_class_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=pc_cols); w.writeheader(); w.writerows(per_class)

    print(f"Rebuilt combined tables from {len(summary)} slices:")
    for r in summary:
        ea = r.get("emo_acc"); sa = r.get("sent_acc")
        ea = f"{ea:.3f}" if isinstance(ea, (int, float)) else "  -  "
        sa = f"{sa:.3f}" if isinstance(sa, (int, float)) else "  -  "
        print(f"  {r['language']:9s} {str(r['model']):26s} n={str(r['n_scored']):>3}  emo_acc={ea}  sent_acc={sa}")
    print(f"\nSaved: {RES/'comparison_full.csv'} + {RES/'per_class_full.csv'}")

if __name__ == "__main__":
    main()
