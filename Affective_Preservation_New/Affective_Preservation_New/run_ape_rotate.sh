#!/bin/bash
# APE preservation rotation runner for ONE model + ONE task (headline|summary),
# both languages, 6-key Groq rotation, concurrency 1. Checkpointed + resumable.
# Usage: ./run_ape_rotate.sh "openai/gpt-oss-120b" headline
#        ./run_ape_rotate.sh "qwen/qwen3.8-27b"    summary
set -u
cd "$(dirname "$0")"

MODEL_ID="${1:?usage: run_ape_rotate.sh <model-id> <headline|summary>}"
TASK="${2:?usage: run_ape_rotate.sh <model-id> <headline|summary>}"
case "$TASK" in headline) GENCOL="generated_headline";; summary) GENCOL="generated_summary";;
  *) echo "task must be headline or summary"; exit 1;; esac
MODEL="groq:${MODEL_ID}"

KEYS=(
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_1"
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_2"
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_3"
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_4"
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_5"
  "gsk_REPLACE_WITH_YOUR_GROQ_KEY_6"
)
NK=${#KEYS[@]}
PASS_TIMEOUT=150
BHO_TOTAL=218
MAI_TOTAL=201
CKPT="ape_${TASK}_checkpoints"
OUT="ape_${TASK}_results"
mkdir -p "$CKPT" "$OUT"

SAFE=$(python3 -c "import re,sys;print(re.sub(r'[^A-Za-z0-9._-]','_',sys.argv[1]))" "$MODEL_ID")
bho_ckpt="$CKPT/bhojpuri__groq__${SAFE}.json"
mai_ckpt="$CKPT/maithili__groq__${SAFE}.json"
LOG="rotate_ape_${TASK}_${SAFE}.log"

count() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

ki=0
for pass in $(seq 1 300); do
  bdone=$(count "$bho_ckpt"); mdone=$(count "$mai_ckpt")
  echo "[ape-$TASK $(date +%H:%M:%S)] $MODEL_ID pass $pass | bho $bdone/$BHO_TOTAL | mai $mdone/$MAI_TOTAL | key#$((ki+1))"
  if [ "$bdone" -ge "$BHO_TOTAL" ] && [ "$mdone" -ge "$MAI_TOTAL" ]; then
    echo "ALL COMPLETE for $MODEL_ID ($TASK)."; break
  fi
  key="${KEYS[$ki]}"
  if [ "$bdone" -lt "$BHO_TOTAL" ]; then
    lang="bhojpuri"; input="ape_${TASK}_bhojpuri.csv"
  else
    lang="maithili"; input="ape_${TASK}_maithili.csv"
  fi
  GROQ_API_KEY="$key" python3 source_affect_classification.py \
      --name "$lang" --input "$input" --models "$MODEL" \
      --task "$TASK" --text-col "$GENCOL" \
      --ckpt-dir "$CKPT" --out-dir "$OUT" \
      --concurrency 1 --checkpoint-every 5 --max-seconds "$PASS_TIMEOUT" \
      >> "$LOG" 2>&1
  ki=$(( (ki + 1) % NK ))
done
echo "[ape-$TASK] DONE $MODEL_ID. (per-slice rows/metrics in $OUT/)"
