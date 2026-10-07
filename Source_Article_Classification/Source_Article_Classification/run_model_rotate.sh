#!/bin/bash
# Generic Groq key-rotation runner for ONE model, both languages, concurrency 1.
# Usage: ./run_model_rotate.sh "openai/gpt-oss-120b"
# Checkpointed + resumable. Rotates 6 keys, short timed passes, loops to full coverage.
set -u
cd "$(dirname "$0")"

MODEL_ID="${1:?usage: run_model_rotate.sh <model-id>}"
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
CKPT="source_classification_checkpoints"

# safe_name(model) as the python script computes it (non [A-Za-z0-9._-] -> _)
SAFE=$(python3 -c "import re,sys;print(re.sub(r'[^A-Za-z0-9._-]','_',sys.argv[1]))" "$MODEL_ID")
bho_ckpt="$CKPT/bhojpuri__groq__${SAFE}.json"
mai_ckpt="$CKPT/maithili__groq__${SAFE}.json"
LOG="rotate_${SAFE}.log"

count() { python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

ki=0
for pass in $(seq 1 300); do
  bdone=$(count "$bho_ckpt"); mdone=$(count "$mai_ckpt")
  echo "[rotate $(date +%H:%M:%S)] $MODEL_ID pass $pass | bho $bdone/$BHO_TOTAL | mai $mdone/$MAI_TOTAL | key#$((ki+1))"
  if [ "$bdone" -ge "$BHO_TOTAL" ] && [ "$mdone" -ge "$MAI_TOTAL" ]; then
    echo "ALL COMPLETE for $MODEL_ID."
    break
  fi
  key="${KEYS[$ki]}"
  if [ "$bdone" -lt "$BHO_TOTAL" ]; then
    lang="bhojpuri"; input="gold_bhojpuri.csv"
  else
    lang="maithili"; input="gold_maithili.csv"
  fi
  GROQ_API_KEY="$key" python3 source_affect_classification.py \
      --name "$lang" --input "$input" --models "$MODEL" \
      --concurrency 1 --checkpoint-every 5 --max-seconds "$PASS_TIMEOUT" \
      >> "$LOG" 2>&1
  ki=$(( (ki + 1) % NK ))
done

echo "[rotate] DONE $MODEL_ID. (per-slice rows/metrics written; rebuild combined table with aggregate_all.py)"
