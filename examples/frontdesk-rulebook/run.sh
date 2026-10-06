#!/bin/sh
# One rulebook → the llm rung's guide → two escalators that differ ONLY in that guide
# (short label definitions vs the whole rulebook) → calibrate → eval on the test rows and on the hard rows.
#
# Offline by default: cache/chat-cache.json holds the llm rung's answers (first-token log-probs)
# for every question this script asks, so no model and no key are needed. Change a row or the
# rulebook and the ladder calls a local OpenAI-compatible gateway for the answers it lacks
# (HEIMDALL_API_URL, HEIMDALL_API_KEY; the model name must be gemma-4-26b) — see README.md.
set -eu
cd "$(dirname "$0")"
LADDER=${LADDER:-"cargo run --quiet --release --bin ladder --"}
OUT=${OUT:-out}
mkdir -p "$OUT"
$LADDER rulebook render --rulebook rulebook.json --guide-out "$OUT/guide.txt" --descriptions-out "$OUT/labels.json"
for V in short rulebook; do
  if [ "$V" = rulebook ]; then GUIDE="--guide $OUT/guide.txt"; else GUIDE=""; fi
  $LADDER train --task frontdesk --train data/train.jsonl --out "$OUT/$V.json" \
    --descriptions "$OUT/labels.json" --no-encoder --llm gemma-4-26b $GUIDE > "$OUT/$V.train.txt"
  $LADDER calibrate --ladder "$OUT/$V.json" --verified data/calibrate.jsonl --out "$OUT/$V.cal.json" \
    --alpha 0.05 --when-unsure answer --chat-cache cache/chat-cache.json > "$OUT/$V.cal.txt"
  for SET in test hard; do
    $LADDER eval --ladder "$OUT/$V.cal.json" --cases "data/$SET.jsonl" --chat-cache cache/chat-cache.json > "$OUT/$V.$SET.txt"
    printf '%-9s %-5s %s\n' "$V" "$SET" "$(grep -m1 -E 'labeled n=' "$OUT/$V.$SET.txt")"
  done
done
