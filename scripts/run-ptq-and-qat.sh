#!/bin/bash
# The grid's PTQ cells, the v1 reference cell, and both QAT runs (concurrently on the GPU).
source "$(dirname "$0")/env.sh"
cd $TR
QAT="--init $V2 --teacher $V2 --kl 1.0 --train $W/corpus/train-v7a $W/corpus/gen-train-s1235 --weights 0.75 0.25 \
     --val $W/corpus/val-v2b --val-limit 40000 --lr 3.5e-4 --steps 12000 --warmup 300 --eval-every 2000"
for F in b243 t34; do
  $PY train/train_ternary.py --format $F $QAT --out $RUNS/qat-$F-train > $LOGS/qat-$F-train.log 2>&1 &
done
# CPU work while the GPU trains
scripts/eval-cell.sh v2-ternary-b243-ptq $V2 b243
scripts/eval-cell.sh v2-ternary-t34-ptq $V2 t34
cd $C4_RECIPE && $PY protocol.py --a onnx-v1:$V1 --b bot:4 --seed 2026 --games 200 --workers 8 --out $LOGS/v1-reference.jsonl > /dev/null 2>&1
echo "v1-reference DONE" >> $LOGS/status.log
wait
cd $TR
for F in b243 t34; do scripts/eval-cell.sh v2-ternary-$F-qat $RUNS/qat-$F-train/model $F; done
echo "RUN1 DONE" >> $LOGS/status.log
