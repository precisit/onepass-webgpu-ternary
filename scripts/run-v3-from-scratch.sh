#!/bin/bash
# The grid's v3 cells: trained ternary from scratch in both formats (concurrently), v2's two stages.
source "$(dirname "$0")/env.sh"
cd $TR
for F in b243 t34; do
  ( $PY train/train_ternary.py --format $F --size L --train $W/corpus/train-v7a $W/corpus/gen-train-s123 --weights 0.8 0.2 \
      --val $W/corpus/val-v2b --val-limit 40000 --lr 7e-4 --steps 6000 --warmup 500 --eval-every 1000 \
      --out $RUNS/v3-$F-stage1 > $LOGS/v3-$F-stage1.log 2>&1 && \
    $PY train/train_ternary.py --format $F --init $RUNS/v3-$F-stage1/model --train $W/corpus/train-v7a $W/corpus/gen-train-s1235 \
      --weights 0.75 0.25 --val $W/corpus/val-v2b --val-limit 40000 --lr 3.5e-4 --steps 12000 --warmup 300 --eval-every 2000 \
      --out $RUNS/v3-$F-stage2 > $LOGS/v3-$F-stage2.log 2>&1 ) &
done
wait
for F in b243 t34; do scripts/eval-cell.sh v3-ternary-$F $RUNS/v3-$F-stage2/model $F; done
echo "RUN2 DONE" >> $LOGS/status.log
