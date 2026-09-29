#!/bin/bash
# Two more T34 v3 runs at once, each changing one thing against the grid cell v3-ternary-t34:
#   seed 8: both stages exactly as run-v3-from-scratch.sh, with --seed 8 (default 7): is the result reproducible?
#   long:   the grid cell's own stage 1, then stage 2 with 24 000 steps instead of 12 000: does longer training help?
# Then the full PROTOCOL-v2 evaluation, export and verification of each (scripts/eval-cell.sh).
source "$(dirname "$0")/env.sh"
cd $TR
STAGE2="--train $W/corpus/train-v7a $W/corpus/gen-train-s1235 --weights 0.75 0.25 --val $W/corpus/val-v2b --val-limit 40000 \
        --lr 3.5e-4 --warmup 300 --eval-every 2000"
( $PY train/train_ternary.py --format t34 --size L --seed 8 --train $W/corpus/train-v7a $W/corpus/gen-train-s123 --weights 0.8 0.2 \
      --val $W/corpus/val-v2b --val-limit 40000 --lr 7e-4 --steps 6000 --warmup 500 --eval-every 1000 \
      --out $RUNS/v3-t34-seed8-stage1 > $LOGS/v3-t34-seed8-stage1.log 2>&1 && \
  $PY train/train_ternary.py --format t34 --seed 8 --init $RUNS/v3-t34-seed8-stage1/model $STAGE2 --steps 12000 \
      --out $RUNS/v3-t34-seed8-stage2 > $LOGS/v3-t34-seed8-stage2.log 2>&1 ) &
$PY train/train_ternary.py --format t34 --init $RUNS/v3-t34-stage1/model $STAGE2 --steps 24000 \
    --out $RUNS/v3-t34-long-stage2 > $LOGS/v3-t34-long-stage2.log 2>&1 &
wait
scripts/eval-cell.sh v3-ternary-t34-seed8 $RUNS/v3-t34-seed8-stage2/model t34
scripts/eval-cell.sh v3-ternary-t34-long $RUNS/v3-t34-long-stage2/model t34
echo "RUN5 DONE" >> $LOGS/status.log
