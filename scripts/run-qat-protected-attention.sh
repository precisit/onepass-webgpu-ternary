#!/bin/bash
# SWEEP-PLAN.md: QAT from v2 with attention protected at int8 (MLP ternary), both formats at once, with
# the grid's QAT recipe unchanged; then the full PROTOCOL-v2 evaluation, export and verification of each.
source "$(dirname "$0")/env.sh"
cd $TR
ALLOC="--select except:qkv,out --protect int8"
QAT="--init $V2 --teacher $V2 --kl 1.0 --train $W/corpus/train-v7a $W/corpus/gen-train-s1235 --weights 0.75 0.25 \
     --val $W/corpus/val-v2b --val-limit 40000 --lr 3.5e-4 --steps 12000 --warmup 300 --eval-every 2000"
for F in b243 t34; do
  $PY train/train_ternary.py --format $F $ALLOC $QAT --out $RUNS/qat-attn-$F-train > $LOGS/qat-attn-$F-train.log 2>&1 &
done
wait
for F in b243 t34; do scripts/eval-cell.sh v2-ternary-$F-qat-attn-int8 $RUNS/qat-attn-$F-train/model $F $ALLOC; done
echo "RUN4 DONE" >> $LOGS/status.log
