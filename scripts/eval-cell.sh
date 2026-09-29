#!/bin/bash
# Full PROTOCOL-v2 evaluation of one derivative (eval set + controls + the 10 game cells), then its size.
#   eval-cell.sh <name> <latent-or-float checkpoint> <b243|t34> [variant.py options, e.g. --select ... --protect int8]
source "$(dirname "$0")/env.sh"
NAME=$1; SRC=$2; FMT=$3; shift 3
cd $TR
$PY ptq/variant.py --checkpoint $SRC --format $FMT "$@" --out $RUNS/$NAME --name $NAME > $LOGS/$NAME-variant.json 2>/dev/null
$PY export/export_onnx.py $RUNS/$NAME > $LOGS/$NAME-onnx.json 2>/dev/null
$PY export/verify_onnx.py $RUNS/$NAME --evalset $W/corpus/evalset-v2 --out $LOGS/$NAME-verify.json > /dev/null 2>&1
cd $C4_RECIPE
C4_ENV=$TR/scripts/c4env.sh bash $TR/scripts/run-full.sh $RUNS/$NAME/model $LOGS/$NAME-eval.jsonl > $LOGS/$NAME-eval.txt 2>&1
echo "$NAME DONE" >> $LOGS/status.log
