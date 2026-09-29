#!/bin/bash
# SWEEP-PLAN.md: the ternary speed rows first, while the machine is idle, then the post-training sweeps.
source "$(dirname "$0")/env.sh"
B=${ONEPASS_WEBGPU:?set ONEPASS_WEBGPU to a checkout of precisit/onepass-webgpu with work/c4-v2 prepared}
T=../work/ternary
mkdir -p $LOGS/speed
ln -sfn $TR $B/work/ternary
row() {  # the SPEED-PROTOCOL row of one v3 file, checked against ONNX Runtime on the same file
  echo "onepass-$1-v3=backend=onepass&precision=f32&file=$T/results/models/v3-ternary-$1/model.onnx&plan=$T/results/models/v3-ternary-$1/plan.json&plugins=$T/kernels/ternary-formats.js&ref=$T/work/ref-v3-ternary-$1/"
}
{ date -u +%FT%TZ; uptime; } >> $LOGS/speed.log
cd $B && ${BENCH_PY:-python} bench/run_bench.py --data work/c4-v2 --out $LOGS/speed --tag ternary \
  --backend "$(row t34)" --backend "$(row b243)" --only onepass-f32 onepass-int8 onepass-t34-v3 onepass-b243-v3 >> $LOGS/speed.log 2>&1
echo "SPEED DONE exit $?" >> $LOGS/status.log
cd $TR && $PY scripts/sweep.py --jobs 4 >> $LOGS/sweep.log 2>&1
echo "SWEEP DONE exit $?" >> $LOGS/status.log
