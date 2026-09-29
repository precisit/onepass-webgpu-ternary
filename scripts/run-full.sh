#!/bin/bash
source "${C4_ENV:?set C4_ENV to a file exporting W, PY, C4LABEL}"
C=$1; OUT=$2
$PY eval_c4.py score --checkpoint $C --evalset $W/corpus/evalset-v2 --controls --device cpu --out $OUT 2>/dev/null | python3 -c "
import json,sys; d=json.loads(sys.stdin.read()); b=d.pop('bands'); print('EVAL', d); print({k:(v['vp'],v['vp_nt']) for k,v in b.items()})"
for spec in "bot:4 2026" "bot:6 2026" "bot:2 2026" "random 2026" "onnx-v1:$W/models/onepass-c4-8x24.onnx 2026" "solver 2026" "bot:4 2027" "bot:4 2028" "bot:6 2027" "bot:6 2028"; do
  set -- $spec
  $PY protocol.py --a model:$C --b $1 --seed $2 --games 200 --workers 8 --out $OUT 2>/dev/null | python3 -c "
import json,sys
d=json.loads(sys.stdin.read()); op=d.get('a_on_policy',{})
print(d['b'][-24:], 'seed', d['seed'], d['a_score'], d['a_score_ci'], d['a_wins'], d['b_wins'], d['draws'], 'distinct', d['distinct_games'], 'onpol', op.get('value_preserving'), op.get('value_preserving_nontrivial'))"
done
