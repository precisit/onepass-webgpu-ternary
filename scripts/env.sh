# Environment for the scripts in this folder (sourced by each of them). Set these before running:
#   C4_ROOT    the Connect Four work area: work/corpus (the training corpora, val-v2b, evalset-v2), work/runs
#              (the dense v2 checkpoint), work/models (v1's ONNX file), a Python venv, and bin/c4label
#   C4_RECIPE  a checkout of precisit/one-pass-specialists at examples/c4 (the recipe: encoder, model, trainer,
#              evaluation protocol); defaults to $C4_ROOT/one-pass-specialists/examples/c4
#   TR         this repository; defaults to the folder above this one
export C4_ROOT=${C4_ROOT:?set C4_ROOT to the Connect Four work area}
export W=$C4_ROOT/work
export C4_RECIPE=${C4_RECIPE:-$C4_ROOT/one-pass-specialists/examples/c4}
export PY=${PY:-$C4_ROOT/.venv/bin/python}
export C4LABEL=${C4LABEL:-$C4_ROOT/bin/c4label}
export TR=${TR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export RUNS=$TR/work/runs
export LOGS=$TR/work/logs
export V2=${V2:-$W/runs/main-r0-L/model}
export V1=${V1:-$W/models/onepass-c4-8x24.onnx}
mkdir -p $RUNS $LOGS
