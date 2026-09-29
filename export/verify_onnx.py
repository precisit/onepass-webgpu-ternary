"""Gate condition for "the file": ONNX Runtime (CPU) runs the exported file through its reference decomposition,
and its choices equal the dense-weight evaluation of the same derivative on the whole eval set.

    python export/verify_onnx.py <variant dir> --evalset <evalset dir> [--out <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", type=Path)
    p.add_argument("--evalset", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recipe = Path(os.environ.get("C4_RECIPE", Path.home() / "dev/one-pass-specialists/examples/c4"))
    sys.path.insert(0, str(recipe))
    import torch

    import encode_c4 as E
    from model_c4 import Batcher, load_checkpoint

    boards = np.load(a.evalset / "board.npy")
    model, _, _ = load_checkpoint(a.variant / "model", "cpu")
    batcher = Batcher(torch.device("cpu"))
    with torch.no_grad():
        dense = np.concatenate([model.eval()(batcher(torch.from_numpy(boards[i:i + 2048]))).float().numpy()
                                for i in range(0, len(boards), 2048)])          # [N, 7] by column
    legal = E.legal_mask(boards)

    # the browser's packing: legal columns in the first slots
    n = len(boards)
    ctx = E.contexts(boards).astype(np.int32) + 1
    table = E.option_ids_table().astype(np.int32)
    opt = np.zeros((n, 7, 8), np.int32)
    mask = np.zeros((n, 7), np.int32)
    cols = [np.flatnonzero(l) for l in legal]
    for i, c in enumerate(cols):
        opt[i, : len(c)] = table[c]
        mask[i, : len(c)] = 1
    sess = ort.InferenceSession(str(a.variant / "model.onnx"), providers=["CPUExecutionProvider"])
    out = np.concatenate([sess.run(None, {"context_ids": ctx[i:i + 1024], "option_ids": opt[i:i + 1024], "option_mask": mask[i:i + 1024]})[0]
                          for i in range(0, n, 1024)])
    same, worst = 0, 0.0
    for i, c in enumerate(cols):
        f = out[i, : len(c)]
        dcol = dense[i, c]
        worst = max(worst, float(np.abs(f - dcol).max()))
        same += int(c[f.argmax()] == c[dcol.argmax()])
    data = (a.variant / "model.onnx").read_bytes()
    res = {"variant": a.variant.name, "positions": n, "same_choice": same, "all_same": same == n,
           "max_abs_logit_diff": worst, "file_bytes": len(data), "file_sha256": hashlib.sha256(data).hexdigest(),
           "onnxruntime": ort.__version__}
    print(json.dumps(res))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
