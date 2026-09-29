"""onepass-webgpu plan for a compact ternary file (export/export_onnx.py output).

    python export/make_plan.py <variant dir>        -> <variant dir>/plan.json

Tensor roles are those of the onepass-webgpu scorer plan (compiler/compile_onepass.py); ternary matrices use
the plugin formats in kernels/ternary-formats.js ("base243" / "t34").
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    v = Path(sys.argv[1])
    report = json.loads((v / "variant.json").read_text())
    config = json.loads((v / "model" / "config.json").read_text())["config"]
    model = onnx.load(str(v / "model.onnx"))
    inits = {t.name: t for t in model.graph.initializer}
    kind = {"b243": "base243", "t34": "t34"}[report["format"]]
    group = report["group"]
    width, ff, rank = config["width"], None, config["rank"]
    tensors = {}

    def dims(name):
        return list(inits[name].dims)

    def plain(key, name, rows=None):
        t = {"initializer": name, "transpose": False, "shape": dims(name)}
        if rows:
            t["rows"] = rows
            t["shape"] = [rows[1] - rows[0]] + dims(name)[1:]
        tensors[key] = t

    def int8(key, name, transpose):
        scale = float(numpy_helper.to_array(inits[name + ".s"]).reshape(-1)[0])
        shape = dims(name)[::-1] if transpose else dims(name)
        tensors[key] = {"initializer": name, "transpose": transpose, "shape": shape,
                        "quant": {"dtype": "int8", "scale": scale, "zero_point": 0}}

    def ternary(key, name):
        n, groups, _ = dims(name + ".c")
        tensors[key] = {"initializer": name + ".c", "transpose": False, "shape": [groups * group, n],
                        "format": {"kind": kind, "group": group, "scales": name + ".s"}}

    int8("embedding", "embedding.weight", False)
    plain("pos_context", "position.weight", [0, config["context_tokens"]])
    plain("pos_option", "position.weight", [0, config["option_tokens"]])
    layers = [(f"layer{i}", f"encoder.layers.{i}") for i in range(config["layers"])] + [("option_layer", "option_encoder.layers.0")]
    for key, pre in layers:
        ternary(f"{key}.qkv.w", f"{pre}.self_attn.in_proj_weight")
        plain(f"{key}.qkv.b", f"{pre}.self_attn.in_proj_bias")
        ternary(f"{key}.out.w", f"{pre}.self_attn.out_proj.weight")
        plain(f"{key}.out.b", f"{pre}.self_attn.out_proj.bias")
        ternary(f"{key}.ff1.w", f"{pre}.linear1.weight")
        plain(f"{key}.ff1.b", f"{pre}.linear1.bias")
        ternary(f"{key}.ff2.w", f"{pre}.linear2.weight")
        plain(f"{key}.ff2.b", f"{pre}.linear2.bias")
        for norm in ("norm1", "norm2"):
            plain(f"{key}.{norm}.w", f"{pre}.{norm}.weight")
            plain(f"{key}.{norm}.b", f"{pre}.{norm}.bias")
        ff = dims(f"{pre}.linear1.bias")[0]
    for role, name in (("q", "query"), ("k", "key"), ("v", "value")):
        int8(f"head.{role}.w", f"head.{name}.weight", True)       # torch [out, in] -> [in, out]
    for norm in ("context_norm", "option_norm"):
        plain(f"head.{norm}.w", f"head.{norm}.weight")
        plain(f"head.{norm}.b", f"head.{norm}.bias")
    data = (v / "model.onnx").read_bytes()
    plan = {"format": "onepass-plan/1", "architecture": "onepass-scorer",
            "config": {"vocab": dims("embedding.weight")[0], "width": width, "heads": config["heads"], "layers": config["layers"],
                       "ff": ff, "rank": rank, "context_len": config["context_tokens"], "option_slots": 7,
                       "option_len": config["option_tokens"], "eps": 1e-5, "activation": "relu", "norm_first": True},
            "model": {"file": "model.onnx", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "inputs": {"context_ids": [config["context_tokens"]], "option_ids": [7, config["option_tokens"]], "option_mask": [7]},
            "tensors": tensors, "plugins": ["ternary-formats.js"]}
    (v / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"wrote {v / 'plan.json'} ({len(tensors)} tensors, {sum('format' in t for t in tensors.values())} ternary)")


if __name__ == "__main__":
    main()
