"""Export a ternary derivative (ptq/variant.py output) as the compact ONNX file the concept gate measures.

    python export/export_onnx.py <variant dir> [--out <dir>/model.onnx]

The graph is written directly from the scorer's math (not the 969-node PyTorch export):

  - ternary matrices: custom ops `com.precisit.Base243MatMul` / `com.precisit.T34MatMul` (inputs: X, codes
    uint8 [N, K/G, B], scales fp16 [N, K/G]; attributes K, N, group). Each op has a model-local FunctionProto
    that decodes with standard ONNX ops (a lookup table, integer arithmetic), so ONNX Runtime and the onnx
    reference implementation run the file unchanged; a runtime with its own kernel uses the packed bytes.
  - int8 tensors (head q/k/v, embedding): DequantizeLinear; fp16 tensors: Cast.
  - Only the legal-option mask is applied. The v2 inputs never contain padding (44 context bytes, 8-byte
    options), so the key-padding and pooling masks of the toolkit model are all-true for every legal option;
    masked option slots get the lowest float, as in the toolkit export.

Inputs and output match the published v2 files: context_ids int32 [B, 44], option_ids int32 [B, 7, 8],
option_mask int32 [B, 7] -> logits float32 [B, 7].
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "formats"))
import ternary as T  # noqa: E402

DOMAIN = "com.precisit"
OPSET = 18


def base243_function(group: int) -> onnx.FunctionProto:
    lut = T.b243_decode5(np.arange(256, dtype=np.uint8)).astype(np.int8)  # [256, 5]
    nodes = [
        helper.make_node("Constant", [], ["lut"], value=numpy_helper.from_array(lut, "lut")),
        helper.make_node("Cast", ["lut"], ["lutf"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["codes"], ["ci"], to=TensorProto.INT32),
        helper.make_node("Gather", ["lutf", "ci"], ["t5"], axis=0),                       # [N, Gs, B, 5]
        helper.make_node("Constant", [], ["s3"], value=numpy_helper.from_array(np.array([0, 0, -1], np.int64), "s3")),
        helper.make_node("Reshape", ["t5", "s3"], ["tg"]),                                # [N, Gs, 5B]
        helper.make_node("Constant", [], ["z"], value=numpy_helper.from_array(np.array([0], np.int64), "z")),
        helper.make_node("Constant", [], ["g"], value=numpy_helper.from_array(np.array([group], np.int64), "g")),
        helper.make_node("Constant", [], ["ax"], value=numpy_helper.from_array(np.array([2], np.int64), "ax")),
        helper.make_node("Slice", ["tg", "z", "g", "ax"], ["t"]),                         # [N, Gs, G]
    ]
    return _finish_function("Base243MatMul", nodes)


def t34_function(group: int) -> onnx.FunctionProto:
    q = group // 4
    bits = 5 * np.arange(q)
    lo, shift = bits >> 3, bits & 7
    hi = np.minimum(lo + 1, 5 * group // 32 - 1)
    lut = T.T34_TABLE.astype(np.int8)                                                  # [32, 4]
    c = lambda name, a: helper.make_node("Constant", [], [name], value=numpy_helper.from_array(a, name))
    nodes = [
        c("lut", lut),
        helper.make_node("Cast", ["lut"], ["lutf"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["codes"], ["ci"], to=TensorProto.INT32),
        c("ilo", lo.astype(np.int64)), c("ihi", hi.astype(np.int64)),
        c("div", (2 ** shift).astype(np.int32)), c("k256", np.array(256, np.int32)), c("k32", np.array(32, np.int32)),
        helper.make_node("Gather", ["ci", "ilo"], ["blo"], axis=2),
        helper.make_node("Gather", ["ci", "ihi"], ["bhi"], axis=2),
        helper.make_node("Mul", ["bhi", "k256"], ["bhi2"]),
        helper.make_node("Add", ["blo", "bhi2"], ["w16"]),                                # 16-bit window
        helper.make_node("Div", ["w16", "div"], ["sh"]),
        helper.make_node("Mod", ["sh", "k32"], ["state"]),                                # [N, Gs, Q]
        helper.make_node("Gather", ["lutf", "state"], ["t4"], axis=0),                    # [N, Gs, Q, 4]
        c("s3", np.array([0, 0, -1], np.int64)),
        helper.make_node("Reshape", ["t4", "s3"], ["t"]),                                 # [N, Gs, G]
    ]
    return _finish_function("T34MatMul", nodes)


def _finish_function(name: str, decode: list) -> onnx.FunctionProto:
    nodes = decode + [
        helper.make_node("Cast", ["scales"], ["sf"], to=TensorProto.FLOAT),
        helper.make_node("Constant", [], ["ax2"], value=numpy_helper.from_array(np.array([2], np.int64), "ax2")),
        helper.make_node("Unsqueeze", ["sf", "ax2"], ["su"]),
        helper.make_node("Mul", ["t", "su"], ["wg"]),
        helper.make_node("Constant", [], ["s2"], value=numpy_helper.from_array(np.array([0, -1], np.int64), "s2")),
        helper.make_node("Reshape", ["wg", "s2"], ["wnk"]),                               # [N, K]
        helper.make_node("Transpose", ["wnk"], ["wkn"], perm=[1, 0]),                     # [K, N]
        helper.make_node("MatMul", ["X", "wkn"], ["Y"]),
    ]
    return helper.make_function(DOMAIN, name, ["X", "codes", "scales"], ["Y"], nodes,
                                [helper.make_opsetid("", OPSET)], ["K", "N", "group"])


class Graph:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, hint="t"):
        self.n += 1
        return f"{hint}{self.n}"

    def init(self, name, array):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def const(self, array, hint="c"):
        return self.init(self.name(hint), np.asarray(array))

    def op(self, op_type, inputs, domain="", hint="t", **attrs):
        out = self.name(hint)
        self.nodes.append(helper.make_node(op_type, inputs, [out], domain=domain or None, **attrs))
        return out


def export(variant: Path, out: Path) -> dict:
    report = json.loads((variant / "variant.json").read_text())
    config = json.loads((variant / "model" / "config.json").read_text())["config"]
    fmt, group = report["format"], report["group"]
    P = dict(np.load(variant / "packed.npz"))
    width, heads, layers, rank = config["width"], config["heads"], config["layers"], config["rank"]
    d = width // heads
    g = Graph()

    def fp16(name):
        return g.op("Cast", [g.init(name, P[f"{name}::fp16"])], to=TensorProto.FLOAT)

    def int8(name):
        return g.op("DequantizeLinear", [g.init(name, P[f"{name}::int8"]), g.init(name + ".s", P[f"{name}::scale"])])

    def matrix(x, name, k, n):
        """x @ W^T for an encoder matrix in whatever role the allocation gave it (ternary, int8 or fp16)."""
        if f"{name}::codes" in P:
            return ternary(x, name, k, n)
        w = int8(name) if f"{name}::int8" in P else fp16(name)                # torch layout [n, k]
        return g.op("MatMul", [x, g.op("Transpose", [w], perm=[1, 0])])

    def ternary(x, name, k, n):
        codes = g.init(name + ".c", P[f"{name}::codes"])
        scales = g.init(name + ".s", P[f"{name}::scales"])
        op = "Base243MatMul" if fmt == "b243" else "T34MatMul"
        return g.op(op, [x, codes, scales], domain=DOMAIN, K=k, N=n, group=group)

    def layer_norm(x, prefix):
        return g.op("LayerNormalization", [x, fp16(f"{prefix}.weight"), fp16(f"{prefix}.bias")], axis=-1, epsilon=1e-5)

    def encoder_layer(x, prefix, length):
        # x [S, L, W]; norm-first; torch MultiheadAttention packed in-projection (q, k, v)
        h = layer_norm(x, f"{prefix}.norm1")
        qkv = g.op("Add", [matrix(h, f"{prefix}.self_attn.in_proj_weight", width, 3 * width), fp16(f"{prefix}.self_attn.in_proj_bias")])
        split = g.const(np.array([width, width, width], np.int64))
        q, k, v = (g.name("q"), g.name("k"), g.name("v"))
        g.nodes.append(helper.make_node("Split", [qkv, split], [q, k, v], axis=-1))
        shape = g.const(np.array([0, length, heads, d], np.int64))
        heads_of = lambda t: g.op("Transpose", [g.op("Reshape", [t, shape])], perm=[0, 2, 1, 3])  # [S, H, L, D]
        qh, kh, vh = heads_of(q), heads_of(k), heads_of(v)
        scores = g.op("Mul", [g.op("MatMul", [qh, g.op("Transpose", [kh], perm=[0, 1, 3, 2])]), g.const(np.array(1 / math.sqrt(d), np.float32))])
        att = g.op("MatMul", [g.op("Softmax", [scores], axis=-1), vh])
        merged = g.op("Reshape", [g.op("Transpose", [att], perm=[0, 2, 1, 3]), g.const(np.array([0, length, width], np.int64))])
        x = g.op("Add", [x, g.op("Add", [matrix(merged, f"{prefix}.self_attn.out_proj.weight", width, width), fp16(f"{prefix}.self_attn.out_proj.bias")])])
        h = layer_norm(x, f"{prefix}.norm2")
        h = g.op("Relu", [g.op("Add", [matrix(h, f"{prefix}.linear1.weight", width, P[f"{prefix}.linear1.bias::fp16"].shape[0]), fp16(f"{prefix}.linear1.bias")])])
        ff = P[f"{prefix}.linear1.bias::fp16"].shape[0]
        return g.op("Add", [x, g.op("Add", [matrix(h, f"{prefix}.linear2.weight", ff, width), fp16(f"{prefix}.linear2.bias")])])

    ctx_len, opt_slots, opt_len = config["context_tokens"], 7, config["option_tokens"]
    emb = int8("embedding.weight")
    pos = fp16("position.weight")
    pos_ctx = g.op("Slice", [pos, g.const(np.array([0], np.int64)), g.const(np.array([ctx_len], np.int64)), g.const(np.array([0], np.int64))])
    pos_opt = g.op("Slice", [pos, g.const(np.array([0], np.int64)), g.const(np.array([opt_len], np.int64)), g.const(np.array([0], np.int64))])

    x = g.op("Add", [g.op("Gather", [emb, "context_ids"], axis=0), pos_ctx])
    for i in range(layers):
        x = encoder_layer(x, f"encoder.layers.{i}", ctx_len)
    oid = g.op("Reshape", ["option_ids", g.const(np.array([-1, opt_len], np.int64))])      # [B*7, 8]
    o = g.op("Add", [g.op("Gather", [emb, oid], axis=0), pos_opt])
    o = encoder_layer(o, "option_encoder.layers.0", opt_len)
    o = g.op("ReduceMean", [o, g.const(np.array([1], np.int64))], keepdims=0)              # [B*7, W]
    o = g.op("Reshape", [o, g.const(np.array([-1, opt_slots, width], np.int64))])        # [B, 7, W]
    c = layer_norm(x, "head.context_norm")
    o = layer_norm(o, "head.option_norm")
    wq, wk, wv = (g.op("Transpose", [int8(f"head.{n}.weight")], perm=[1, 0]) for n in ("query", "key", "value"))
    q = g.op("MatMul", [o, wq])                                                            # [B, 7, R]
    k = g.op("MatMul", [c, wk])                                                            # [B, L, R]
    v = g.op("MatMul", [c, wv])
    inv = g.const(np.array(1 / math.sqrt(rank), np.float32))
    s = g.op("Mul", [g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 2, 1])]), inv])
    attended = g.op("MatMul", [g.op("Softmax", [s], axis=-1), v])
    logit = g.op("Mul", [g.op("ReduceSum", [g.op("Mul", [q, attended]), g.const(np.array([-1], np.int64))], keepdims=0), inv])
    legal = g.op("Cast", ["option_mask"], to=TensorProto.BOOL)
    g.nodes.append(helper.make_node("Where", [legal, logit, g.const(np.array(np.finfo(np.float32).min, np.float32))], ["logits"]))

    graph = helper.make_graph(
        g.nodes, "onepass-scorer-ternary",
        [helper.make_tensor_value_info("context_ids", TensorProto.INT32, ["B", ctx_len]),
         helper.make_tensor_value_info("option_ids", TensorProto.INT32, ["B", opt_slots, opt_len]),
         helper.make_tensor_value_info("option_mask", TensorProto.INT32, ["B", opt_slots])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", opt_slots])],
        g.inits)
    func = base243_function(group) if fmt == "b243" else t34_function(group)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid(DOMAIN, 1)],
                              functions=[func], producer_name="onepass-webgpu-ternary")
    model.ir_version = 9
    alloc = "" if report.get("select", "all") == "all" else f", ternary {report['select']}, others {report['protect']}"
    model.doc_string = f"{report['name']}: v2 architecture, {fmt} G{group} ternary matrices{alloc} (derivative, research)"
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    data = out.read_bytes()
    tensor_bytes = sum(t.ByteSize() for t in model.graph.initializer)
    import gzip
    return {"file": str(out.name), "bytes": len(data), "gzip_bytes": len(gzip.compress(data, 9)),
            "initializer_bytes": tensor_bytes, "graph_and_function_bytes": len(data) - tensor_bytes,
            "nodes": len(model.graph.node), "under_2MB": len(data) < 2_000_000}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variant", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or a.variant / "model.onnx"
    print(json.dumps(export(a.variant, out)))


if __name__ == "__main__":
    main()
