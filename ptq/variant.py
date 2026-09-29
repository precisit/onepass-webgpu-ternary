"""Turn a float Connect Four scorer state into a ternary derivative under the CONCEPT-GATE default allocation.

    python ptq/variant.py --checkpoint <toolkit checkpoint dir> --format b243|t34 [--group 128] --out <dir>

Writes to <dir>:
  model/            a toolkit checkpoint whose float weights are exactly the values the packed file holds
                    (ternary * fp16 scale, int8 * scale, fp16), so the PROTOCOL-v2 harness evaluates the
                    shippable model unchanged (`run-full.sh <dir>/model ...`)
  packed.npz        the packed tensors (codes, scales, int8 and fp16 tensors), input to the ONNX exporter
  variant.json      allocation, bits per weight and predicted byte counts

The same `quantize_state` is used for PTQ (on v2's weights), and to export QAT / from-scratch runs from their
latent float weights, so an exported model is exactly what its training forward pass computed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "formats"))
import ternary as T  # noqa: E402

TERNARY_SUFFIXES = ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight")
INT8 = ("head.query.weight", "head.key.weight", "head.value.weight", "embedding.weight")


# Selectors over the 36 eligible matrices (SWEEP-PLAN.md): "all" (the default allocation), "none",
# "only:<tokens>" or "except:<tokens>". Tokens: qkv, out, up, down (a matrix type in all 9 layers),
# L0..L7 (a context layer, all four matrices), opt (the option-encoder layer).
TYPES = {"qkv": "self_attn.in_proj_weight", "out": "self_attn.out_proj.weight", "up": "linear1.weight",
         "down": "linear2.weight"}


def eligible(name: str) -> bool:
    return name.startswith(("encoder.layers.", "option_encoder.layers.")) and name.endswith(TERNARY_SUFFIXES)


def _matches(name: str, token: str) -> bool:
    if token in TYPES:
        return name.endswith(TYPES[token])
    if token == "opt":
        return name.startswith("option_encoder.layers.")
    if token[:1] == "L" and token[1:].isdigit():
        return name.startswith(f"encoder.layers.{int(token[1:])}.")
    raise ValueError(f"unknown selector token {token!r}")


def selected(name: str, select: str = "all") -> bool:
    if select == "all":
        return True
    if select == "none":
        return False
    mode, _, tokens = select.partition(":")
    if mode not in ("only", "except") or not tokens:
        raise ValueError(f"bad selector {select!r}")
    hit = any(_matches(name, t) for t in tokens.split(","))
    return hit if mode == "only" else not hit


def role(name: str, select: str = "all", protect: str = "fp16") -> str:
    """ternary / int8 / fp16. Eligible matrices outside the selection get `protect` (fp16 or int8)."""
    if eligible(name):
        return "ternary" if selected(name, select) else protect
    if name in INT8:
        return "int8"
    return "fp16"


def int8_symmetric(w: np.ndarray) -> tuple[np.ndarray, np.float32]:
    scale = np.float32(np.abs(w).max() / 127.0)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def method_for(name: str, fit: str) -> str:
    """fit: "<method>[,<token>=<method>...]", e.g. "ls,qkv=lloydmax" (tokens as in the selectors)."""
    default, *overrides = fit.split(",")
    for o in overrides:
        token, _, method = o.partition("=")
        if _matches(name, token):
            return method
    return default


def fit_matrix(fit_fn, w: np.ndarray, group: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    """method: ls | absmean | lloydmax | rms (least-squares zeros, energy-preserving scale: ||s t|| = ||w|| per
    group), optionally "*<factor>" to multiply the scale (e.g. ls*1.2)."""
    base, _, factor = method.partition("*")
    if base == "rms":
        trits, _ = fit_fn(w, group, "ls")
        n, k = w.shape
        g = np.asarray(w, np.float64).reshape(n, k // group, group)
        nnz = (trits.reshape(n, k // group, group) != 0).sum(-1)
        scale = np.sqrt((g ** 2).sum(-1) / np.maximum(nnz, 1))
    else:
        trits, scale = fit_fn(w, group, base)
    scale = np.asarray(scale, np.float64) * (float(factor) if factor else 1.0)
    return trits, scale.astype(np.float16)


def quantize_state(state: dict, fmt: str, group: int, method: str = "ls", select: str = "all",
                   protect: str = "fp16") -> tuple[dict, dict, dict]:
    """state: name -> float32 array (torch layout, matrices [out, in]).
    Returns (dense float32 state of the shippable values, packed arrays, report)."""
    fit, pack, unpack = {"b243": (T.fit_b243, T.pack_b243, T.unpack_b243),
                         "t34": (T.fit_t34, T.pack_t34, T.unpack_t34)}[fmt]
    dense, packed = {}, {}
    count = {"ternary": 0, "int8": 0, "fp16": 0}
    nbytes = {"ternary_codes": 0, "ternary_scales": 0, "int8": 0, "fp16": 0}
    for name, w in state.items():
        w = np.asarray(w, dtype=np.float32)
        r = role(name, select, protect)
        count[r] += w.size
        if r == "ternary":
            trits, scales = fit_matrix(fit, w, group, method_for(name, method))
            codes = pack(trits, group)
            if not np.array_equal(unpack(codes, group), trits):
                raise AssertionError(f"{name}: pack/unpack mismatch")
            dense[name] = T.dequant(trits, scales, group)
            packed[f"{name}::codes"] = codes
            packed[f"{name}::scales"] = scales
            nbytes["ternary_codes"] += codes.nbytes
            nbytes["ternary_scales"] += scales.nbytes
        elif r == "int8":
            q, scale = int8_symmetric(w)
            dense[name] = q.astype(np.float32) * scale
            packed[f"{name}::int8"] = q
            packed[f"{name}::scale"] = np.array(scale, dtype=np.float32)
            nbytes["int8"] += q.nbytes + 4
        else:
            h = w.astype(np.float16)
            dense[name] = h.astype(np.float32)
            packed[f"{name}::fp16"] = h
            nbytes["fp16"] += h.nbytes
    total_bytes = sum(nbytes.values())
    params = sum(count.values())
    bpw = T.bits_per_weight("base243" if fmt == "b243" else "t34", group)
    report = {"format": fmt, "group": group, "fit": method, "select": select, "protect": protect, "parameters": count, "bytes": nbytes, "tensor_bytes": total_bytes,
              "bits_per_weight": {"ternary_codes": bpw["codes"], "ternary_codes_and_scales": bpw["codes_and_scales"],
                                  "all_parameters": round(8 * total_bytes / params, 4)}}
    return dense, packed, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--format", choices=["b243", "t34"], required=True)
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fit", default="ls", help="ls | absmean | lloydmax | rms, optional *factor, optional per-token overrides "
                                          "(e.g. ls*1.2 or ls,qkv=lloydmax)")
    p.add_argument("--select", default="all", help="which eligible matrices are ternary (see TYPES above)")
    p.add_argument("--protect", choices=["fp16", "int8"], default="fp16", help="format of the unselected ones")
    p.add_argument("--name", help="derivative name recorded in the checkpoint metadata")
    a = p.parse_args()

    from safetensors.numpy import load_file

    state = load_file(str(a.checkpoint / "model.safetensors"))
    config = json.loads((a.checkpoint / "config.json").read_text())
    dense, packed, report = quantize_state(state, a.format, a.group, a.fit, a.select, a.protect)
    def rel_err(r):
        keys = [k for k in state if role(k, a.select, a.protect) == r]
        if not keys:
            return None
        return float(np.sqrt(sum(((dense[k] - state[k]) ** 2).sum() for k in keys) / sum((state[k] ** 2).sum() for k in keys)))
    err = {r: rel_err(r) for r in ("ternary", "int8", "fp16")}
    report["relative_rms_error"] = err
    report["source"] = {"checkpoint": str(a.checkpoint.name), "state_signature": config.get("state_signature")}
    report["name"] = a.name or f"ternary-{a.format}-g{a.group}"

    # save through the toolkit, so the harness loads it like any other checkpoint
    recipe = Path(os.environ.get("C4_RECIPE", Path.home() / "dev/one-pass-specialists/examples/c4"))
    sys.path.insert(0, str(recipe))
    import torch
    from model_c4 import make_system, save_checkpoint

    model, _ = make_system(config["config"], "cpu")
    model.load_state_dict({k: torch.from_numpy(v) for k, v in dense.items()}, strict=True)
    a.out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(a.out / "model", model, config["config"], {"derivative": report["name"], "from": report["source"],
                                                                 "format": a.format, "group": a.group, "fit": a.fit,
                                                                 "select": a.select, "protect": a.protect})
    np.savez(a.out / "packed.npz", **packed)
    (a.out / "variant.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("name", "tensor_bytes", "bits_per_weight", "relative_rms_error")}))


if __name__ == "__main__":
    main()
