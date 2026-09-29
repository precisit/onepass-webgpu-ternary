"""Ternary-aware training for the Connect Four scorer (QAT from v2, or v3 from scratch), on MLX.

This is the v2 trainer (`train_mlx.py` in one-pass-specialists/examples/c4: same data, sampling, targets, schedule
and metrics) with one change to the forward pass: every weight the default allocation quantizes is replaced
by its quantized value (straight-through gradient), in the exact target format:

  - the 36 encoder matrices: Base243 (arbitrary ternary) or T34 (exact 3-of-4), per-group fp16 scales,
    fitted by least squares every step (as `ptq/variant.py` fits them once);
  - head q/k/v and the embedding: symmetric per-tensor int8;
  - everything else: rounded to fp16.

Optional distillation (`--teacher`): KL divergence from the teacher's move distribution (v2) to the student's,
added to the v2 loss with weight `--kl`. Checkpoints hold the latent float weights; the shippable model is
made from them with `ptq/variant.py`, which applies the same quantization once.

  python train/train_ternary.py --format t34 --init <v2 checkpoint> --teacher <v2 checkpoint> --kl 1.0 \
      --train <corpora> --weights 0.75 0.25 --val <val> --out <run> --lr 3.5e-4 --steps 12000 --warmup 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

RECIPE = Path(os.environ.get("C4_RECIPE", Path.home() / "dev/one-pass-specialists/examples/c4"))  # the Connect Four recipe
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ptq"))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import encode_c4 as E  # noqa: E402
import mlx_scorer as X  # noqa: E402
from model_c4 import config_for, load_checkpoint, make_system, parameter_count  # noqa: E402
from train_c4 import PlySampler, load_corpus, move_metrics  # noqa: E402
from train_mlx import decays, save, targets  # noqa: E402
from variant import role  # noqa: E402


# ---------------------------------------------------------------- fake quantization (forward = shipped values)

def ste(w, q):
    return w + mx.stop_gradient(q - w)


def fp16_round(x):
    return x.astype(mx.float16).astype(mx.float32)


def fq_b243(w, group):
    """Least-squares arbitrary ternary per group of `group` along K (rows = outputs), fp16 scale."""
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    mag = mx.abs(g)
    srt = -mx.sort(-mag, axis=-1)                      # descending
    csum = mx.cumsum(srt, axis=-1)
    counts = mx.arange(1, group + 1).astype(mx.float32)
    best = mx.argmax(csum * csum / counts, axis=-1)    # index of k-1
    kth = mx.take_along_axis(srt, best[..., None], axis=-1)
    scale = mx.take_along_axis(csum, best[..., None], axis=-1) / (best[..., None].astype(mx.float32) + 1)
    q = mx.where(mag >= kth, mx.sign(g), 0.0) * fp16_round(scale)
    return q.reshape(n, k)


def fq_t34(w, group):
    """Exact 3-of-4: zero the smallest magnitude of every quad; scale = survivors' mean |w| per group."""
    n, k = w.shape
    quads = w.reshape(n, k // 4, 4)
    mag = mx.abs(quads)
    zero = mx.argmin(mag, axis=-1)
    keep = mx.arange(4)[None, None, :] != zero[..., None]
    signs = mx.where(quads >= 0, 1.0, -1.0) * keep
    g_mag = (mag * keep).reshape(n, k // group, group)
    scale = g_mag.sum(-1, keepdims=True) / (0.75 * group)
    q = signs.reshape(n, k // group, group) * fp16_round(scale)
    return q.reshape(n, k)


def fq_int8(w):
    scale = mx.max(mx.abs(w)) / 127.0
    return mx.clip(mx.round(w / scale), -127, 127) * scale


def quantizer(fmt: str, group: int, select: str = "all", protect: str = "fp16"):
    fit = fq_b243 if fmt == "b243" else fq_t34

    def apply(p: dict) -> dict:
        out = {}
        for name, w in p.items():
            r = role(name, select, protect)
            if r == "ternary":
                out[name] = ste(w, fit(w, group))
            elif r == "int8":
                out[name] = ste(w, fq_int8(w))
            else:
                out[name] = ste(w, fp16_round(w))
        return out
    return apply


# ---------------------------------------------------------------- training (the v2 loop, with the quantized forward)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=["b243", "t34"], required=True)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--select", default="all", help="allocation: which eligible matrices are ternary (ptq/variant.py)")
    parser.add_argument("--protect", choices=["fp16", "int8"], default="fp16", help="format of the unselected matrices")
    parser.add_argument("--teacher", help="toolkit checkpoint to distil from (v2)")
    parser.add_argument("--kl", type=float, default=1.0)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float)
    parser.add_argument("--val", required=True)
    parser.add_argument("--val-limit", type=int, default=40_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", choices=["S", "M", "L"], default="L")
    parser.add_argument("--loss", choices=["soft", "set"], default="soft")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--ply-floor", type=int, default=20_000)
    parser.add_argument("--ply-min-weight", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init", help="toolkit checkpoint dir to start from (omit: from scratch)")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    corpora = [load_corpus([d]) for d in args.train]
    samplers = [PlySampler(b, args.ply_floor, args.seed + i, args.ply_min_weight) for i, (b, _) in enumerate(corpora)]
    sizes = np.array([len(b) for b, _ in corpora], dtype=np.float64)
    share = np.array(args.weights, dtype=np.float64) if args.weights else sizes
    share = share / share.sum()

    vb = np.load(Path(args.val) / "board.npy", mmap_mode="r")
    vs = np.load(Path(args.val) / "scores.npy", mmap_mode="r")
    pick = np.sort(np.random.default_rng(1).permutation(len(vb))[: args.val_limit])
    vb, vs = np.asarray(vb[pick]), np.asarray(vs[pick])

    if args.init:
        torch_model, _, config = load_checkpoint(Path(args.init), "cpu")
    else:
        config = config_for(args.size)
        torch_model, _ = make_system(config, "cpu")
    params = X.from_torch(torch_model.state_dict())
    n_params = parameter_count(torch_model)
    option_ids = mx.array(E.option_ids_table().astype(np.int32))
    quantize = quantizer(args.format, args.group, args.select, args.protect)
    teacher = None
    if args.teacher:
        t_model, _, t_config = load_checkpoint(Path(args.teacher), "cpu")
        teacher = (X.from_torch(t_model.state_dict()), t_config)

    def loss_fn(p, ctx, legal, scores, t_logp):
        logits = X.forward(quantize(p), config, ctx, option_ids, legal)
        t = targets(scores, args.loss, args.tau)
        logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        loss = -(t * mx.where(legal, logp, 0.0)).sum(1).mean()
        if teacher is not None:
            pt = mx.exp(t_logp)
            loss = loss + args.kl * (pt * mx.where(legal, t_logp - logp, 0.0)).sum(1).mean()
        return loss

    def teacher_logp(ctx, legal):
        tl = X.forward(teacher[0], teacher[1], ctx, option_ids, legal)
        return mx.stop_gradient(tl - mx.logsumexp(tl, axis=1, keepdims=True))

    def lr_at(step: int) -> float:
        warm = min(1.0, (step + 1) / args.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        return args.lr * warm * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)

    optimizer = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.95])
    step_fn = mx.compile(mx.value_and_grad(loss_fn))
    t_fn = mx.compile(teacher_logp) if teacher is not None else None
    decay_mask = {k: decays(k, v) for k, v in params.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train-log.jsonl").open("a")
    meta = {"trainer": "mlx-ternary", "args": vars(args), "params": n_params, "config": config,
            "train_rows": [int(s) for s in sizes], "share": share.tolist()}
    print(json.dumps(meta), flush=True)
    log.write(json.dumps(meta) + "\n")
    log.flush()

    def evaluate(p) -> dict:
        q = quantize(p)
        logits = []
        for i in range(0, len(vb), 8192):
            ctx, _, legal = X.inputs(vb[i:i + 8192])
            logits.append(np.array(X.forward(q, config, ctx, option_ids, legal)))
        return move_metrics(np.concatenate(logits), vb, vs)

    zeros = mx.zeros((args.batch, 7))
    best_vp, running, started = -1.0, None, time.time()
    for step in range(args.steps):
        which = rng.choice(len(corpora), size=args.batch, p=share)
        boards, scores = [], []
        for c in range(len(corpora)):
            n = int((which == c).sum())
            if n:
                idx = np.sort(samplers[c](n))
                boards.append(np.asarray(corpora[c][0][idx]))
                scores.append(np.asarray(corpora[c][1][idx]))
        b = np.concatenate(boards)
        s = np.concatenate(scores)
        flip = rng.random(len(b)) < 0.5
        b[flip] = E.mirror_boards(b[flip])
        s[flip] = s[flip][:, ::-1]
        ctx, _, legal = X.inputs(b)
        lr = lr_at(step)
        optimizer.learning_rate = lr
        t_logp = t_fn(ctx, legal) if t_fn is not None else zeros
        loss, grads = step_fn(params, ctx, legal, mx.array(s), t_logp)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        params = {k: (v * (1 - lr * args.wd) if decay_mask[k] else v) for k, v in params.items()}
        params = optimizer.apply_gradients(grads, params)
        mx.eval(params, optimizer.state, loss)
        value = loss.item()
        running = value if running is None else 0.98 * running + 0.02 * value
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            report = evaluate(params)   # the quantized model's metrics
            report.update({"step": step + 1, "loss": round(running, 4), "lr": lr, "elapsed_s": round(time.time() - started, 1)})
            if report["value_preserving"] > best_vp:
                best_vp = report["value_preserving"]
                report["parity"] = save(out / "model", params, config,
                                        {"step": step + 1, "value_preserving_quantized": best_vp, "trainer": "mlx-ternary",
                                         "format": args.format, "group": args.group, "select": args.select,
                                         "protect": args.protect, "latent": True,
                                         "train": [Path(t).name for t in args.train]}, vb[:512])
            save(out / "last", params, config, {"step": step + 1, "trainer": "mlx-ternary", "latent": True}, vb[:64])
            print(json.dumps(report), flush=True)
            log.write(json.dumps(report) + "\n")
            log.flush()
    print(json.dumps({"done": True, "best_value_preserving_quantized": best_vp}), flush=True)


if __name__ == "__main__":
    main()
