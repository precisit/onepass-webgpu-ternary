"""PTQ sweeps of SWEEP-PLAN.md on the compute host: every variant is quantized from v2 (ptq/variant.py),
scored on the eval set (eval_c4.py score) and played against bot:4 (protocol.py, seed 2026, 200 games).

    source scripts/env.sh && $PY scripts/sweep.py [--jobs 4] [--only NAME ...] [--dry-run]

Records: $LOGS/sweep/<name>.jsonl (eval line, then game line) and <name>-variant.json. A variant whose
records are complete is skipped, so the script can be re-run after a failure. Phase 2 (S4b, S5) is decided
from phase 1's results by the rules in SWEEP-PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = os.environ
TR, PY, V2, W, RECIPE = (Path(ENV[k]) for k in ("TR", "PY", "V2", "W", "C4_RECIPE"))
RUNS = Path(ENV["RUNS"]) / "sweep"
LOGS = Path(ENV["LOGS"]) / "sweep"
FORMATS = ("b243", "t34")
FAMILIES = ("qkv", "out", "up", "down", "opt")


def v(name, fmt, group=128, fit="ls", select="all", protect="fp16", source=None):
    return {"name": name, "format": fmt, "group": group, "fit": fit, "select": select, "protect": protect, "source": source}


def phase1() -> list[dict]:
    out = [v("sweep-v2-dense", "none", source=str(V2)),
           v("sweep-none", "t34", select="none")]   # only the int8 / fp16 roles (the format is irrelevant)
    for f in FORMATS:
        out += [v(f"sweep-{f}-g{g}", f, group=g) for g in (64, 128, 256)]                          # S1
        out += [v(f"sweep-{f}-{m}", f, fit=m) for m in ("absmean", "lloydmax")]                   # S2
        out += [v(f"sweep-{f}-only-{t}", f, select=f"only:{t}") for t in FAMILIES]                # S3
        out += [v(f"sweep-{f}-except-{t}", f, select=f"except:{t}") for t in FAMILIES]
        out += [v(f"sweep-{f}-only-L{i}L{i + 1}", f, select=f"only:L{i},L{i + 1}") for i in (0, 2, 4, 6)]
        out += [v(f"sweep-{f}-attn-int8", f, select="except:qkv,out", protect="int8")]           # S4
    return out


def posthoc() -> list[dict]:
    """S6 (SWEEP-PLAN.md addendum, post-hoc): why the Lloyd-Max scale helps. Same zeros, other scales."""
    out = []
    for f in FORMATS:
        out += [v(f"sweep-{f}-ls-x{x}", f, fit=f"ls*{x}") for x in ("1.1", "1.2", "1.3", "1.4")]
        out += [v(f"sweep-{f}-rms", f, fit="rms"),
                v(f"sweep-{f}-lloydmax-qkv-only", f, fit="ls,qkv=lloydmax"),
                v(f"sweep-{f}-lloydmax-except-qkv", f, fit="lloydmax,qkv=ls")]
    return out


def result(name: str) -> dict | None:
    path = LOGS / f"{name}.jsonl"
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ev = next((x for x in lines if "value_preserving" in x and "a_score" not in x), None)
    game = next((x for x in lines if x.get("b") == "bot:4"), None)
    return {"vp": ev["value_preserving"], "bot4": game["a_score"]} if ev and game else None


def phase2(done: list[dict]) -> list[dict]:
    """S4b: the family whose single ternary conversion costs the most VP, protected at int8, if it dominates
    (its VP loss is at least half the sum over the five families). S5: the best group size with the best
    fitting method (by VP), if that combination is new."""
    out = []
    base = result("sweep-none")
    for f in FORMATS:
        loss = {t: base["vp"] - result(f"sweep-{f}-only-{t}")["vp"] for t in FAMILIES}
        worst = max(loss, key=loss.get)
        if loss[worst] >= 0.5 * sum(max(x, 0) for x in loss.values()):
            out.append(v(f"sweep-{f}-protect-{worst}-int8", f, select=f"except:{worst}", protect="int8"))
        g = max((64, 128, 256), key=lambda g: result(f"sweep-{f}-g{g}")["vp"])
        m = max(("ls", "absmean", "lloydmax"),
                key=lambda m: result(f"sweep-{f}-g128")["vp"] if m == "ls" else result(f"sweep-{f}-{m}")["vp"])
        if g != 128 and m != "ls":
            out.append(v(f"sweep-{f}-g{g}-{m}", f, group=g, fit=m))
    return out


def run(spec: dict, workers: int) -> str:
    name = spec["name"]
    if result(name):
        return f"{name}: already done"
    log = LOGS / f"{name}.jsonl"
    log.unlink(missing_ok=True)
    started = time.time()
    env = dict(ENV, OMP_NUM_THREADS="4")
    if spec["source"]:
        ckpt = Path(spec["source"])
    else:
        out = RUNS / name
        cmd = [str(PY), "ptq/variant.py", "--checkpoint", str(V2), "--format", spec["format"], "--group", str(spec["group"]),
               "--fit", spec["fit"], "--select", spec["select"], "--protect", spec["protect"], "--out", str(out), "--name", name]
        subprocess.run(cmd, cwd=TR, env=env, check=True, capture_output=True, text=True)
        (LOGS / f"{name}-variant.json").write_text((out / "variant.json").read_text())
        ckpt = out / "model"
    subprocess.run([str(PY), "eval_c4.py", "score", "--checkpoint", str(ckpt), "--evalset", str(W / "corpus/evalset-v2"),
                    "--device", "cpu", "--out", str(log)], cwd=RECIPE, env=env, check=True, capture_output=True, text=True)
    subprocess.run([str(PY), "protocol.py", "--a", f"model:{ckpt}", "--b", "bot:4", "--seed", "2026", "--games", "200",
                    "--workers", str(workers), "--out", str(log)], cwd=RECIPE, env=env, check=True, capture_output=True, text=True)
    r = result(name)
    return f"{name}: vp {r['vp']:.4f} bot4 {r['bot4']:.3f} ({time.time() - started:.0f} s)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--only", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--posthoc", action="store_true", help="run only S6")
    a = p.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    def batch(specs):
        specs = [s for s in specs if not a.only or s["name"] in a.only]
        if a.dry_run:
            for s in specs:
                print(json.dumps(s))
            return
        failures = []
        with ThreadPoolExecutor(a.jobs) as pool:
            futures = {pool.submit(run, s, 4): s["name"] for s in specs}
            for fut, name in futures.items():
                try:
                    print(fut.result(), flush=True)
                except subprocess.CalledProcessError as e:
                    failures.append(name)
                    print(f"{name}: FAILED {e.cmd[1]} {e.stderr[-400:] if e.stderr else ''}", flush=True)
        if failures:
            print(f"failed: {failures}", flush=True)

    if a.posthoc:
        batch(posthoc())
        print("POSTHOC DONE", flush=True)
        return
    first = phase1()
    batch(first)
    if a.dry_run or a.only:
        return
    if all(result(s["name"]) for s in first):
        second = phase2(first)
        print(f"phase 2: {[s['name'] for s in second] or 'nothing to add'}", flush=True)
        batch(second)
    else:
        print("phase 2 skipped: phase 1 incomplete", flush=True)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
