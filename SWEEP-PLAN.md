# SWEEP-PLAN: cheap post-training sweeps around the ternary grid, and ternary speed

Written 2026-09-28, before the sweep runs. It does not change CONCEPT-GATE.md or
PROTOCOL-v2. The sweeps are measurements, not gate cells: no sweep variant replaces a grid cell. A promising
variant is reported as a candidate for a later, separately named cell.

## Questions

1. Which layer families break first when v2 is made ternary after training (PTQ), and how much of the PTQ
   loss does each family carry? Does that differ between Base243 and T34?
2. How much does the group size matter (G64 / G128 / G256) at this model size?
3. Does the fitting method matter (least squares, absmean, Lloyd-Max), given that least squares is already
   the weight-error optimum per group?
4. Does protecting attention (int8 instead of ternary) recover PTQ, and after training (QAT) is there
   anything left to recover?
5. How fast are the ternary files on an idle machine, next to f32 and int8 in the same run?

## Fixed for every sweep variant

- Source: v2 (`main-r0-L`), the same checkpoint as the grid. Tooling: `ptq/variant.py`, which writes a
  toolkit checkpoint holding exactly the values the packed file would hold.
- Non-ternary roles as in the default allocation: int8 per-tensor for the head q/k/v and the byte
  embedding, fp16 for positions, biases and norms.
- Measurements per variant:
  - eval set `evalset-v2` (17 325 positions): value-preserving rate, value-preserving non-trivial, exact
    top-1 (`eval_c4.py score`, no controls);
  - one game cell: vs `bot:4`, seed 2026, 200 games, PROTOCOL-v2 rules (0.05 noise, empty board), with its
    95 % Wilson interval;
  - predicted tensor bytes and bits per parameter (`variant.json`). Sweep variants are not exported to
    ONNX; a variant promoted to a cell is exported and verified as the grid cells were.
- Reference rows: v2 dense; the grid's PTQ cells (G128, least squares, all 36 matrices).

## S1: group size (both formats, least squares, all 36 matrices)

G64, G128 (re-run of the grid PTQ cell, a reproducibility check), G256.

## S2: fitting method (both formats, G128, all 36 matrices)

- `ls`: least squares (the grid). Base243: for every count k the k largest magnitudes are non-zero with the
  mean as scale, and the best k wins. T34: the smallest magnitude of each quad is zero (forced by the
  format), and the scale is the survivors' mean magnitude.
- `absmean`: the BitNet b1.58 rule. Scale = mean |w| of the group. Base243: trit = round(clip(w / scale,
  -1, 1)). T34: the same forced zeros, with the absmean scale.
- `lloydmax`: the Gaussian 3-level Lloyd-Max quantizer with sigma = the group's RMS. Base243: zero below
  0.612 sigma, level 1.224 sigma. T34: the same forced zeros, with the level 1.224 sigma as scale.

For T34 only the scale differs between the methods, because the format fixes which weights are zero.

## S3: sensitivity (both formats, G128, least squares)

Selectors over the 36 matrices. Types: `qkv` (fused in-projection), `out` (out-projection), `up`, `down`
(MLP), each in all 9 layers. Position: `L0`..`L7` (a context layer, all four matrices) and `opt` (the
option-encoder layer). Unselected matrices stay at fp16 (the v2 values).

- `none`: no ternary matrices (only the int8 / fp16 roles), which measures their own loss.
- only one family ternary: `only:qkv`, `only:out`, `only:up`, `only:down`, `only:opt`.
- all but one family ternary: `except:qkv`, `except:out`, `except:up`, `except:down`, `except:opt`.
- depth, one pair of context layers ternary: `only:L0,L1`, `only:L2,L3`, `only:L4,L5`, `only:L6,L7`.

## S4: allocations after training-free conversion (both formats, G128, least squares)

- `attn-int8`: attention in-projections and out-projections int8 (per tensor), MLP ternary. About 3.5 MB of
  tensors, over the gate's 2 MB. It is a research point: what protecting attention buys.
- If S3 finds a single dominant family, the same family protected at int8.

## S5: combinations

The best group size from S1 with the best fitting method from S2, per format, if either improves on the
grid's G128 least squares.

## QAT with protected attention (a separate, follow-up booking)

The grid's QAT recipe unchanged (initialised from v2, KL 1.0 to v2, 12 000 steps, same data), with the
`attn-int8` allocation in the training forward pass, both formats at once. That is about 3 h of training
(the grid's QAT took 10 750 s), then the full PROTOCOL-v2 evaluation as for a grid cell. The question: after
training, does protecting attention recover anything beyond the default allocation (grid QAT: bot:4 0.915
Base243 / 0.885 T34, value-preserving 0.9873 / 0.9863; v2 0.915 / 0.9872)? It does not fit the 2 h sweep
booking, so it gets its own booking (about 4 h).

## Speed (SPEED-PROTOCOL, an idle Apple M5 Pro, the same run as f32 and int8)

- Rows: onepass-webgpu f32 and int8 (the public rows, repeated as the in-run reference), and T34 v3 and
  Base243 v3 with the ternary plugin (f32 activations).
- Each ternary row checks its own file first: 50 positions against ONNX Runtime CPU on the same file
  (FunctionProto reference decode), exact argmax required.
- The public bench gets generic options only: a model file, a plugin module and a reference directory
  from the query string. The ternary plugin, models and records stay in this repository.

## Outputs

- `results/records/sweep/*.jsonl`: raw records, one line per measurement.
- `results/SWEEP-SUMMARY.md`: generated by `results/make_sweep_summary.py`.
- `results/records/speed/`: the speed record with the ternary rows.
- Updated: the GRID-SUMMARY sweep section and the Size demo page.

## Addendum 2026-09-28 (post-hoc, after S1-S5)

S2 found that the Lloyd-Max scale beats least squares, strongly for T34 (vs bot:4 0.805 against 0.128, with the
same zeros), although least squares has the lower weight error by construction. S6 is added after seeing that
result, to find out why; it is labelled post-hoc wherever it is reported.

- `ls*1.1` .. `ls*1.4`: the least-squares fit with its scale multiplied (same zeros).
- `rms`: least-squares zeros, energy-preserving scale (the quantized group has the weights' norm).
- Lloyd-Max on the q/k/v matrices only (least squares elsewhere), and everywhere except q/k/v.

For Gaussian weights the scales are about 0.96 sigma (T34 least squares), 1.13 sigma (energy-preserving) and
1.19 sigma (Lloyd-Max level), so the multipliers bracket Lloyd-Max. Same measurements as S1-S5.
