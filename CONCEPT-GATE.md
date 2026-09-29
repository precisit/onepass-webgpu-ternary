# CONCEPT-GATE: ternary Connect Four models (pre-registered before the first B1 run)

This file is hashed (`CONCEPT-GATE.sha256`) before the first B1 run. Amendments are dated and additive,
appended at the end, and only after asking the project owner. A cell that fails is reported as failed.

## What is tested

The v2 Connect Four model (toolkit `TinyTransformerScorer`, 8 x 256, 7 384 576 parameters, `main-r0-L`)
with its matrix weights in one of two ternary formats. v2 stays unchanged as the dense baseline; every
result here is a separately named derivative.

The core grid, 2 formats x 3 regimes. All six cells are always run and always reported:

| | Base243 (arbitrary ternary, 5 trits per byte) | T34 (exact 3-of-4 structure, 5 bits per 4 weights) |
| --- | --- | --- |
| 1. post-training conversion of v2 (PTQ) | `v2-ternary-b243-ptq` | `v2-ternary-t34-ptq` |
| 2. ternary-aware training initialised from v2 (QAT) | `v2-ternary-b243-qat` | `v2-ternary-t34-qat` |
| 3. trained ternary from scratch, same data and recipe | `v3-ternary-b243` | `v3-ternary-t34` |

### Default allocation (the six core cells)

- Ternary: the 36 encoder matrices (per layer: fused q/k/v in-projection, out-projection, MLP up, MLP down;
  8 context layers and the option-encoder layer), 7 077 888 weights.
- Group scales: one fp16 scale per group of 128 consecutive weights along K (the input dimension) for one
  output column.
- Base243: the Base243 codec of the MLX Base243 spec (trit t in {-1, 0, 1} as digit t + 1, five digits per
  byte, most significant first, byte = ceil(256 s / 243), the 13 alias bytes rejected). Each group of 128 is
  stored byte-aligned in ceil(128 / 5) = 26 bytes; the last byte carries 3 trits and two zero trits.
- T34: the Sherry T34 state (bits 1:0 = position of the zero in the quad, bits 4:2 = signs of the three
  survivors in K order, 0 = +1). The published record holds 64 weights (16 states in 10 bytes, LSB first, plus
  an fp16 scale: 12 bytes); at G128 a record holds 32 states in 20 bytes plus one fp16 scale (22 bytes).
- int8 with a per-tensor scale (as in the published int8 file): the scoring head's q, k, v matrices and the
  byte embedding.
- fp16: positions, biases, layer norms.

Byte budget of the tensors under this allocation: Base243 1 899 281 bytes (1.75 bits per ternary weight
including scales, 2.06 bits per parameter over the whole model); T34 1 567 505 bytes (1.375 and 1.70).

### Regime defaults

- **PTQ:** per-group least-squares fit, then the scale rounded to fp16. Base243: exhaustive over the size of
  the non-zero set (the k largest magnitudes, scale = their mean). T34: in each quad the smallest magnitude is
  zeroed (optimal for any positive scale), scale = mean magnitude of the survivors. No training.
- **QAT:** initialised from `main-r0-L`; fake-quantised forward pass in the exact target format (Base243:
  arbitrary ternary with the group scales; T34: exact 3-of-4 pattern with the group scales), straight-through
  gradient; loss = the v2 training loss on the v2 corpora plus KL divergence to v2's logits; at most 12 000
  steps.
- **v3 from scratch:** random initialisation, the same fake-quantised forward pass, the v2 data, loss and
  schedule (6 000 + 12 000 steps, as `g2-L-soft` then `main-r0-L`), no distillation.

The exact commands and settings of every run are recorded next to its results. Extra variants (group size,
fitting method, per-layer sensitivity, other allocations) are reported as additional, labelled cells; they
never replace a core cell.

## The file

"The file" is the ONNX file a browser would download for the model: the ternary matrices as uint8
initializers consumed by the custom op `com.precisit.TernaryMatMul`, with a model-local `FunctionProto`
reference decomposition. Its size is counted in bytes on disk, uncompressed, and it must:

1. run in ONNX Runtime (CPU) through the reference decomposition, and
2. produce the same choices as the dense-weight evaluation of the same cell on the whole eval set
   (the file must be the model that was evaluated).

The graph may be simplified and renamed (any valid ONNX model qualifies); its bytes count. Until the file
exists, a size is reported as "predicted" and cannot pass the gate.

## The concept gate (lenient, per cell)

A cell passes when all of these hold:

1. **Size:** the file is smaller than 2 000 000 bytes.
2. **Beats v1 head to head:** against the v1 demo model (`onnx-v1:onepass-c4-8x24.onnx`, played as the demo
   plays it), under the PROTOCOL-v2 game rules (200 games, empty board, 0.05 noise on both players, colours
   alternating, seed 2026), score > 0.5.
3. **Beats v1 against the depth-4 bot:** the cell's score against `bot:4` under the same rules, and v1's score
   against `bot:4` measured in the same harness for this gate, have non-overlapping Wilson 95 % intervals,
   with the cell above.

## Always reported (no gate)

- The Strong bars of PROTOCOL-v2 (production reference): score vs `bot:4` (>= 0.85), vs `bot:6` (>= 0.70),
  value-preserving rate on the eval set (>= 0.97), with intervals.
- Eval-set metrics per ply band, agreement with v2's choices, the PROTOCOL-v2 controls.
- Honest bits per weight: ternary codes alone; codes plus scales; every stored bit over all parameters.
- File size, gzip size, and the size of the graph without its tensors.

## Stop rule

When all six core cells are measured, the grid is summarised (pass or fail, bits per weight, file sizes)
and reported to the project owner before any further work on these models and before anything is published.
