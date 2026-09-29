# onepass-webgpu-ternary

Ternary weights for a small game-playing model: two packing formats (**T34** and **Base243**), training with the
format in the loop, a compact ONNX form, and WebGPU kernels for the
[onepass-webgpu](https://github.com/precisit/onepass-webgpu) runtime.

The model is the Connect Four one-pass scorer v2 (7.4 M parameters) from
[one-pass-specialists](https://github.com/precisit/one-pass-specialists/tree/main/examples/c4). Its best ternary
version is a **1.59 MB** file (the dense int8 file is 7.8 MB) that plays at the dense model's level.

**Try it:** [the Size demo](https://precisit.github.io/onepass-web/demo/c4-size/) runs the models in your browser.

**The story:** [A game-playing AI in 1.6 MB](https://precisit.com/en/blog/onepass-c4-size/)
(also [in Swedish](https://precisit.com/blog/onepass-c4-size/)).

## Results

Game scores: 200 games from the empty board, colours alternating, both sides playing a random move 5 % of the time,
a win counts 1 and a draw ½. "Value-preserving" is the share of 17 325 held-out positions where the model's move
keeps the game's exact outcome (solver-graded).

| model | file | vs depth-4 bot | vs depth-6 bot | value-preserving | vs a perfect player |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 dense (fp32 file) | 29.7 MB | 0.915 | 0.893 | 0.987 | 0.475 |
| **T34, from scratch, longer final stage** | **1.59 MB** | **0.930** | **0.910** | **0.989** | 0.427 |
| T34, from scratch (grid run) | 1.59 MB | 0.945 | 0.870 | 0.985 | 0.410 |
| T34, from scratch (second seed) | 1.59 MB | 0.882 | 0.892 | 0.984 | 0.427 |
| T34, fine-tuned from v2 (QAT) | 1.59 MB | 0.885 | 0.895 | 0.986 | 0.438 |
| T34, converted after training (PTQ) | 1.59 MB | 0.128 | 0.107 | 0.860 | 0.048 |
| Base243, from scratch | 1.93 MB | 0.885 | 0.877 | 0.986 | 0.427 |
| Base243, fine-tuned from v2 (QAT) | 1.93 MB | 0.915 | 0.880 | 0.987 | 0.485 |
| Base243, converted after training (PTQ) | 1.93 MB | 0.547 | 0.512 | 0.920 | 0.105 |

- **Training with the format in the loop recovers everything; converting a finished model does not.** All trained
  models reach the dense model's level; the converted ones fall short.
- **Seeds matter:** two training runs of the same T34 recipe score 0.945 and 0.882 against the depth-4 bot
  (0.87-0.95 over three game seeds each), while agreeing on the held-out positions (0.985 and 0.984).
- **Converting without training:** the least-squares scale is too small for the model's function. With the same
  zeros, a Lloyd-Max scale takes T34 from 0.13 to 0.81 against the depth-4 bot, and 1.2 x the least-squares scale
  takes Base243 from 0.55 to 0.81. Attention's q/k/v matrices are the sensitive ones; group size barely matters.
- **Every file is the model that was evaluated:** ONNX Runtime running each file chooses the same move as the
  evaluated model on all 17 325 positions, and the WebGPU runtime matches each file's reference decode on all
  17 325.
- **Speed** (idle Apple M5 Pro, Chrome 154, the onepass-webgpu speed protocol): T34 1.1 ms per move, Base243 1.3 ms,
  against dense int8 0.9 ms and f32 1.3 ms. Ternary shrinks the download; the T34 kernel is not optimized for its
  format yet ([`kernels/IMPROVEMENTS.md`](kernels/IMPROVEMENTS.md)).

Full tables: [`results/GRID-SUMMARY.md`](results/GRID-SUMMARY.md) (the pre-registered six-model grid and the extra
runs) and [`results/SWEEP-SUMMARY.md`](results/SWEEP-SUMMARY.md) (60 training-free variants). Every raw record is in
[`results/records/`](results/records/).

## The formats

- **T34** is the 3:4 structured ternary format of Sherry (Huang et al., ACL 2026,
  [arXiv:2601.07892](https://arxiv.org/abs/2601.07892)): in every group of four weights exactly one is zero and three
  are -1 or +1, so 32 states in 5 bits. With one fp16 scale per 128 weights: 1.375 bits per weight.
- **Base243** is our name for base-3 packing, five ternary values per byte (3^5 = 243 < 256), as in llama.cpp's
  `TQ1_0` ([PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151), compilade). Our layout, 128 values in
  26 bytes plus an fp16 scale (1.75 bits per weight), is the same as prism-ml's GGUF `PTQ1_0`.
- The byte embedding and the scoring head's q/k/v are int8; positions, biases and norms fp16. With them, the T34
  tensors are 1.69 bits per parameter; the whole file, including the ONNX graph, 1.73.

## Layout

| path | what |
| --- | --- |
| [`CONCEPT-GATE.md`](CONCEPT-GATE.md) (+ `.sha256`) | the gate, allocation and regimes, frozen before the first run |
| [`SWEEP-PLAN.md`](SWEEP-PLAN.md) | the post-training sweeps, written before they ran (with one dated post-hoc addendum) |
| `formats/` | the codecs, closed-form least-squares fits and other fitting rules, bits-per-weight accounting, exhaustive tests |
| `ptq/variant.py` | turns a float state into any allocation (formats, group sizes, fits, matrix selectors) |
| `train/train_ternary.py` | MLX training with the exact format in the forward pass (straight-through estimator), QAT or from scratch, optional distillation |
| `export/` | the compact ONNX export (custom ops with FunctionProto reference decodes), verification against the evaluated model, runtime plans |
| `kernels/` | the onepass-webgpu weight-format plugins (WGSL) and the list of known kernel improvements |
| `scripts/` | the scripts that produced the results, with their environment (`scripts/env.sh`) |
| `results/` | the summaries, their generators, every raw record, and the exported models with their WebGPU plans |

## Reproduce

The scripts expect the Connect Four work area of the recipe (corpora, the v2 checkpoint, the solver labeller) and
a checkout of [one-pass-specialists](https://github.com/precisit/one-pass-specialists) (`examples/c4`), as described
in `scripts/env.sh`.

```sh
python formats/test_ternary.py                                   # codecs and fits
python ptq/variant.py --checkpoint <v2> --format t34 --out runs/t34-ptq     # convert
python export/export_onnx.py runs/t34-ptq && python export/verify_onnx.py runs/t34-ptq --evalset <evalset-v2>
bash scripts/run-v3-from-scratch.sh                              # train both formats from scratch, then evaluate
```

In the browser:

```js
import { Engine } from "onepass-webgpu.js";
import { formats } from "./kernels/ternary-formats.js";
const engine = await Engine.load(plan, onnxBytes, { formats });
```

## Credits

Sherry (Tencent) for the T34 format; compilade's `TQ1_0` for base-3 packing in llama.cpp; BitNet b1.58 (Ma et al.,
[arXiv:2402.17764](https://arxiv.org/abs/2402.17764)) for ternary language models; Lloyd (1982) and Max (1960) for
the Lloyd-Max quantizer; Bengio, Léonard and Courville (2013) for the straight-through estimator; Hinton, Vinyals
and Dean (2015) for distillation; [MLX](https://github.com/ml-explore/mlx) for training;
[connect-four-ai](https://github.com/benjaminrall/connect-four-ai) (Benjamin Rall) for the solver labels.

## License

MIT, see [`LICENSE`](LICENSE).
