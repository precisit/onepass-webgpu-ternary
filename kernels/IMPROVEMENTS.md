# Known further improvements to the ternary kernels

Written 2026-09-29. None of this is needed for the Connect Four demo; it is the list to pick from when a workload
makes the kernels matter.

## How the kernels work now

`ternary-formats.js` plugs into onepass-webgpu's split-K matmul by replacing only its inner loop (the `inner`
hook). Per group of four K positions and per output column, the T34 loop:

1. reads the 5-bit state from two bytes (shift and mask; a state can straddle a byte boundary);
2. splits it into the zero's position (2 bits) and the three survivors' signs (3 bits);
3. builds each of the four weights as 0 or ±1 (the sign bit is the position's rank among the survivors) and
   multiplies it by the group's scale (fp16, loaded once per 128 weights);
4. accumulates `acc = fma(x, w, acc)` for each of the RM = 4 rows, exactly as with fp32 weights.

So the arithmetic is an ordinary float multiply-accumulate, including a multiply by 0 for the zero weight. The
ternary structure is only used to store the weights compactly. Base243 is the same, with a x3 digit decode per
byte instead of the bit arithmetic.

## Measured (idle M5 Pro, `results/records/speed/2026-09-28-M5Pro-chrome-ternary.json`)

| weights | per move, batch 1 | GPU time, batch 1 | batch 64, per position |
| --- | ---: | ---: | ---: |
| f32 | 1.30 ms | 0.98 ms | 0.32 ms |
| int8 (weight only) | 0.90 ms | 0.66 ms | 0.29 ms |
| T34 | 1.10 ms | 0.79 ms | 0.38 ms |
| Base243 | 1.30 ms | 1.05 ms | 0.52 ms |

At batch 1 the time goes to 70 dispatches and the read-back, so the decode barely shows. At batch 64 the GPU does
real arithmetic, and the decode cost per weight shows: T34 is 18 % slower than f32 per position, Base243 63 %.

## When it matters

| regime | example | bound by | what helps |
| --- | --- | --- | --- |
| small model, one position | this demo (7.4 M, batch 1) | dispatches, read-back | nothing below; the runtime's layout already did it |
| large model, one position | a 100 M+ scorer, or an LLM decoding one token | memory bandwidth | ternary itself: fewer bytes read is faster, if the decode stays cheaper than the memory time (the per-record decode does) |
| many positions at once | an arena or self-play running 100 AIs in parallel; longer inputs or more options (larger M) | arithmetic | items 1 to 5 below |

In the third regime every workgroup that covers a block of rows decodes the same weights again, so the decode is
repeated M / RM times. That is the cost to cut.

## Improvements, in the order we would try them

1. **Take the scale out of the inner loop.** Accumulate ±x (skip 0) into a per-group partial and multiply by the
   group's scale once per group and row: one multiply less per weight. The K split can start or end inside a group,
   so flush the partial at the existing group-change point (`kk == 0 || r == 0`) and at the end of the split.
   Cheap to try; applies to both formats.
2. **More rows per thread at large batch** (RM 8 or 16), or larger row tiles, so each decoded weight feeds more
   activations. Needs the tuning heuristic to pick RM by M.
3. **Decode once per workgroup into shared memory**, then a standard tiled matmul on the decoded tile: the usual
   design for quantized GEMM at large M (llama.cpp, MLX). Worth it only well into regime 3.
4. **Table decode.** A 32-entry table from T34 state to four signs (it already exists for the `w4` path) instead of
   the rank arithmetic; for Base243 a 256 x 5 table (1.3 KB) instead of the sequential x3 digits. Measure against
   the ALU decode; constant-memory lookups are not free either.
5. **Integer path (the BitNet b1.58 model).** Quantize each row's activations to int8, expand the weights to int8
   {-1, 0, +1}, and use `dot4I8Packed` (WGSL `packed_4x8_integer_dot_product`) for four multiply-accumulates per
   instruction with integer accumulation; apply the weight scale times the activation scale at the group end. The
   largest possible gain in regime 3, but it changes the model's arithmetic: it needs its own reference and an
   eval-set gate (as the int8 path did), not bit-for-bit parity with the float reference.
6. **Add/subtract instead of multiply** (select +x or -x, skip 0) only pays where an add is cheaper than an fma: CPU
   or WebAssembly SIMD ports, or dedicated hardware. On a GPU an fma costs the same as an add, and a per-weight
   branch to skip zeros costs more than multiplying by 0.
7. **Base243 for speed:** a 2-bit container decodes much faster (26-41 % faster than Base243 on Metal in our
   earlier measurements) at 2 bits instead of 1.6 per weight. A size/speed trade, not a free win.

## Gates for any change

- Items 1-4 and 6 must keep bit-level argmax parity with the file's FunctionProto reference decode on all 17 325
  eval positions (`tests/parity.html` with `plugins=`), as the current kernels do.
- Speed: SPEED-PROTOCOL rows with the plugin (`bench/run_bench.py --backend NAME=QUERY`, see `scripts/run-speed-and-sweeps.sh`),
  at batch 1 and batch 64 in the same run as f32 and int8. The batch-64 column is the target: T34 at or below f32
  (0.32 ms per position) would be the first milestone.
