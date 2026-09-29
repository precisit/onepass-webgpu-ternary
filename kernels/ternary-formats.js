// onepass-webgpu weight-format plugins for ternary matmul weights (see WeightFormat in onepass-webgpu).
// Layout (formats/ternary.py): codes uint8 [N, K/G, B] per output column n and group g; scales fp16 [N, K/G].
// pack(): codes are uploaded as-is; scales become float side data; params = [G, B].
//
//   import { formats } from "./ternary-formats.js";
//   const engine = await Engine.load(plan, onnxBytes, { formats });

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

function halfTensor(init) {
  if (init.dataType !== 10) throw new Error(`ternary: ${init.name} must be float16`);
  const h = new Uint16Array(init.bytes.byteLength / 2);
  new Uint8Array(h.buffer).set(init.bytes);
  return Float32Array.from(h, halfToFloat);
}

function packGroups(t, inits) {
  const codes = inits.get(t.initializer);
  const scales = inits.get(t.format.scales);
  if (!codes || !scales) throw new Error(`ternary: missing initializers for ${t.initializer}`);
  if (codes.dataType !== 2) throw new Error(`ternary: ${t.initializer} must be uint8`);
  const [n, groups, bytesPerGroup] = codes.dims;
  const [k, nOut] = t.shape;
  if (nOut !== n || groups * t.format.group !== k) throw new Error(`ternary: ${t.initializer} shape does not match the plan`);
  return { bytes: codes.bytes, floats: halfTensor(scales), params: [t.format.group, bytesPerGroup] };
}

// Base243: five trits per byte, most significant first; digit j of a byte is found by repeating
// state *= 3; trit = (state >> 8) - 1; state &= 255 (j + 1 times).
const base243 = {
  kind: "base243",
  pack: packGroups,
  inner: true,
  wgsl: `
// the K-split loop: scales once per group, one byte per 5 rows, digits by the x3 decoder
fn inner(k0: u32, n4: u32, acc: ptr<function, array<vec4<f32>, RM>>) {
  let G = p.x0;
  let B = p.x1;
  let groups = p.K / G;
  let n0 = n4 * 4u;
  var sc = vec4<f32>(0.0);
  var st = vec4<u32>(0u);
  for (var kk = 0u; kk < KS; kk += 1u) {
    let k = k0 + kk;
    let g = k / G;
    let r = k % G;
    let j = r % 5u;
    if (kk == 0u || r == 0u) {
      sc = vec4<f32>(fval(n0 * groups + g), fval((n0 + 1u) * groups + g), fval((n0 + 2u) * groups + g), fval((n0 + 3u) * groups + g));
    }
    if (kk == 0u || j == 0u) {
      let b = r / 5u;
      st = vec4<u32>(qbyte((n0 * groups + g) * B + b), qbyte(((n0 + 1u) * groups + g) * B + b),
                     qbyte(((n0 + 2u) * groups + g) * B + b), qbyte(((n0 + 3u) * groups + g) * B + b));
      for (var i = 0u; i < j; i += 1u) { st = (st * 3u) & vec4<u32>(255u); }
    }
    st = st * 3u;
    let w = (vec4<f32>(st >> vec4<u32>(8u)) - vec4<f32>(1.0)) * sc;
    st = st & vec4<u32>(255u);
    for (var rr = 0u; rr < RM; rr += 1u) { (*acc)[rr] = fma(vec4<f32>(at[rr * KS + kk]), w, (*acc)[rr]); }
  }
}
fn b243_trit(code: u32, j: u32) -> f32 {
  var s = code;
  var t = 0u;
  for (var i = 0u; i <= j; i += 1u) { s = s * 3u; t = s >> 8u; s = s & 255u; }
  return f32(i32(t) - 1);
}
fn w4(k: u32, n4: u32) -> vec4<f32> {
  let G = p.x0;
  let B = p.x1;
  let groups = p.K / G;
  let g = k / G;
  let r = k % G;
  var out: vec4<f32>;
  for (var c = 0u; c < 4u; c += 1u) {
    let rec = (n4 * 4u + c) * groups + g;
    out[c] = b243_trit(qbyte(rec * B + r / 5u), r % 5u) * fval(rec);
  }
  return out;
}`,
};

// T34: one 5-bit state per quad (bits 1:0 = the zero's position, bits 4:2 = survivor signs), LSB first.
const t34 = {
  kind: "t34",
  pack: packGroups,
  inner: true,
  wgsl: `
// the K-split loop: scales once per group, one state per quad and column, signs by bit arithmetic
fn inner(k0: u32, n4: u32, acc: ptr<function, array<vec4<f32>, RM>>) {
  let G = p.x0;
  let B = p.x1;
  let groups = p.K / G;
  let n0 = n4 * 4u;
  var sc = vec4<f32>(0.0);
  for (var kk = 0u; kk < KS; kk += 4u) {
    let k = k0 + kk;
    let g = k / G;
    let r = k % G;
    if (kk == 0u || r == 0u) {
      sc = vec4<f32>(fval(n0 * groups + g), fval((n0 + 1u) * groups + g), fval((n0 + 2u) * groups + g), fval((n0 + 3u) * groups + g));
    }
    let bit = 5u * (r / 4u);
    var st: vec4<u32>;
    for (var c = 0u; c < 4u; c += 1u) {
      let addr = ((n0 + c) * groups + g) * B + (bit >> 3u);
      st[c] = ((qbyte(addr) | (qbyte(addr + 1u) << 8u)) >> (bit & 7u)) & 31u;
    }
    let z = st & vec4<u32>(3u);
    let signs = st >> vec4<u32>(2u);
    for (var i = 0u; i < 4u; i += 1u) {
      let iv = vec4<u32>(i);
      let rank = iv - select(vec4<u32>(0u), vec4<u32>(1u), iv > z);
      let negative = ((signs >> rank) & vec4<u32>(1u)) == vec4<u32>(1u);
      let w = select(select(vec4<f32>(1.0), vec4<f32>(-1.0), negative), vec4<f32>(0.0), z == iv) * sc;
      for (var rr = 0u; rr < RM; rr += 1u) { (*acc)[rr] = fma(vec4<f32>(at[rr * KS + kk + i]), w, (*acc)[rr]); }
    }
  }
}
var<private> T34: array<vec4<f32>, 32> = array<vec4<f32>, 32>(vec4<f32>(0.0, 1.0, 1.0, 1.0),vec4<f32>(1.0, 0.0, 1.0, 1.0),vec4<f32>(1.0, 1.0, 0.0, 1.0),vec4<f32>(1.0, 1.0, 1.0, 0.0),vec4<f32>(0.0, -1.0, 1.0, 1.0),vec4<f32>(-1.0, 0.0, 1.0, 1.0),vec4<f32>(-1.0, 1.0, 0.0, 1.0),vec4<f32>(-1.0, 1.0, 1.0, 0.0),vec4<f32>(0.0, 1.0, -1.0, 1.0),vec4<f32>(1.0, 0.0, -1.0, 1.0),vec4<f32>(1.0, -1.0, 0.0, 1.0),vec4<f32>(1.0, -1.0, 1.0, 0.0),vec4<f32>(0.0, -1.0, -1.0, 1.0),vec4<f32>(-1.0, 0.0, -1.0, 1.0),vec4<f32>(-1.0, -1.0, 0.0, 1.0),vec4<f32>(-1.0, -1.0, 1.0, 0.0),vec4<f32>(0.0, 1.0, 1.0, -1.0),vec4<f32>(1.0, 0.0, 1.0, -1.0),vec4<f32>(1.0, 1.0, 0.0, -1.0),vec4<f32>(1.0, 1.0, -1.0, 0.0),vec4<f32>(0.0, -1.0, 1.0, -1.0),vec4<f32>(-1.0, 0.0, 1.0, -1.0),vec4<f32>(-1.0, 1.0, 0.0, -1.0),vec4<f32>(-1.0, 1.0, -1.0, 0.0),vec4<f32>(0.0, 1.0, -1.0, -1.0),vec4<f32>(1.0, 0.0, -1.0, -1.0),vec4<f32>(1.0, -1.0, 0.0, -1.0),vec4<f32>(1.0, -1.0, -1.0, 0.0),vec4<f32>(0.0, -1.0, -1.0, -1.0),vec4<f32>(-1.0, 0.0, -1.0, -1.0),vec4<f32>(-1.0, -1.0, 0.0, -1.0),vec4<f32>(-1.0, -1.0, -1.0, 0.0));
fn w4(k: u32, n4: u32) -> vec4<f32> {
  let G = p.x0;
  let B = p.x1;
  let groups = p.K / G;
  let g = k / G;
  let r = k % G;
  let bit = 5u * (r / 4u);
  var out: vec4<f32>;
  for (var c = 0u; c < 4u; c += 1u) {
    let rec = (n4 * 4u + c) * groups + g;
    let at = rec * B + (bit >> 3u);
    let v = qbyte(at) | (qbyte(at + 1u) << 8u);
    out[c] = T34[(v >> (bit & 7u)) & 31u][r % 4u] * fval(rec);
  }
  return out;
}`,
};

export const formats = [base243, t34];
