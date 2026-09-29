"""Ternary weight formats for onepass-webgpu: Base243 and Sherry T34, with per-group fp16 scales.

Layout convention (both formats). A matmul weight is the [K, N] matrix of `x @ W` (K = inputs, N = outputs).
It is stored per output column n, in groups of G consecutive weights along K:

    codes   uint8  [N, K / G, B]   B = bytes per group (Base243: ceil(G / 5); T34: 5 G / 32)
    scales  fp16   [N, K / G]      value = trit * scale

Base243 (arbitrary ternary): five trits per byte. Trit t in {-1, 0, 1} is the digit t + 1; digits are
combined most significant first, s = ((((d0 3 + d1) 3 + d2) 3 + d3) 3 + d4), stored as ceil(256 s / 243).
Decoding repeats five times: state *= 3; trit = (state >> 8) - 1; state &= 255. Thirteen byte values
never occur (the aliases); a group whose length is not a multiple of 5 is padded with zero trits.

T34 (exact 3-of-4): in every quad of four consecutive weights exactly one is zero and three are +-1. A
quad is a 5-bit state: bits 1:0 = position of the zero, bits 4:2 = signs of the three survivors in K order
(0 = +1). States are packed LSB first into a bit stream per group (G = 64: 10 bytes, G = 128: 20 bytes).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- Base243

def b243_encode5(trits: np.ndarray) -> np.ndarray:
    """[..., 5] trits in {-1, 0, 1} -> [...] uint8 codes."""
    d = np.asarray(trits, dtype=np.int32) + 1
    if d.shape[-1] != 5 or d.min(initial=0) < 0 or d.max(initial=2) > 2:
        raise ValueError("expected [..., 5] trits in {-1, 0, 1}")
    s = (((d[..., 0] * 3 + d[..., 1]) * 3 + d[..., 2]) * 3 + d[..., 3]) * 3 + d[..., 4]
    return ((256 * s + 242) // 243).astype(np.uint8)


def b243_decode5(codes: np.ndarray) -> np.ndarray:
    """[...] uint8 codes -> [..., 5] int8 trits (the arithmetic decoder the kernels use)."""
    state = np.asarray(codes, dtype=np.uint16).copy()
    out = []
    for _ in range(5):
        state *= 3
        out.append(((state >> 8).astype(np.int16) - 1).astype(np.int8))
        state &= 255
    return np.stack(out, axis=-1)


CANONICAL_B243 = b243_encode5(np.array(np.meshgrid(*[[-1, 0, 1]] * 5, indexing="ij")).reshape(5, -1).T)
ALIASES_B243 = np.setdiff1d(np.arange(256), CANONICAL_B243).astype(np.uint8)


def b243_bytes_per_group(group: int) -> int:
    return -(-group // 5)


def pack_b243(trits: np.ndarray, group: int) -> np.ndarray:
    """trits [N, K] int8 -> codes [N, K / G, ceil(G / 5)] uint8."""
    n, k = trits.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the group {group}")
    g = trits.reshape(n, k // group, group)
    pad = b243_bytes_per_group(group) * 5 - group
    if pad:
        g = np.concatenate([g, np.zeros((n, k // group, pad), dtype=g.dtype)], axis=-1)
    return b243_encode5(g.reshape(n, k // group, -1, 5))


def unpack_b243(codes: np.ndarray, group: int) -> np.ndarray:
    """codes [N, K / G, B] -> trits [N, K] int8. Rejects alias bytes."""
    if np.isin(codes, ALIASES_B243).any():
        raise ValueError("Base243 stream contains alias bytes")
    n, groups, _ = codes.shape
    t = b243_decode5(codes).reshape(n, groups, -1)[..., :group]
    return np.ascontiguousarray(t.reshape(n, groups * group))


# ---------------------------------------------------------------- T34

def t34_encode_quads(quads: np.ndarray) -> np.ndarray:
    """[..., 4] legal T34 quads (exactly one zero, others +-1) -> [...] uint8 5-bit states."""
    q = np.asarray(quads, dtype=np.int8)
    zero_mask = q == 0
    if q.shape[-1] != 4 or not np.all(zero_mask.sum(-1) == 1) or not np.all(np.abs(q) <= 1):
        raise ValueError("quad violates exact T34 (one zero, three +-1)")
    zero = zero_mask.argmax(-1).astype(np.uint8)
    signs = np.zeros(zero.shape, dtype=np.uint8)
    for column in range(4):
        rank = column - (column > zero)
        negative = (zero != column) & (q[..., column] < 0)
        signs |= np.where(negative, np.left_shift(np.uint8(1), rank.astype(np.uint8)), 0).astype(np.uint8)
    return (zero | (signs << 2)).astype(np.uint8)


def _t34_table() -> np.ndarray:
    table = np.zeros((32, 4), dtype=np.int8)
    for state in range(32):
        zero, signs = state & 3, state >> 2
        rank = 0
        for column in range(4):
            if column == zero:
                continue
            table[state, column] = -1 if (signs >> rank) & 1 else 1
            rank += 1
    return table


T34_TABLE = _t34_table()


def t34_decode_states(states: np.ndarray) -> np.ndarray:
    return T34_TABLE[np.asarray(states, dtype=np.uint8)]


def t34_bytes_per_group(group: int) -> int:
    if group % 32:
        raise ValueError("T34 groups must hold a multiple of 32 weights (8 quads = 5 bytes)")
    return 5 * group // 32


def pack_t34(trits: np.ndarray, group: int) -> np.ndarray:
    """trits [N, K] int8 (exact T34) -> codes [N, K / G, 5 G / 32] uint8, states LSB first."""
    n, k = trits.shape
    if k % group:
        raise ValueError(f"K={k} is not a multiple of the group {group}")
    states = t34_encode_quads(trits.reshape(n, k // group, group // 4, 4)).astype(np.uint64)
    q = group // 4
    out = np.zeros((n, k // group, t34_bytes_per_group(group)), dtype=np.uint8)
    for i in range(q):
        bit = 5 * i
        byte, shift = bit >> 3, bit & 7
        v = states[..., i] << np.uint64(shift)
        out[..., byte] |= (v & np.uint64(255)).astype(np.uint8)
        if shift > 3:
            out[..., byte + 1] |= (v >> np.uint64(8)).astype(np.uint8)
    return out


def unpack_t34(codes: np.ndarray, group: int) -> np.ndarray:
    n, groups, nbytes = codes.shape
    if nbytes != t34_bytes_per_group(group):
        raise ValueError("T34 record size does not match the group")
    c = codes.astype(np.uint16)
    states = np.empty((n, groups, group // 4), dtype=np.uint8)
    for i in range(group // 4):
        bit = 5 * i
        byte, shift = bit >> 3, bit & 7
        v = c[..., byte] >> shift
        if shift > 3:
            v = v | (c[..., byte + 1] << (8 - shift))
        states[..., i] = (v & 31).astype(np.uint8)
    return np.ascontiguousarray(t34_decode_states(states).reshape(n, groups * group))


# ---------------------------------------------------------------- post-training fit (least squares per group)

# Gaussian 3-level Lloyd-Max quantizer (unit variance): zero below THRESHOLD, reconstruction level LEVEL.
LLOYD_MAX_THRESHOLD = 0.6120
LLOYD_MAX_LEVEL = 1.2240
FIT_METHODS = ("ls", "absmean", "lloydmax")


def fit_b243(w: np.ndarray, group: int, method: str = "ls") -> tuple[np.ndarray, np.ndarray]:
    """w [N, K] float -> (trits [N, K] int8, scales [N, K / G] fp16).
    ls: exact least squares per group (for every k, the k largest magnitudes are non-zero with scale = their
    mean; the best k wins). absmean: BitNet b1.58 (scale = mean |w|, trit = round(clip(w / scale, -1, 1))).
    lloydmax: Gaussian Lloyd-Max with sigma = the group's RMS (zero below 0.612 sigma, level 1.224 sigma)."""
    n, kk = w.shape
    g = np.asarray(w, dtype=np.float64).reshape(n, kk // group, group)
    if method in ("absmean", "lloydmax"):
        if method == "absmean":
            scale = np.abs(g).mean(-1)
            trits = np.clip(np.rint(g / np.maximum(scale, 1e-30)[..., None]), -1, 1)
        else:
            sigma = np.sqrt((g ** 2).mean(-1))
            trits = np.sign(g) * (np.abs(g) > LLOYD_MAX_THRESHOLD * sigma[..., None])
            scale = LLOYD_MAX_LEVEL * sigma
        return trits.astype(np.int8).reshape(n, kk), scale.astype(np.float16)
    if method != "ls":
        raise ValueError(f"unknown fit method {method}")
    mag = np.abs(g)
    order = np.argsort(-mag, axis=-1)
    sorted_mag = np.take_along_axis(mag, order, -1)
    csum = np.cumsum(sorted_mag, -1)
    counts = np.arange(1, group + 1)
    # SSE(k) = sum w^2 - (sum top-k |w|)^2 / k  -> maximise (csum_k)^2 / k
    best_k = np.argmax(csum ** 2 / counts, axis=-1) + 1
    scale = np.take_along_axis(csum, (best_k - 1)[..., None], -1)[..., 0] / best_k
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(group), order.shape), -1)
    keep = rank < best_k[..., None]
    trits = (np.sign(g) * keep).astype(np.int8)
    return trits.reshape(n, kk), scale.astype(np.float16)


def fit_t34(w: np.ndarray, group: int, method: str = "ls") -> tuple[np.ndarray, np.ndarray]:
    """w [N, K] float -> (trits exact T34, scales fp16). In every quad the smallest magnitude is zeroed
    (optimal for any positive scale), so only the scale depends on the method. ls: the survivors' mean
    magnitude (the least-squares optimum). absmean: mean |w| of the group. lloydmax: 1.224 x the group's RMS."""
    n, kk = w.shape
    q = np.asarray(w, dtype=np.float64).reshape(n, kk // 4, 4)
    zero = np.abs(q).argmin(-1)
    signs = np.where(q >= 0, 1, -1).astype(np.int8)
    np.put_along_axis(signs, zero[..., None], 0, -1)
    trits = signs.reshape(n, kk)
    g = np.asarray(w, dtype=np.float64).reshape(n, kk // group, group)
    t = trits.reshape(n, kk // group, group)
    if method == "ls":
        scale = (np.abs(g) * (t != 0)).sum(-1) / (t != 0).sum(-1)
    elif method == "absmean":
        scale = np.abs(g).mean(-1)
    elif method == "lloydmax":
        scale = LLOYD_MAX_LEVEL * np.sqrt((g ** 2).mean(-1))
    else:
        raise ValueError(f"unknown fit method {method}")
    return trits, scale.astype(np.float16)


def dequant(trits: np.ndarray, scales: np.ndarray, group: int) -> np.ndarray:
    """trits [N, K], scales [N, K / G] -> float32 [N, K]."""
    n, k = trits.shape
    s = np.asarray(scales, dtype=np.float32)
    return (trits.reshape(n, k // group, group).astype(np.float32) * s[..., None]).reshape(n, k)


def bits_per_weight(fmt: str, group: int) -> dict:
    code = 8 * (b243_bytes_per_group(group) if fmt == "base243" else t34_bytes_per_group(group)) / group
    return {"codes": code, "codes_and_scales": code + 16 / group}
