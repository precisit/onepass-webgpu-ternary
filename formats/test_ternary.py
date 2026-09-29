"""Exhaustive and randomised tests for formats/ternary.py; cross-checked against the reference codecs of the
earlier Base243 / Sherry T34 work when those files are present.

    python -m pytest formats/test_ternary.py -q     (or: python formats/test_ternary.py)
"""
from __future__ import annotations

import importlib.util
import itertools
import os
from pathlib import Path

import numpy as np

import ternary as T

REF_B243 = Path(os.environ.get("REF_B243", "/nonexistent"))  # optional: an earlier Base243 reference codec (test file)
REF_T34 = Path(os.environ.get("REF_T34", "/nonexistent"))  # optional: an earlier T34 compact-table reference codec


def load(path: Path):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


ALL5 = np.array(list(itertools.product([-1, 0, 1], repeat=5)), dtype=np.int8)


def test_b243_all_243_tuples_round_trip():
    codes = T.b243_encode5(ALL5)
    assert len(set(codes.tolist())) == 243
    assert np.array_equal(T.b243_decode5(codes), ALL5)


def test_b243_all_256_bytes():
    decoded = T.b243_decode5(np.arange(256, dtype=np.uint8))
    again = T.b243_encode5(decoded)
    canonical = again == np.arange(256)
    assert canonical.sum() == 243
    assert T.ALIASES_B243.tolist() == [1, 20, 40, 60, 79, 99, 119, 138, 158, 178, 197, 217, 237]
    assert np.array_equal(np.flatnonzero(~canonical), T.ALIASES_B243)


def b243_reference():
    """encode5 / decode5 from the reference test file, without importing it (it needs MLX)."""
    import ast
    import types

    if not REF_B243.exists():
        return None
    tree = ast.parse(REF_B243.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("encode5", "decode5", "as_trits", "as_codes")]
    mod = types.ModuleType("ref_b243")
    mod.np = np
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(REF_B243), "exec"), mod.__dict__)
    if not hasattr(mod, "as_trits"):
        mod.as_trits = lambda x: np.asarray(x, dtype=np.int8)
    if not hasattr(mod, "as_codes"):
        mod.as_codes = lambda x: np.asarray(x, dtype=np.uint8)
    return mod


def test_b243_matches_reference():
    ref = b243_reference()
    if ref is None:
        print("   (Base243 reference not present, skipped)")
        return
    assert np.array_equal(ref.encode5(ALL5), T.b243_encode5(ALL5))
    all_bytes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(ref.decode5(all_bytes), T.b243_decode5(all_bytes))


def legal_quads():
    out = []
    for zero in range(4):
        for signs in itertools.product([-1, 1], repeat=3):
            q = list(signs)
            q.insert(zero, 0)
            out.append(q)
    return np.array(out, dtype=np.int8)


def test_t34_all_32_states():
    quads = legal_quads()
    states = T.t34_encode_quads(quads)
    assert sorted(states.tolist()) == list(range(32))
    assert np.array_equal(T.t34_decode_states(states), quads)
    assert np.array_equal(T.t34_encode_quads(T.T34_TABLE), np.arange(32))


def test_t34_rejects_illegal_quads():
    for bad in ([0, 0, 1, 1], [1, 1, 1, 1], [2, 0, 1, 1]):
        try:
            T.t34_encode_quads(np.array([bad], dtype=np.int8))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def random_t34(rng, n, k):
    return legal_quads()[rng.integers(0, 32, size=(n, k // 4))].reshape(n, k)


def test_t34_matches_reference_g64_records():
    ref = load(REF_T34)
    if ref is None or not hasattr(ref, "encode_t34_states"):
        return
    rng = np.random.default_rng(1)
    groups = getattr(ref, "GROUPS", None)
    if groups is None:
        return
    trits = random_t34(rng, 3, 64 * groups)
    want = ref.encode_t34_states(trits.reshape(3, groups, 64))
    got = T.pack_t34(trits, 64)
    assert np.array_equal(want, got)


def test_group_packing_round_trips():
    rng = np.random.default_rng(0)
    for n, k, g in [(4, 256, 128), (8, 1024, 128), (3, 256, 64), (5, 512, 256), (2, 250, 125), (2, 130, 5)]:
        trits = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
        codes = T.pack_b243(trits, g)
        assert codes.shape == (n, k // g, T.b243_bytes_per_group(g))
        assert not np.isin(codes, T.ALIASES_B243).any()
        assert np.array_equal(T.unpack_b243(codes, g), trits)
        if g % 32 == 0:
            t = random_t34(rng, n, k)
            c = T.pack_t34(t, g)
            assert c.shape == (n, k // g, T.t34_bytes_per_group(g))
            assert np.array_equal(T.unpack_t34(c, g), t)


def brute_best(group_w: np.ndarray, candidates: np.ndarray) -> float:
    best = np.inf
    for t in candidates:
        nz = t != 0
        if not nz.any():
            err = float((group_w ** 2).sum())
        else:
            s = float((np.abs(group_w) * nz).sum() / nz.sum())
            if (np.sign(group_w)[nz] != t[nz]).any():
                s = float((group_w * t).sum() / (t * t).sum())  # general least squares for any sign pattern
            err = float(((group_w - s * t) ** 2).sum())
        best = min(best, err)
    return best


def test_fits_are_least_squares_optimal_before_fp16():
    rng = np.random.default_rng(2)
    all8 = np.array(list(itertools.product([-1, 0, 1], repeat=8)), dtype=np.int8)
    q = legal_quads()
    t34_8 = np.array([np.concatenate([a, b]) for a in q for b in q], dtype=np.int8)
    for _ in range(20):
        w = rng.normal(size=(1, 8))
        tb, sb = T.fit_b243(w, 8)
        err_b = float(((w - T.dequant(tb, sb.astype(np.float32), 8)) ** 2).sum())
        assert err_b <= brute_best(w[0], all8) * (1 + 2e-3) + 1e-6
        tt, st = T.fit_t34(w, 8)
        err_t = float(((w - T.dequant(tt, st.astype(np.float32), 8)) ** 2).sum())
        assert err_t <= brute_best(w[0], t34_8) * (1 + 2e-3) + 1e-6


def test_fit_methods_valid_and_least_squares_is_best():
    # every method gives valid trits (T34: exactly one zero per quad) and packs; least squares has the lowest
    # weight error of the three for every group (up to the fp16 rounding of the scales)
    rng = np.random.default_rng(3)
    for group in (64, 128, 256):
        w = rng.standard_normal((8, 512)) * rng.uniform(0.01, 0.2, (8, 1))
        for fit, pack, unpack in ((T.fit_b243, T.pack_b243, T.unpack_b243), (T.fit_t34, T.pack_t34, T.unpack_t34)):
            err = {}
            for method in T.FIT_METHODS:
                trits, scales = fit(w, group, method)
                assert set(np.unique(trits)) <= {-1, 0, 1}
                if fit is T.fit_t34:
                    assert ((trits.reshape(8, -1, 4) == 0).sum(-1) == 1).all()
                assert np.array_equal(unpack(pack(trits, group), group), trits)
                d = T.dequant(trits, scales, group) - w
                err[method] = (d.reshape(8, -1, group) ** 2).sum(-1)
            for method in ("absmean", "lloydmax"):
                assert (err["ls"] <= err[method] * (1 + 2e-3) + 1e-12).all(), (fit.__name__, group, method)


def test_bits_per_weight():
    assert T.bits_per_weight("base243", 128) == {"codes": 1.625, "codes_and_scales": 1.75}
    assert T.bits_per_weight("t34", 128) == {"codes": 1.25, "codes_and_scales": 1.375}


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if failed else 0)
