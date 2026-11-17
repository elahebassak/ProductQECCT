"""
Classical baseline decoders evaluated with the *same* noise model / code / metrics as the neural decoders.

    bposd : belief propagation + ordered-statistics decoding (ldpc package, Roffe et al.)
    bp    : belief propagation only (min-sum), failure if no convergence (counted in LER_strict)
    mwpm  : minimum-weight perfect matching (PyMatching). Only valid when every qubit is in
            <= 2 checks of each type (toric / surface codes).

X errors are decoded with H_Z, Z errors with H_X, independently (the standard CSS decoding of
BP+OSD / MWPM; it ignores X-Z correlations under depolarizing noise).

    python eval_classical.py --code hgp-400 --noise depolarizing --err_types xz --decoder bposd \
        --osd_order 10 --max_iter 100 --p_test 0.01:0.10:9 --samples 100000 --workers 8 --out bposd_hgp400.json
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
from train import parse_p_list

_G = {}


def _import_ldpc():
    """Import the ldpc package; if its optional stim/sinter dependencies are missing, stub them
    (they are only needed for Stim-circuit decoding, which we do not use)."""
    try:
        import ldpc  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name not in ("stim", "sinter"):
            raise
        import types
        sys.modules.setdefault("stim", types.ModuleType("stim"))
        sinter = types.ModuleType("sinter")
        sinter.Decoder = object; sinter.CompiledDecoder = object
        sys.modules.setdefault("sinter", sinter)
        import ldpc  # noqa: F401
    from ldpc import BpOsdDecoder, BpDecoder
    return BpOsdDecoder, BpDecoder


def _make_decoder(kind, H, p_bit, args):
    if kind == "mwpm":
        from pymatching import Matching
        return Matching(H)
    BpOsdDecoder, BpDecoder = _import_ldpc()
    H = H.astype(np.uint8)
    if kind == "bposd":
        return BpOsdDecoder(H, error_rate=float(p_bit), max_iter=args.max_iter, bp_method=args.bp_method,
                            ms_scaling_factor=args.ms_scaling, schedule=args.schedule,
                            osd_method=args.osd_method, osd_order=args.osd_order)
    if kind == "bp":
        return BpDecoder(H, error_rate=float(p_bit), max_iter=args.max_iter, bp_method=args.bp_method,
                         ms_scaling_factor=args.ms_scaling, schedule=args.schedule)
    raise ValueError(kind)


def _decode_chunk(task):
    """Decode one chunk of shots. Returns counts."""
    p, n_shots, seed = task
    code, args = _G["code"], _G["args"]
    sampler = NoiseSampler(code, args.noise, args.err_types, device="cpu")
    rng = np.random.default_rng(seed)
    e_x, e_z, s_x, s_z = sampler.sample_batch_numpy(p, n_shots, rng)
    # per-qubit flip probability of each error type
    p_bit = p if args.noise == "independent" else 2.0 * p / 3.0
    out = dict(samples=n_shots, bit_errors=0.0, bits=0)
    fail_any = np.zeros(n_shots, dtype=bool)
    fail_strict = np.zeros(n_shots, dtype=bool)
    t0 = time.time()
    if args.decoder == "bposd_corr":
        return _decode_chunk_correlated(code, args, p, n_shots, e_x, e_z, s_x, s_z, out, t0)
    for typ in ("x", "z"):
        if typ not in args.err_types:
            continue
        if typ == "x":
            H, L, e, s = code.Hz, code.Lz, e_x, s_z
        else:
            H, L, e, s = code.Hx, code.Lx, e_z, s_x
        dec = _make_decoder(args.decoder, H, p_bit, args)
        e_hat = np.zeros_like(e)
        for i in range(n_shots):
            e_hat[i] = np.asarray(dec.decode(s[i].astype(np.uint8)), dtype=np.uint8)[: code.n]
        r = (e_hat ^ e).astype(np.int64)
        lflip = ((r @ L.T.astype(np.int64)) % 2).any(1)
        sres = ((r @ H.T.astype(np.int64)) % 2).any(1)
        out[f"ler_{typ}"] = float(lflip.sum())
        out[f"ler_{typ}_strict"] = float((lflip | sres).sum())
        out[f"synd_mismatch_{typ}"] = float(sres.sum())
        out["bit_errors"] += float((e_hat != e).sum()); out["bits"] += n_shots * code.n
        fail_any |= lflip
        fail_strict |= (lflip | sres)
    out["ler"] = float(fail_any.sum())
    out["ler_strict"] = float(fail_strict.sum())
    out["time"] = time.time() - t0
    return out


def _decode_chunk_correlated(code, args, p, n_shots, e_x, e_z, s_x, s_z, out, t0):
    """Depolarizing-aware BP+OSD: one decoder over 3n error columns (X-only | Z-only | Y on each qubit),
    each with prior p/3, on the stacked syndrome [s_x; s_z].  An X error is seen by H_Z, a Z error by
    H_X and a Y error by both, so the check matrix is
            H3 = [[ 0  , H_X, H_X ],
                  [ H_Z, 0  , H_Z ]]      (rows: X-checks then Z-checks)
    and the decoded (x, z, y) gives e_x = x ^ y, e_z = z ^ y.  Unlike separate CSS decoding, this uses
    the X-Z correlation of depolarizing noise (the same information the neural decoders can exploit)."""
    assert args.noise == "depolarizing" and args.err_types == "xz", "bposd_corr is for depolarizing noise with --err_types xz"
    n = code.n
    Z_x = np.zeros((code.rx, n), dtype=np.uint8)
    Z_z = np.zeros((code.rz, n), dtype=np.uint8)
    H3 = np.block([[Z_x, code.Hx, code.Hx], [code.Hz, Z_z, code.Hz]]).astype(np.uint8)
    BpOsdDecoder, _ = _import_ldpc()
    dec = BpOsdDecoder(H3, error_channel=[p / 3.0] * (3 * n), max_iter=args.max_iter, bp_method=args.bp_method,
                       ms_scaling_factor=args.ms_scaling, schedule=args.schedule,
                       osd_method=args.osd_method, osd_order=args.osd_order)
    s3 = np.concatenate([s_x, s_z], axis=1).astype(np.uint8)
    ex_hat = np.zeros_like(e_x); ez_hat = np.zeros_like(e_z)
    for i in range(n_shots):
        e3 = np.asarray(dec.decode(s3[i]), dtype=np.uint8)
        x, z, y = e3[:n], e3[n:2 * n], e3[2 * n:]
        ex_hat[i] = x ^ y; ez_hat[i] = z ^ y
    fail_any = np.zeros(n_shots, dtype=bool); fail_strict = np.zeros(n_shots, dtype=bool)
    for typ, H, L, e, e_hat in (("x", code.Hz, code.Lz, e_x, ex_hat), ("z", code.Hx, code.Lx, e_z, ez_hat)):
        r = (e_hat ^ e).astype(np.int64)
        lflip = ((r @ L.T.astype(np.int64)) % 2).any(1)
        sres = ((r @ H.T.astype(np.int64)) % 2).any(1)
        out[f"ler_{typ}"] = float(lflip.sum())
        out[f"ler_{typ}_strict"] = float((lflip | sres).sum())
        out[f"synd_mismatch_{typ}"] = float(sres.sum())
        out["bit_errors"] += float((e_hat != e).sum()); out["bits"] += n_shots * n
        fail_any |= lflip; fail_strict |= (lflip | sres)
    out["ler"] = float(fail_any.sum()); out["ler_strict"] = float(fail_strict.sum())
    out["time"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", type=str, required=True)
    ap.add_argument("--noise", type=str, default="depolarizing", choices=["independent", "depolarizing"])
    ap.add_argument("--err_types", type=str, default="xz", choices=["x", "z", "xz"])
    ap.add_argument("--decoder", type=str, default="bposd", choices=["bposd", "bp", "mwpm", "bposd_corr"],
                    help="bposd_corr = depolarizing-aware BP+OSD on the joint X/Z/Y check matrix")
    ap.add_argument("--p_test", type=str, required=True)
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--bp_method", type=str, default="minimum_sum", choices=["minimum_sum", "product_sum"])
    ap.add_argument("--ms_scaling", type=float, default=0.625)
    ap.add_argument("--schedule", type=str, default="parallel", choices=["parallel", "serial"])
    ap.add_argument("--osd_method", type=str, default="OSD_CS", choices=["OSD_0", "OSD_E", "OSD_CS"])
    ap.add_argument("--osd_order", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tag", type=str, default="", help="suffix for the model name in the results (e.g. a BP setting)")
    args = ap.parse_args()

    try:
        if args.decoder == "mwpm":
            import pymatching  # noqa: F401
        else:
            _import_ldpc()
    except Exception as e:  # noqa
        raise SystemExit(f"missing dependency for decoder '{args.decoder}': {e!r}\n"
                         "On Nibi run:  source ~/pqecct_env/bin/activate && pip install stim sinter pymatching 'ldpc>=2.1'")
    code = get_code(args.code)
    print(code.summary(), flush=True)
    if args.decoder == "mwpm":
        assert code.Hx.sum(0).max() <= 2 and code.Hz.sum(0).max() <= 2, "MWPM needs a matchable (toric/surface) code"
    _G["code"] = code; _G["args"] = args
    p_list = parse_p_list(args.p_test)
    results = {}
    t_all = time.time()
    for pi, p in enumerate(p_list):
        n_chunks = int(np.ceil(args.samples / args.chunk))
        tasks = [(p, min(args.chunk, args.samples - c * args.chunk), args.seed + 100000 * pi + c) for c in range(n_chunks)]
        if args.workers > 1:
            with Pool(args.workers) as pool:
                outs = pool.map(_decode_chunk, tasks)
        else:
            outs = [_decode_chunk(t) for t in tasks]
        acc = {}
        for o in outs:
            for k, v in o.items():
                acc[k] = acc.get(k, 0.0) + v
        S = acc["samples"]
        res = {k: v / S for k, v in acc.items() if k not in ("samples", "bits", "bit_errors", "time")}
        res["ber"] = acc["bit_errors"] / max(acc["bits"], 1)
        res["samples"] = int(S)
        res["decode_time_per_shot_us"] = 1e6 * acc["time"] / S
        results[f"{p:.6f}"] = res
        print(f"{args.decoder} p={p:.4f}: LER={res['ler']:.3e} LER_strict={res['ler_strict']:.3e} BER={res['ber']:.3e} "
              f"(n={S}, {res['decode_time_per_shot_us']:.0f} us/shot)", flush=True)
    out = dict(model=args.decoder + args.tag, code=args.code, noise=args.noise, err_types=args.err_types, n=code.n, k=code.k,
               results=results, args=vars(args), total_time_s=time.time() - t_all)
    out_path = args.out or f"classical_{args.decoder}_{args.code}_{args.noise}_{args.err_types}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
