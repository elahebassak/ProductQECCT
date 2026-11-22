#!/usr/bin/env python
"""Decoding latency per codeword at batch size 1, one protocol for every decoder.

Protocol (identical for neural and classical decoders):
  * N syndromes are sampled once (depolarizing noise, X/Z decoded jointly, fixed seed) and stored in memory,
    in the decoder's input format (numpy uint8 for classical, torch tensors on the device for neural);
  * the timed region is syndrome -> hard correction for ONE codeword: for separate CSS decoders both the
    X and the Z decode, for the joint decoder the single 3n-column decode, for neural decoders the forward
    pass plus the threshold (logits > 0) and, on GPU, a device synchronization;
  * every decoder is constructed once and warmed up (20 codewords) before timing; N timed codewords follow;
  * classical decoders run single-threaded on the CPU (OMP/MKL threads = 1); neural decoders are timed on
    the GPU (fp32) and, for a like-for-like comparison, on one CPU thread as well;
  * reported: mean, median, 90th percentile (us/codeword). The amortized throughput at batch 512 is also
    recorded for the neural decoders.

Example (one GPU node):
  python tools/measure_latency.py --code toric-L8 --p 0.102941 --bposd_schedule serial
  python tools/measure_latency.py --code hgp-625  --p 0.041765 --bposd_schedule parallel
Writes results/latency_<code>.json and prints the LaTeX rows.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, platform, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
import eval_classical as ec


def stats(ts):
    ts = np.asarray(ts) * 1e6
    return dict(mean_us=float(ts.mean()), median_us=float(np.median(ts)), p90_us=float(np.percentile(ts, 90)),
                std_us=float(ts.std()), n=int(len(ts)))


def time_classical(kind, code, p, s_x, s_z, args):
    p_bit = 2.0 * p / 3.0                       # per-qubit flip probability of each error type (depolarizing)
    n = code.n
    N = s_x.shape[0]
    if kind == "bposd_corr":
        Z_x = np.zeros((code.rx, n), dtype=np.uint8); Z_z = np.zeros((code.rz, n), dtype=np.uint8)
        H3 = np.block([[Z_x, code.Hx, code.Hx], [code.Hz, Z_z, code.Hz]]).astype(np.uint8)
        BpOsdDecoder, _ = ec._import_ldpc()
        dec = BpOsdDecoder(H3, error_channel=[p / 3.0] * (3 * n), max_iter=args.max_iter, bp_method=args.bp_method,
                           ms_scaling_factor=args.ms_scaling, schedule=args.schedule_corr,
                           osd_method=args.osd_method, osd_order=args.osd_order)
        s3 = np.ascontiguousarray(np.concatenate([s_x, s_z], axis=1).astype(np.uint8))

        def decode(i):
            e3 = np.asarray(dec.decode(s3[i]), dtype=np.uint8)
            return e3[:n] ^ e3[2 * n:], e3[n:2 * n] ^ e3[2 * n:]
    else:
        a = argparse.Namespace(**vars(args)); a.schedule = args.bposd_schedule
        dec_x = ec._make_decoder(kind, code.Hz, p_bit, a)     # X errors from Z-checks
        dec_z = ec._make_decoder(kind, code.Hx, p_bit, a)     # Z errors from X-checks
        sz = np.ascontiguousarray(s_z.astype(np.uint8)); sx = np.ascontiguousarray(s_x.astype(np.uint8))

        def decode(i):
            ex = np.asarray(dec_x.decode(sz[i]), dtype=np.uint8)[:n]
            ez = np.asarray(dec_z.decode(sx[i]), dtype=np.uint8)[:n]
            return ex, ez
    for i in range(min(args.warmup, N)):
        decode(i)
    ts = []
    for i in range(N):
        t = time.perf_counter(); decode(i); ts.append(time.perf_counter() - t)
    return stats(ts)


def time_neural(model_name, code, run_dir, s_x, s_z, args, device):
    import torch
    from pqecct.models import build_model, count_params
    ck_path = os.path.join(run_dir, args.ckpt_name)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    margs = argparse.Namespace(**ck["args"])
    model = build_model(margs, code).to(device).eval()
    model.load_state_dict(ck["model"])
    sx = torch.as_tensor(s_x, dtype=torch.float32, device=device)
    sz = torch.as_tensor(s_z, dtype=torch.float32, device=device)
    N = sx.shape[0]
    cuda = device.type == "cuda"

    def decode(i):
        logits, _ = model(sx[i:i + 1], sz[i:i + 1])
        corr = logits > 0
        if cuda:
            torch.cuda.synchronize()
        return corr

    out = dict(params=count_params(model), epoch=ck["epoch"], ckpt=ck_path)
    with torch.no_grad():
        for i in range(min(args.warmup, N)):
            decode(i)
        ts = []
        for i in range(N):
            t = time.perf_counter(); decode(i); ts.append(time.perf_counter() - t)
        out.update(stats(ts))
        if not cuda:
            return out
        # CUDA-graph capture of the same batch-1 syndrome->correction path (removes kernel-launch overhead;
        # the timed region still includes copying the syndrome into the graph's input buffers)
        if args.cuda_graph:
            try:
                st_x = sx[:1].clone(); st_z = sz[:1].clone()
                side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model(st_x, st_z)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    g_logits, _ = model(st_x, st_z)
                    g_corr = g_logits > 0

                def decode_graph(i):
                    st_x.copy_(sx[i:i + 1]); st_z.copy_(sz[i:i + 1]); g.replay(); torch.cuda.synchronize()
                    return g_corr
                # correctness check: graph output == eager output on the first 50 codewords
                mism = 0
                for i in range(min(50, N)):
                    mism += int((decode_graph(i) != decode(i)).any())
                for i in range(min(args.warmup, N)):
                    decode_graph(i)
                ts = []
                for i in range(N):
                    t = time.perf_counter(); decode_graph(i); ts.append(time.perf_counter() - t)
                out["cuda_graph"] = stats(ts); out["cuda_graph"]["mismatch_vs_eager"] = mism
            except Exception as e:  # capture not supported for this model
                out["cuda_graph"] = dict(error=repr(e))
        # amortized throughput at a large batch (GPU only)
        bs = min(args.batch, N)
        for _ in range(3):
            model(sx[:bs], sz[:bs]); cuda and torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            logits, _ = model(sx[:bs], sz[:bs]); corr = logits > 0; cuda and torch.cuda.synchronize()
        out[f"amortized_us_b{bs}"] = (time.perf_counter() - t) / 5 / bs * 1e6
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--p", type=float, required=True, help="physical error rate at which syndromes are sampled")
    ap.add_argument("--n", type=int, default=1000, help="timed codewords")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--ckpt_name", default="best.pt")
    ap.add_argument("--neural", default="product,qecct,crossqecct")
    ap.add_argument("--classical", default="auto", help="comma list of mwpm,bp,bposd,bposd_corr or 'auto'")
    ap.add_argument("--cpu_neural", type=int, default=1, help="also time the neural decoders on one CPU thread")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cuda_graph", type=int, default=1, help="also time the GPU path captured in a CUDA graph")
    # BP settings (the paper reports the best per code; defaults = the settings used in the tables)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--bp_method", default="minimum_sum")
    ap.add_argument("--ms_scaling", type=float, default=0.625)
    ap.add_argument("--bposd_schedule", default="parallel", help="schedule of the separate BP / BP+OSD decoders")
    ap.add_argument("--schedule_corr", default="serial", help="schedule of the joint BP+OSD decoder")
    ap.add_argument("--osd_method", default="OSD_CS")
    ap.add_argument("--osd_order", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    code = get_code(args.code)
    sampler = NoiseSampler(code, "depolarizing", "xz", device="cpu")
    rng = np.random.default_rng(args.seed)
    e_x, e_z, s_x, s_z = sampler.sample_batch_numpy(args.p, args.n, rng)
    s_x = np.asarray(s_x, dtype=np.uint8); s_z = np.asarray(s_z, dtype=np.uint8)
    is_toric = args.code.startswith("toric")
    classical = (["mwpm", "bposd", "bposd_corr"] if is_toric else ["bp", "bposd", "bposd_corr"]) \
        if args.classical == "auto" else args.classical.split(",")

    import torch
    res = dict(code=args.code, n_qubits=code.n, noise="depolarizing", err_types="xz", p=args.p, n=args.n,
               warmup=args.warmup, seed=args.seed, protocol=__doc__.split("Example")[0].strip(),
               hardware=dict(cpu=platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip(),
                             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                             torch=torch.__version__, python=platform.python_version(),
                             omp_threads=os.environ.get("OMP_NUM_THREADS")),
               bp_settings=dict(max_iter=args.max_iter, bp_method=args.bp_method, ms_scaling=args.ms_scaling,
                                schedule_separate=args.bposd_schedule, schedule_joint=args.schedule_corr,
                                osd_method=args.osd_method, osd_order=args.osd_order),
               classical={}, neural={})
    for kind in classical:
        print(f"[classical] {kind} ...", flush=True)
        res["classical"][kind] = time_classical(kind, code, args.p, s_x, s_z, args)
        print(f"  {kind:12s} {res['classical'][kind]['mean_us']:10.1f} us (median {res['classical'][kind]['median_us']:.1f})", flush=True)

    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if args.cpu_neural:
        torch.set_num_threads(1); devices.append(torch.device("cpu"))
    for m in args.neural.split(","):
        run_dir = os.path.join(args.runs_root, f"{m}_{args.code}_depolarizing_xz_N6_d128_s42")
        if not os.path.exists(os.path.join(run_dir, args.ckpt_name)):
            print(f"[neural] {m}: no checkpoint in {run_dir}, skipped"); continue
        res["neural"][m] = {}
        for dev in devices:
            print(f"[neural] {m} on {dev.type} ...", flush=True)
            r = time_neural(m, code, run_dir, s_x, s_z, args, dev)
            res["neural"][m][dev.type] = r
            cg = r.get("cuda_graph", {})
            cg_s = f"; CUDA graph {cg['mean_us']:.1f} us (mismatch {cg['mismatch_vs_eager']})" if "mean_us" in cg else (f"; CUDA graph failed: {cg['error']}" if cg else "")
            print(f"  {m:12s} {dev.type}: {r['mean_us']:10.1f} us (median {r['median_us']:.1f}; batch-{args.batch} amortized {r.get(f'amortized_us_b{min(args.batch,args.n)}', float('nan')):.1f}{cg_s})", flush=True)

    out = args.out or os.path.join("results", f"latency_{args.code}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")

    # LaTeX rows (mean us/codeword)
    names = dict(mwpm="MWPM", bp="BP", bposd="BP+OSD (separate)", bposd_corr="BP+OSD (joint)",
                 qecct="QECCT", crossqecct="CrossQECCT", product=r"\textbf{ProductQECCT}")
    print("\n% LaTeX rows:  decoder & GPU eager & GPU CUDA-graph & CPU(1 thread) & GPU batch-512 amortized")
    for k, v in res["classical"].items():
        print(f"{names[k]:22s} & -- & -- & {v['mean_us']:.0f} & -- \\\\")
    for k, v in res["neural"].items():
        g = v.get("cuda", {}); c = v.get("cpu", {})
        cg = g.get("cuda_graph", {}).get("mean_us")
        am = g.get(f"amortized_us_b{min(args.batch, args.n)}")
        f = lambda x: f"{x:.0f}" if x is not None else "--"
        print(f"{names[k]:22s} & {f(g.get('mean_us'))} & {f(cg)} & {f(c.get('mean_us'))} & {f(am)} \\\\")


if __name__ == "__main__":
    main()
