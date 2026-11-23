#!/usr/bin/env python
"""Unified inference benchmark: latency, throughput and memory of every decoder on every code, one protocol.

Measurements (per code, per physical error rate p):
  A. single-codeword latency (batch size 1), N timed codewords after warm-up; median, 95th percentile and mean.
     * classical decoders (MWPM / BP / BP+OSD separate / BP+OSD joint): one CPU core, syndrome -> binary correction
       (both CSS decodes for the separate decoders, the single 3n-column decode for the joint one);
     * neural decoders: END-TO-END, syndrome in CPU memory -> binary correction in CPU memory (host->device copy,
       forward pass, threshold, device->host copy), in eager PyTorch and with the same forward captured in a CUDA
       graph (corrections verified identical on the first 50 codewords); plus the device-only time of the
       graph-captured forward (input already on the GPU), labelled separately.
  B. throughput: neural decoders at batch sizes B in --batches on the GPU (codewords/s and T(B)/B, inputs on the
     device, 5 repetitions, median); classical decoders with W worker processes (--workers), codewords/s.
  C. peak memory: neural, torch.cuda.max_memory_allocated at B=1 and at the largest feasible B (OOM -> recorded);
     classical, max resident set size of the decoding process.
Everything is written to results/benchmark/<code>.json and appended as flat rows to results/benchmark_inference.csv.
Decoders are only benchmarked where the paper evaluated them (MWPM: toric; BP: HGP; joint BP+OSD and the
global-attention neural baselines: toric and the mid-size HGP codes).

Example (one GPU node, 8 CPU cores):
  python tools/benchmark_inference.py --codes toric-L4,toric-L6,toric-L8,toric-L10,hgp-400,hgp-asym,hgp-625,hgp-1600,hgp-3600
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, csv, json, platform, resource, sys, time, traceback
from multiprocessing import get_context
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
import eval_classical as ec

# nominal table points -> nearest grid points (see the paper's protocol)
P_POINTS = {"toric": [0.076471, 0.102941, 0.147059], "hgp": [0.020588, 0.041765, 0.078824]}
BEST_SEP_SCHEDULE = {"toric-L4": "serial", "toric-L6": "serial", "toric-L8": "serial", "toric-L10": "serial",
                     "hgp-400": "parallel", "hgp-asym": "parallel", "hgp-625": "parallel",
                     "hgp-1600": "parallel", "hgp-3600": "parallel"}
LARGE = ("hgp-1600", "hgp-3600")


def pct(ts):
    ts = np.asarray(ts) * 1e6
    return dict(median_us=float(np.median(ts)), p95_us=float(np.percentile(ts, 95)), mean_us=float(ts.mean()),
                p05_us=float(np.percentile(ts, 5)), n=int(len(ts)))


def time_loop(decode, N, warmup):
    for i in range(min(warmup, N)):
        decode(i)
    ts = []
    for i in range(N):
        t = time.perf_counter(); decode(i); ts.append(time.perf_counter() - t)
    return pct(ts)


# ----------------------------------------------------------------------------------------------- classical
def make_classical(kind, code, p, args):
    n = code.n
    p_bit = 2.0 * p / 3.0
    if kind == "bposd_corr":
        Z_x = np.zeros((code.rx, n), dtype=np.uint8); Z_z = np.zeros((code.rz, n), dtype=np.uint8)
        H3 = np.block([[Z_x, code.Hx, code.Hx], [code.Hz, Z_z, code.Hz]]).astype(np.uint8)
        BpOsdDecoder, _ = ec._import_ldpc()
        dec = BpOsdDecoder(H3, error_channel=[p / 3.0] * (3 * n), max_iter=args.max_iter, bp_method=args.bp_method,
                           ms_scaling_factor=args.ms_scaling, schedule="serial",
                           osd_method=args.osd_method, osd_order=args.osd_order)

        def decode(s_x, s_z):
            e3 = np.asarray(dec.decode(np.concatenate([s_x, s_z]).astype(np.uint8)), dtype=np.uint8)
            return e3[:n] ^ e3[2 * n:], e3[n:2 * n] ^ e3[2 * n:]
        return decode
    a = argparse.Namespace(**vars(args)); a.schedule = BEST_SEP_SCHEDULE.get(args.code, "parallel")
    dec_x = ec._make_decoder(kind, code.Hz, p_bit, a)
    dec_z = ec._make_decoder(kind, code.Hx, p_bit, a)

    def decode(s_x, s_z):
        return (np.asarray(dec_x.decode(s_z.astype(np.uint8)), dtype=np.uint8)[:n],
                np.asarray(dec_z.decode(s_x.astype(np.uint8)), dtype=np.uint8)[:n])
    return decode


_W = {}


def _worker_init(kind, code_name, p, args_dict):
    _W["decode"] = make_classical(kind, get_code(code_name), p, argparse.Namespace(**args_dict))


def _worker_chunk(task):
    s_x, s_z = task
    d = _W["decode"]
    for i in range(s_x.shape[0]):
        d(s_x[i], s_z[i])
    return s_x.shape[0]


def bench_classical(kind, code, p, s_x, s_z, args):
    out = {}
    decode = make_classical(kind, code, p, args)
    N = s_x.shape[0]
    out["batch1"] = time_loop(lambda i: decode(s_x[i], s_z[i]), N, args.warmup)
    out["batch1"]["hardware"] = "cpu, 1 core"
    # throughput with W worker processes
    out["throughput"] = {}
    for W in args.workers:
        M = args.throughput_n
        rng = np.random.default_rng(args.seed + 1)
        sampler = NoiseSampler(code, "depolarizing", "xz", device="cpu")
        _, _, tx, tz = sampler.sample_batch_numpy(p, M, rng)
        tx = np.asarray(tx, dtype=np.uint8); tz = np.asarray(tz, dtype=np.uint8)
        chunks = [(tx[i::W], tz[i::W]) for i in range(W)]
        ctx = get_context("spawn")
        with ctx.Pool(W, initializer=_worker_init, initargs=(kind, args.code, p, vars(args))) as pool:
            pool.map(_worker_chunk, [(c[0][:8], c[1][:8]) for c in chunks])      # warm-up
            t = time.perf_counter(); done = sum(pool.map(_worker_chunk, chunks)); T = time.perf_counter() - t
        out["throughput"][f"workers{W}"] = dict(codewords_per_s=done / T, amortized_us=T / done * 1e6, n=done)
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


# ----------------------------------------------------------------------------------------------- neural
def load_model(model_name, code, args, device):
    import torch
    from pqecct.models import build_model, count_params
    ck_name = "last.pt" if args.code in LARGE else "best.pt"
    run_dir = os.path.join(args.runs_root, f"{model_name}_{args.code}_depolarizing_xz_N6_d128_s42")
    path = os.path.join(run_dir, ck_name)
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    margs = argparse.Namespace(**ck["args"])
    model = build_model(margs, code).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, dict(params=count_params(model), epoch=ck["epoch"], ckpt=path)


def bench_neural(model_name, code, p, s_x, s_z, args):
    import torch
    device = torch.device("cuda")
    model, meta = load_model(model_name, code, args, device)
    if model is None:
        return None
    out = dict(meta)
    N = s_x.shape[0]
    sx_np = np.ascontiguousarray(s_x, dtype=np.float32); sz_np = np.ascontiguousarray(s_z, dtype=np.float32)
    sx_dev = torch.as_tensor(sx_np, device=device); sz_dev = torch.as_tensor(sz_np, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # --- A1 end-to-end, eager: CPU syndrome -> GPU -> forward -> threshold -> CPU correction
        def e2e_eager(i):
            x = torch.from_numpy(sx_np[i:i + 1]).to(device, non_blocking=False)
            z = torch.from_numpy(sz_np[i:i + 1]).to(device, non_blocking=False)
            logits, _ = model(x, z)
            return (logits > 0).to("cpu", dtype=torch.uint8).numpy()
        out["e2e_eager"] = time_loop(e2e_eager, N, args.warmup)
        out["peak_mem_b1_mib"] = torch.cuda.max_memory_allocated() / 2**20
        ref = [e2e_eager(i) for i in range(min(50, N))]
        # --- A2 end-to-end with CUDA graph
        try:
            st_x = sx_dev[:1].clone(); st_z = sz_dev[:1].clone()
            pin_x = torch.empty_like(st_x, device="cpu").pin_memory(); pin_z = torch.empty_like(st_z, device="cpu").pin_memory()
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    model(st_x, st_z)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                g_logits, _ = model(st_x, st_z)
                g_corr = (g_logits > 0).to(torch.uint8)
            host_out = torch.empty_like(g_corr, device="cpu").pin_memory()

            def e2e_graph(i):
                pin_x.copy_(torch.from_numpy(sx_np[i:i + 1])); pin_z.copy_(torch.from_numpy(sz_np[i:i + 1]))
                st_x.copy_(pin_x, non_blocking=True); st_z.copy_(pin_z, non_blocking=True)
                g.replay()
                host_out.copy_(g_corr, non_blocking=True); torch.cuda.synchronize()
                return host_out.numpy()
            mism = sum(int((e2e_graph(i) != ref[i]).any()) for i in range(len(ref)))
            out["e2e_graph"] = time_loop(e2e_graph, N, args.warmup); out["e2e_graph"]["mismatch_vs_eager"] = mism

            def dev_graph(i):
                st_x.copy_(sx_dev[i:i + 1]); st_z.copy_(sz_dev[i:i + 1]); g.replay(); torch.cuda.synchronize()
            out["device_graph"] = time_loop(dev_graph, N, args.warmup)
        except Exception as e:
            out["e2e_graph"] = dict(error=repr(e)); traceback.print_exc()
        # --- B throughput vs batch size (inputs on the device) and C peak memory
        out["throughput"] = {}
        for B in args.batches:
            if B > N:
                break
            try:
                torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    logits, _ = model(sx_dev[:B], sz_dev[:B]); c = logits > 0
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    t = time.perf_counter(); logits, _ = model(sx_dev[:B], sz_dev[:B]); c = logits > 0
                    torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
                T = float(np.median(ts))
                out["throughput"][f"B{B}"] = dict(codewords_per_s=B / T, amortized_us=T / B * 1e6, batch_time_ms=T * 1e3,
                                                  peak_mem_mib=torch.cuda.max_memory_allocated() / 2**20)
            except torch.cuda.OutOfMemoryError:
                out["throughput"][f"B{B}"] = dict(error="OOM"); torch.cuda.empty_cache(); break
    del model; torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------------------------- main
def flat_rows(code, n, p, name, kind, r):
    rows = []
    base = dict(code=code, n=n, p=p, decoder=name, kind=kind)
    if kind == "classical":
        b = r["batch1"]
        rows.append(dict(base, measurement="latency_b1_cpu1core", median_us=b["median_us"], p95_us=b["p95_us"], mean_us=b["mean_us"]))
        for k, v in r["throughput"].items():
            rows.append(dict(base, measurement=f"throughput_cpu_{k}", codewords_per_s=v["codewords_per_s"], amortized_us=v["amortized_us"]))
        rows.append(dict(base, measurement="peak_rss_mib", value=r["peak_rss_mib"]))
    else:
        for key in ("e2e_eager", "e2e_graph", "device_graph"):
            if key in r and "median_us" in r[key]:
                rows.append(dict(base, measurement=f"latency_b1_gpu_{key}", median_us=r[key]["median_us"], p95_us=r[key]["p95_us"], mean_us=r[key]["mean_us"]))
        for k, v in r.get("throughput", {}).items():
            if "codewords_per_s" in v:
                rows.append(dict(base, measurement=f"throughput_gpu_{k}", codewords_per_s=v["codewords_per_s"], amortized_us=v["amortized_us"], value=v["peak_mem_mib"]))
        rows.append(dict(base, measurement="params", value=r["params"]))
        rows.append(dict(base, measurement="peak_mem_b1_mib", value=r["peak_mem_b1_mib"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="toric-L4,toric-L6,toric-L8,toric-L10,hgp-400,hgp-asym,hgp-625,hgp-1600,hgp-3600")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--large_n", type=int, default=200, help="timed codewords for hgp-1600/3600 (BP+OSD is slow there)")
    ap.add_argument("--throughput_n", type=int, default=2000, help="codewords for the classical throughput runs")
    ap.add_argument("--batches", default="1,8,32,128,512"); ap.add_argument("--workers", default="1,4,8")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--max_iter", type=int, default=100); ap.add_argument("--bp_method", default="minimum_sum")
    ap.add_argument("--ms_scaling", type=float, default=0.625); ap.add_argument("--osd_method", default="OSD_CS")
    ap.add_argument("--osd_order", type=int, default=10)
    ap.add_argument("--out_dir", default="results/benchmark"); ap.add_argument("--csv", default="results/benchmark_inference.csv")
    ap.add_argument("--only", default="", help="comma list to restrict decoders, e.g. product,bposd")
    args = ap.parse_args()
    args.batches = [int(b) for b in args.batches.split(",")]; args.workers = [int(w) for w in args.workers.split(",")]
    import torch
    hw = dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip(),
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, torch=torch.__version__,
              python=platform.python_version(), omp_threads=os.environ.get("OMP_NUM_THREADS"), precision="fp32",
              cpus_visible=len(os.sched_getaffinity(0)))
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    for code_name in args.codes.split(","):
        args.code = code_name
        code = get_code(code_name)
        fam = "toric" if code_name.startswith("toric") else "hgp"
        classical = (["mwpm", "bposd", "bposd_corr"] if fam == "toric" else
                     ["bp", "bposd"] if code_name in LARGE else ["bp", "bposd", "bposd_corr"])
        neural = ["product"] if code_name in LARGE else ["product", "qecct", "crossqecct"]
        if args.only:
            keep = args.only.split(","); classical = [c for c in classical if c in keep]; neural = [m for m in neural if m in keep]
        res = dict(code=code_name, n=code.n, k=code.k if hasattr(code, "k") else None, hardware=hw, protocol=__doc__.split("Example")[0],
                   bp_settings=dict(max_iter=args.max_iter, bp_method=args.bp_method, ms_scaling=args.ms_scaling,
                                    schedule_separate=BEST_SEP_SCHEDULE.get(code_name), schedule_joint="serial",
                                    osd_method=args.osd_method, osd_order=args.osd_order), per_p={})
        N = args.large_n if code_name in LARGE else args.n
        args.throughput_n = min(args.throughput_n, 2 * N)
        for p in P_POINTS[fam]:
            rng = np.random.default_rng(args.seed)
            sampler = NoiseSampler(code, "depolarizing", "xz", device="cpu")
            _, _, s_x, s_z = sampler.sample_batch_numpy(p, N, rng)
            s_x = np.asarray(s_x, dtype=np.uint8); s_z = np.asarray(s_z, dtype=np.uint8)
            R = res["per_p"][str(p)] = dict(classical={}, neural={})
            for kind in classical:
                print(f"[{code_name} p={p}] classical {kind}", flush=True)
                try:
                    R["classical"][kind] = r = bench_classical(kind, code, p, s_x, s_z, args)
                    print(f"   median {r['batch1']['median_us']:.1f} us  p95 {r['batch1']['p95_us']:.1f}  "
                          + "  ".join(f"{k}:{v['codewords_per_s']:.0f}/s" for k, v in r["throughput"].items()), flush=True)
                    all_rows += flat_rows(code_name, code.n, p, kind, "classical", r)
                except Exception as e:
                    R["classical"][kind] = dict(error=repr(e)); traceback.print_exc()
            if torch.cuda.is_available():
                for m in neural:
                    print(f"[{code_name} p={p}] neural {m}", flush=True)
                    try:
                        r = bench_neural(m, code, p, s_x, s_z, args)
                        if r is None:
                            print("   no checkpoint, skipped"); continue
                        R["neural"][m] = r
                        g = r.get("e2e_graph", {})
                        print(f"   e2e eager median {r['e2e_eager']['median_us']:.1f} us; e2e graph {g.get('median_us', float('nan')):.1f} us "
                              f"(mismatch {g.get('mismatch_vs_eager', '?')}); device graph {r.get('device_graph', {}).get('median_us', float('nan')):.1f} us; "
                              + "  ".join(f"{k}:{v.get('codewords_per_s', 0):.0f}/s" for k, v in r["throughput"].items()), flush=True)
                        all_rows += flat_rows(code_name, code.n, p, m, "neural", r)
                    except Exception as e:
                        R["neural"][m] = dict(error=repr(e)); traceback.print_exc()
        with open(os.path.join(args.out_dir, f"{code_name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.out_dir}/{code_name}.json", flush=True)
    cols = ["code", "n", "p", "decoder", "kind", "measurement", "median_us", "p95_us", "mean_us", "codewords_per_s", "amortized_us", "value"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {args.csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
