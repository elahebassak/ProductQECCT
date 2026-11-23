#!/usr/bin/env python
"""Training cost of every neural decoder on every code, measured on ONE GPU in one run.

For each (code, model) the script builds the model exactly as train.py does (same args, batch size, loss, Adam,
clipping), runs `--warmup` steps, then times `--steps` full training steps (sample -> forward -> loss -> backward
-> optimizer) with CUDA synchronization, and reports: parameters, median step time, the implied time per epoch
(1000 batches), training samples per second, and peak GPU memory. The global-attention baselines are attempted
on the large codes too; an out-of-memory or an instantiation failure is recorded as such, not invented.
Batch sizes follow the paper's protocol: 128, except 64 (hgp-1600) and 32 (hgp-3600) with bf16 autocast, as in
the actual runs; a second fp32 pass at batch 128 is run for ProductQECCT on the large codes when it fits.

  python tools/benchmark_training_step.py            # writes results/benchmark_training.json / .csv
"""
import os, sys, argparse, csv, json, time, platform, traceback
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
from pqecct.models import build_model, count_params
from pqecct.models.common import decoder_loss
import train as T

CODES = ["toric-L4", "toric-L6", "toric-L8", "toric-L10", "hgp-400", "hgp-asym", "hgp-625", "hgp-1600", "hgp-3600"]
PROTO = {"hgp-1600": dict(batch_size=64, amp=True), "hgp-3600": dict(batch_size=32, amp=True)}


def bench(model_name, code_name, batch_size, amp, args, device):
    targs = T.get_args(["--model", model_name, "--code", code_name, "--noise", "depolarizing", "--err_types", "xz",
                        "--batch_size", str(batch_size)] + (["--amp"] if amp else []))
    code = get_code(code_name)
    sampler = NoiseSampler(code, "depolarizing", "xz", device=device)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    model = build_model(targs, code).to(device).train()
    params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=targs.lr)
    p_train = T.default_p_grid(targs)

    def step():
        p = T.sample_p(targs, p_train, batch_size, device)
        e_x, e_z, s_x, s_z = sampler.sample_batch(p, batch_size)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits, g_logits = model(s_x, s_z)
        logits = logits.float(); g_logits = None if g_logits is None else g_logits.float()
        loss, _ = decoder_loss(logits, g_logits, e_x, e_z, sampler, "xz", targs.lam_ber, targs.lam_ler, targs.lam_g)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), targs.grad_clip)
        opt.step()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(args.steps):
        t = time.perf_counter(); step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    st = float(np.median(ts))
    out = dict(params=params, batch_size=batch_size, amp=amp, step_ms=st * 1e3, s_per_epoch_1000=st * 1000,
               samples_per_s=batch_size / st, peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30, steps=args.steps)
    del model, opt; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=",".join(CODES)); ap.add_argument("--models", default="product,qecct,crossqecct")
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results/benchmark_training")
    args = ap.parse_args()
    device = torch.device("cuda")
    T.set_seed(42)
    hw = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version())
    res = dict(hardware=hw, protocol=__doc__.split("  python")[0], runs={})
    rows = []
    for code_name in args.codes.split(","):
        for m in args.models.split(","):
            cfgs = [dict(batch_size=128, amp=False)] if code_name not in PROTO else [PROTO[code_name]] + ([dict(batch_size=128, amp=False)] if m == "product" else [])
            for cfg in cfgs:
                key = f"{m}|{code_name}|b{cfg['batch_size']}|{'bf16' if cfg['amp'] else 'fp32'}"
                print(f"[{key}]", flush=True)
                try:
                    r = bench(m, code_name, cfg["batch_size"], cfg["amp"], args, device)
                    print(f"   params {r['params']:,}  step {r['step_ms']:.1f} ms  -> {r['s_per_epoch_1000']:.0f} s/epoch  "
                          f"{r['samples_per_s']:.0f} samples/s  peak {r['peak_mem_gib']:.1f} GiB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    r = dict(error="OOM", batch_size=cfg["batch_size"], amp=cfg["amp"]); print("   OOM", flush=True); torch.cuda.empty_cache()
                except Exception as e:
                    r = dict(error=repr(e), batch_size=cfg["batch_size"], amp=cfg["amp"]); print("   failed:", repr(e)[:200], flush=True); traceback.print_exc()
                res["runs"][key] = r
                rows.append(dict(model=m, code=code_name, **{k: r.get(k, "") for k in ("params", "batch_size", "amp", "step_ms", "s_per_epoch_1000", "samples_per_s", "peak_mem_gib", "error")}))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out + ".json", "w"), indent=2)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}.json / .csv")


if __name__ == "__main__":
    main()
