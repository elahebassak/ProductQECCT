"""
Train a neural syndrome decoder (proposed ProductQECCT or the QECCT / CrossQECCT baselines)
from scratch on an HGP / toric code.

Example (toric L=6, depolarizing, proposed model):
    python train.py --model product --code toric-L6 --noise depolarizing --err_types xz \
        --N_dec 6 --d_model 128 --h 8 --epochs 200 --batches_per_epoch 1000 --batch_size 128

Training follows DQEC: Adam, lr 5e-4 with cosine decay, physical error rate drawn per sample
from the test grid (or uniformly in [p_min, p_max] with --p_sampling uniform), objective
    lam_ber * BCE(e_hat, e) + lam_ler * BCE(soft logical parity) + lam_g * BCE(g, e).
Checkpoints: <out>/last.pt (every epoch), <out>/best.pt (lowest training loss re-weighted with the final loss weights; equals DQEC's lowest-training-loss rule when --ler_warmup_epochs 0).
"""
import argparse
import json
import logging
import os
import sys
import time
import math
import random

import numpy as np
import torch

from pqecct.codes import get_code
from pqecct.noise import NoiseSampler
from pqecct.models import build_model, count_params
from pqecct.models.common import decoder_loss, decode_metrics, accumulate, finalize


def parse_p_list(spec):
    """'0.01:0.2:9' -> linspace ; '0.05,0.1,0.15' -> list"""
    if spec is None:
        return None
    if ":" in spec:
        a, b, k = spec.split(":")
        return [float(x) for x in np.linspace(float(a), float(b), int(k))]
    return [float(x) for x in spec.split(",")]


def get_args(argv=None):
    p = argparse.ArgumentParser(description="ProductQECCT training")
    # experiment
    p.add_argument("--model", type=str, default="product", choices=["product", "qecct", "crossqecct"])
    p.add_argument("--code", type=str, default="toric-L4", help="toric-L{L} | hgp-400 | hgp-625 | hgp-1600 | hgp-3600 | hgp-asym | hgp:<spec>:<spec>")
    p.add_argument("--noise", type=str, default="depolarizing", choices=["independent", "depolarizing"])
    p.add_argument("--err_types", type=str, default="xz", choices=["x", "z", "xz"],
                   help="which error types are generated/decoded. DQEC 'independent' results use 'x'.")
    p.add_argument("--out_root", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="0")
    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--N_dec", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--d_g", type=int, default=32, help="[product] width of factorized noise estimator")
    p.add_argument("--N_g", type=int, default=1, help="[product] layers of factorized noise estimator")
    p.add_argument("--no_share", action="store_true", help="[product] do not share parameters across processes A-D")
    p.add_argument("--no_out_mix", action="store_true", help="[product] disable factorized output mixing")
    p.add_argument("--g_hidden_mult", type=int, default=5, help="[qecct/crossqecct] g_omega hidden width multiplier")
    p.add_argument("--no_g", type=int, default=0, help="ablation: no initial noise estimator")
    p.add_argument("--no_mask", type=int, default=0, help="[qecct] ablation: no attention mask")
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batches_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.0, help=">0 switches to AdamW with this decay")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="split each batch into this many micro-batches (gradient accumulation; same update as the full batch, less memory)")
    p.add_argument("--warmup_steps", type=int, default=1000, help="linear LR warm-up (steps)")
    p.add_argument("--max_skipped", type=int, default=50, help="abort after this many non-finite losses")
    p.add_argument("--amp", action="store_true", help="bf16 autocast (A100/H100)")
    p.add_argument("--lam_ber", type=float, default=0.5)
    p.add_argument("--lam_ler", type=float, default=1.0)
    p.add_argument("--lam_g", type=float, default=0.5)
    p.add_argument("--ler_warmup_epochs", type=int, default=0, help="linearly ramp lam_ler from 0 over this many epochs")
    # noise rates
    p.add_argument("--p_test", type=str, default=None, help="e.g. '0.05:0.2:9' or '0.02,0.05'")
    p.add_argument("--p_train", type=str, default=None, help="default: same grid as p_test")
    p.add_argument("--p_sampling", type=str, default="discrete", choices=["discrete", "uniform"])
    # evaluation
    p.add_argument("--test_every", type=int, default=20)
    p.add_argument("--test_samples", type=int, default=20000, help="samples per p during training")
    p.add_argument("--final_samples", type=int, default=200000, help="samples per p for the final evaluation")
    p.add_argument("--final_p_test", type=str, default=None, help="finer grid for final eval (default: p_test with 2x points)")
    p.add_argument("--test_batch_size", type=int, default=512)
    # control
    p.add_argument("--resume", type=str, default="auto", help="'auto' resumes <out>/last.pt if present; a path; or 'none'")
    p.add_argument("--max_hours", type=float, default=0.0, help="stop training (then run final eval) after this wall time; 0 = no limit")
    p.add_argument("--final_on_timeout", type=int, default=1,
                   help="when stopped by --max_hours before the last epoch: 1 = evaluate anyway (written to results_partial_epoch{N}.json, never to results_final.json), 0 = exit and let the next chained job resume")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(argv)
    return args


def default_p_grid(args):
    if args.p_test is not None:
        return parse_p_list(args.p_test)
    if args.code.startswith("toric"):
        return list(np.linspace(0.01, 0.2, 9)) if args.noise == "independent" else list(np.linspace(0.05, 0.2, 9))
    return list(np.linspace(0.01, 0.10, 9))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sample_p(args, p_grid, batch_size, device):
    if args.p_sampling == "discrete":
        idx = torch.randint(0, len(p_grid), (batch_size,), device=device)
        return torch.tensor(p_grid, device=device)[idx]
    lo, hi = min(p_grid), max(p_grid)
    return lo + (hi - lo) * torch.rand(batch_size, device=device)


@torch.no_grad()
def evaluate(model, sampler, p_list, n_samples, batch_size, args, device, log_prefix="Test"):
    model.eval()
    results = {}
    t0 = time.time()
    for p in p_list:
        acc = {}
        done = 0
        while done < n_samples:
            bs = min(batch_size, n_samples - done)
            e_x, e_z, s_x, s_z = sampler.sample_batch(p, bs)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.amp and device.type == "cuda")):
                    logits, _ = model(s_x, s_z)
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                logging.warning(f"OOM in evaluation; test batch size -> {batch_size}")
                continue
            accumulate(acc, decode_metrics(logits.float(), e_x, e_z, s_x, s_z, sampler, args.err_types))
            done += bs
        res = finalize(acc)
        results[f"{p:.6f}"] = res
        logging.info(f"{log_prefix} p={p:.4f}: LER={res['ler']:.3e} LER_strict={res['ler_strict']:.3e} BER={res['ber']:.3e} (n={res['samples']})")
    logging.info(f"{log_prefix} time {time.time() - t0:.1f}s")
    model.train()
    return results


def main():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"{args.model}_{args.code}_{args.noise}_{args.err_types}_N{args.N_dec}_d{args.d_model}_s{args.seed}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out_dir, "log.txt")), logging.StreamHandler(sys.stdout)])
    logging.info("args: " + json.dumps(vars(args)))
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    code = get_code(args.code)
    logging.info(code.summary(distance_trials=100 if code.n <= 1000 else 0))
    if args.model == "product" and not code.is_hgp():
        raise SystemExit("ProductQECCT needs an HGP-structured code")
    sampler = NoiseSampler(code, args.noise, args.err_types, device=device)

    model = build_model(args, code).to(device)
    n_params = count_params(model)
    logging.info(f"model={args.model} params={n_params:,} device={device}")
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump(dict(params=n_params, n=code.n, k=code.k, rx=code.rx, rz=code.rz), f, indent=2)

    if args.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scale = 1.0   # halved automatically after a non-finite loss

    def lr_at(epoch, step):
        """cosine over epochs (DQEC) x linear warm-up over the first steps."""
        frac = (epoch - 1) / max(args.epochs, 1)
        cos = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * frac))
        warm = min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps > 0 else 1.0
        return cos * warm * lr_scale
    p_test = default_p_grid(args)
    p_train = parse_p_list(args.p_train) or p_test
    logging.info(f"p_test={['%.4f' % p for p in p_test]}  p_train={['%.4f' % p for p in p_train]} ({args.p_sampling})")

    start_epoch = 1
    best_loss = float("inf")
    ckpt_path = os.path.join(out_dir, "last.pt")
    resume = None
    if args.resume == "auto" and os.path.exists(ckpt_path):
        resume = ckpt_path
    elif args.resume not in ("auto", "none"):
        resume = args.resume
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1; best_loss = ck.get("best_loss", best_loss); lr_scale = ck.get("lr_scale", 1.0)
        logging.info(f"resumed from {resume} at epoch {start_epoch}")
        if start_epoch > args.epochs and os.path.exists(os.path.join(out_dir, "results_final.json")):
            logging.info("training and final evaluation already complete; nothing to do")
            return

    t_start = time.time()
    history = []
    global_step = (start_epoch - 1) * args.batches_per_epoch
    n_skipped = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t_ep = time.time()
        acc = {}
        cum = dict(loss=0.0, ber=0.0, ler=0.0, g=0.0)
        nb_ok = 0
        for it in range(args.batches_per_epoch):
            for grp in opt.param_groups:
                grp["lr"] = lr_at(epoch, global_step)
            global_step += 1
            p = sample_p(args, p_train, args.batch_size, device)
            e_x, e_z, s_x, s_z = sampler.sample_batch(p, args.batch_size)
            lam_ler = args.lam_ler * min(1.0, epoch / args.ler_warmup_epochs) if args.ler_warmup_epochs > 0 else args.lam_ler
            opt.zero_grad(set_to_none=True)
            # gradient accumulation over equal micro-batches: the mean losses of the micro-batches average
            # to the full-batch mean losses, so the update is identical to a single batch of args.batch_size
            A = max(1, args.accum_steps)
            mb = args.batch_size // A
            loss_tot = 0.0; parts_tot = dict(ber=0.0, ler=0.0, g=0.0); logits_all = []
            bad = False
            for a in range(A):
                sl = slice(a * mb, (a + 1) * mb) if A > 1 else slice(None)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.amp and device.type == "cuda")):
                    logits, g_logits = model(s_x[sl], s_z[sl])
                logits = logits.float(); g_logits = None if g_logits is None else g_logits.float()
                loss, parts = decoder_loss(logits, g_logits, e_x[sl], e_z[sl], sampler, args.err_types,
                                           args.lam_ber, lam_ler, args.lam_g)
                if not torch.isfinite(loss):
                    bad = True
                    break
                (loss / A).backward()
                loss_tot += loss.item() / A
                for k in parts_tot:
                    parts_tot[k] += parts[k] / A
                logits_all.append(logits.detach())
            if bad:
                # guard: skip the step, halve the learning rate, and abort if it keeps happening
                n_skipped += 1
                lr_scale *= 0.5
                logging.warning(f"non-finite loss at epoch {epoch} step {it} (skipped {n_skipped}); lr_scale -> {lr_scale:.3g}")
                opt.zero_grad(set_to_none=True)
                if n_skipped >= args.max_skipped:
                    raise RuntimeError("too many non-finite losses; aborting")
                continue
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(gn):
                    n_skipped += 1
                    lr_scale *= 0.5
                    logging.warning(f"non-finite gradient at epoch {epoch} step {it} (skipped {n_skipped}); lr_scale -> {lr_scale:.3g}")
                    opt.zero_grad(set_to_none=True)
                    continue
            logits = torch.cat(logits_all, 0); loss_val = loss_tot; parts = parts_tot
            opt.step()
            nb_ok += 1
            cum["loss"] += loss_val; cum["ber"] += parts["ber"]; cum["ler"] += parts["ler"]; cum["g"] += parts["g"]
            accumulate(acc, decode_metrics(logits, e_x, e_z, s_x, s_z, sampler, args.err_types))
        nb = max(nb_ok, 1)
        met = finalize(acc)
        ep_loss = cum["loss"] / nb
        # checkpoint-selection metric: the epoch loss re-weighted with the *final* loss weights, so that it is
        # comparable across epochs when --ler_warmup_epochs > 0 (the raw loss is lower while lam_ler is ramping up)
        sel_loss = (args.lam_ber * cum["ber"] + args.lam_ler * cum["ler"] + args.lam_g * cum["g"]) / nb
        logging.info(f"Epoch {epoch}/{args.epochs} lr={opt.param_groups[0]['lr']:.2e} loss={ep_loss:.4e} "
                     f"(ber {cum['ber']/nb:.3e} ler {cum['ler']/nb:.3e} g {cum['g']/nb:.3e}) "
                     f"train LER={met['ler']:.3e} LER_strict={met['ler_strict']:.3e} BER={met['ber']:.3e} "
                     f"time={time.time()-t_ep:.1f}s"
                     + (f" peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB" if device.type == "cuda" else ""))
        history.append(dict(epoch=epoch, loss=ep_loss, ler=met["ler"], ler_strict=met["ler_strict"], ber=met["ber"]))
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), lr_scale=lr_scale,
                     epoch=epoch, best_loss=best_loss, args=vars(args), code=args.code, skipped=n_skipped)
        torch.save(state, ckpt_path)
        if sel_loss < best_loss:
            best_loss = sel_loss
            state["best_loss"] = best_loss
            torch.save(state, os.path.join(out_dir, "best.pt"))
            logging.info(f"best model saved (selection loss {sel_loss:.4e})")
        if epoch % args.test_every == 0 or epoch in (1, args.epochs):
            res = evaluate(model, sampler, p_test, args.test_samples, args.test_batch_size, args, device, f"Test[ep{epoch}]")
            with open(os.path.join(out_dir, f"test_epoch{epoch}.json"), "w") as f:
                json.dump(res, f, indent=2)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        if args.max_hours > 0 and (time.time() - t_start) / 3600.0 > args.max_hours:
            logging.info(f"max_hours reached after epoch {epoch}; stopping training")
            if epoch < args.epochs and not args.final_on_timeout:
                logging.info("final evaluation deferred to the next chained job (resumes from last.pt)")
                return
            break
    last_epoch = epoch if start_epoch <= args.epochs else args.epochs
    partial = last_epoch < args.epochs

    # final evaluation with the best (lowest training loss) checkpoint, as in DQEC
    best_path = os.path.join(out_dir, "best.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        logging.info(f"loaded best checkpoint (epoch {ck['epoch']}, loss {ck['best_loss']:.4e})")
    p_final = parse_p_list(args.final_p_test) if args.final_p_test else list(np.linspace(min(p_test), max(p_test), 2 * len(p_test)))
    res = evaluate(model, sampler, p_final, args.final_samples, args.test_batch_size, args, device, "Final")
    out = dict(model=args.model, code=args.code, noise=args.noise, err_types=args.err_types, params=n_params,
               n=code.n, k=code.k, results=res, args=vars(args))
    out["evaluated_epoch"] = int(ck["epoch"]) if os.path.exists(best_path) else last_epoch
    out["training_epochs_completed"] = last_epoch
    if partial:
        # never present an unfinished run as a final result: a chained job will resume from last.pt and
        # write results_final.json when training reaches --epochs
        fn = f"results_partial_epoch{last_epoch}.json"
        logging.warning(f"training stopped at epoch {last_epoch}/{args.epochs}: writing {fn}, NOT results_final.json")
    else:
        fn = "results_final.json"
    with open(os.path.join(out_dir, fn), "w") as f:
        json.dump(out, f, indent=2)
    logging.info("done")


if __name__ == "__main__":
    main()
