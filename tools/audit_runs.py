"""Audit every run directory: did training reach --epochs, which checkpoint was evaluated, with how many shots
and on how many error rates, and were the protocol arguments the standard ones.

    python tools/audit_runs.py [runs_dir]

Flags:  NOT-FINISHED  last.pt epoch != --epochs           EARLY-BEST   evaluated checkpoint before epoch 200
        PARTIAL       only a results_partial_*.json         GRID/SHOTS   final evaluation not 18 points / 5e5 shots
        BPE/HPARAM    non-standard batches per epoch / lr / depth / width
Exit code 1 if any flag is raised (so it can gate a release)."""
import glob, json, os, sys
import torch

runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
bad = 0
print(f"{'run':62s} {'last':>4} {'best':>4} {'eval':>4} {'warm':>4} {'bs':>3} {'bpe':>4} {'npts':>4} {'shots':>6}  flags")
for d in sorted(glob.glob(os.path.join(runs, "*/"))):
    r = d.rstrip("/").split("/")[-1]
    try:
        last = torch.load(d + "last.pt", map_location="cpu", weights_only=False)
        best = torch.load(d + "best.pt", map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{r:62s} MISSING CHECKPOINT: {e}"); bad += 1; continue
    a = last["args"]
    fin = d + "results_final.json"
    npts = shots = ev = "-"
    flags = []
    if os.path.exists(fin):
        R = json.load(open(fin))
        ps = [k for k in R["results"] if not k.startswith("_")]
        npts, shots = len(ps), R["results"][ps[0]].get("samples", "-")
        ev = R.get("evaluated_epoch", R.get("epoch", best["epoch"]))
    else:
        if glob.glob(d + "results_partial_*.json"): flags.append("PARTIAL")
        else: flags.append("NO-RESULT")
    if last["epoch"] != a["epochs"]: flags.append("NOT-FINISHED")
    if isinstance(ev, int) and ev < 200: flags.append("EARLY-BEST")
    if a["batches_per_epoch"] != 1000: flags.append("BPE")
    if npts not in ("-", 18): flags.append("GRID")
    if shots not in ("-", 500000): flags.append("SHOTS")
    if a["lr"] != 5e-4 or a["N_dec"] not in (6, 10) or a["d_model"] != 128: flags.append("HPARAM")
    bad += bool(flags)
    print(f"{r:62s} {last['epoch']:4d} {best['epoch']:4d} {str(ev):>4} {a.get('ler_warmup_epochs',0):4d} {a['batch_size']:3d} {a['batches_per_epoch']:4d} {str(npts):>4} {str(shots):>6}  {' '.join(flags)}")
print(f"\n{bad} run(s) flagged" if bad else "\nall runs clean")
sys.exit(1 if bad else 0)
