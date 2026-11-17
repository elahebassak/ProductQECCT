"""Copy the small per-run files (everything except checkpoints) from runs/ into results/runs/ so that they can
be committed to the repository: args.json, model_info.json, history.json, test_epoch*.json, results_final.json,
results_partial_*.json, log.txt.

    python tools/export_run_metadata.py [--runs runs] [--out results/runs]
"""
import argparse, glob, os, shutil

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results/runs")
a = ap.parse_args()
n = 0
for d in sorted(glob.glob(f"{a.runs}/*/")):
    r = d.rstrip("/").split("/")[-1]
    os.makedirs(f"{a.out}/{r}", exist_ok=True)
    for f in glob.glob(d + "*"):
        if f.endswith(".pt"):
            continue
        shutil.copy2(f, f"{a.out}/{r}/"); n += 1
print(f"copied {n} files into {a.out}/")
