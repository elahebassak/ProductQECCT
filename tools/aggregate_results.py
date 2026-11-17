"""Merge every final result (neural runs/*/results_final.json and classical results/*.json) into
results/all_results.json and results/all_results.csv, one record per (decoder, code, noise, p).

    python tools/aggregate_results.py [--runs runs] [--results results]
"""
import argparse, csv, glob, json, os
from collections import Counter

ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--results", default="results")
a = ap.parse_args()
files = sorted(glob.glob(f"{a.runs}/*/results_final.json")) + \
        sorted(f for f in glob.glob(f"{a.results}/*.json") if not os.path.basename(f).startswith("all_results"))
entries = []
for f in files:
    d = json.load(open(f))
    if "results" not in d or "model" not in d:
        continue
    run = f.split("/")[-2] if f.startswith(a.runs) else os.path.basename(f)
    tag = run.split("_abl-")[1] if "_abl-" in run else ""
    entries.append(dict(source=f, run=run, decoder=d["model"] + (f"[abl-{tag}]" if tag else ""),
                        code=d["code"], noise=d["noise"], err_types=d.get("err_types"), params=d.get("params"),
                        evaluated_epoch=d.get("evaluated_epoch", d.get("epoch")),
                        results={p: {k: v for k, v in r.items() if k in ("ler", "ler_strict", "ber", "ler_x", "ler_z", "samples")}
                                 for p, r in d["results"].items() if not p.startswith("_")}))
os.makedirs(a.results, exist_ok=True)
json.dump(dict(description="ProductQECCT: all final results", n_entries=len(entries), entries=entries),
          open(f"{a.results}/all_results.json", "w"), indent=1)
with open(f"{a.results}/all_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["decoder", "code", "noise", "err_types", "params", "p", "ler", "ler_strict", "ber", "samples", "source"])
    for e in entries:
        for p, r in sorted(e["results"].items(), key=lambda kv: float(kv[0])):
            w.writerow([e["decoder"], e["code"], e["noise"], e["err_types"], e["params"], float(p),
                        r.get("ler"), r.get("ler_strict"), r.get("ber"), r.get("samples"), e["source"]])
print(len(entries), "entries:", dict(Counter(e["decoder"].split("[")[0] for e in entries)))
print(f"wrote {a.results}/all_results.json and {a.results}/all_results.csv")
