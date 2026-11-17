"""
Print the comparison tables (DQEC-style LER / LER_strict) for all finished runs.

    python report.py                 # toric + HGP tables
    python report.py --family hgp    # only HGP codes
Re-collects results/summary.csv first (runs/*/results_final.json + results/*.json).
"""
import argparse
import csv
import subprocess
import sys

MODELS = ["mwpm", "bp", "bposd", "bposd_corr", "qecct", "crossqecct", "product"]
SETTINGS = {
    "toric": (["toric-L4", "toric-L6", "toric-L8", "toric-L10"],
              [("independent", "x", [0.03, 0.05, 0.1]), ("depolarizing", "xz", [0.08, 0.1, 0.15])]),
    "hgp": (["hgp-400", "hgp-asym", "hgp-625", "hgp-1600", "hgp-3600"],
            [("independent", "x", [0.02, 0.04, 0.06, 0.08]), ("depolarizing", "xz", [0.02, 0.04, 0.06, 0.08])]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["toric", "hgp", "all"], default="all")
    ap.add_argument("--no_collect", action="store_true")
    a = ap.parse_args()
    if not a.no_collect:
        subprocess.run([sys.executable, "collect_results.py", "--out", "results/summary.csv"], check=True)
    rows = [r for r in csv.DictReader(open("results/summary.csv")) if "runs_old" not in r["source"]]
    fams = ["toric", "hgp"] if a.family == "all" else [a.family]
    for fam in fams:
        codes, settings = SETTINGS[fam]
        for code in codes:
            for noise, et, ps in settings:
                block = [r for r in rows if r["code"] == code and r["noise"] == noise and r["err_types"] == et]
                if not block:
                    continue
                print(f"\n{code} {noise}   LER / LER_strict at p = " + ", ".join(f"{p:g}" for p in ps))
                present = sorted(set(r["model"] for r in block),
                                 key=lambda m: (next((i for i, b in enumerate(MODELS) if m == b or m.startswith(b + "[")), 99), m))
                for m in present:
                    sel = [r for r in block if r["model"] == m]
                    if not sel:
                        continue
                    out = []
                    for p in ps:
                        v = min(sel, key=lambda r: abs(float(r["p"]) - p))
                        ok = abs(float(v["p"]) - p) < 0.006
                        out.append(f"{float(v['ler']):.3e}/{float(v['ler_strict']):.2e}" if ok else "       --         ")
                    params = sel[0].get("params") or ""
                    print(f"  {m:24s} " + "  ".join(out) + (f"   params={params}" if params else ""))


if __name__ == "__main__":
    main()
