"""
Collect all result files into one CSV and (optionally) LER-vs-p plots.

    python collect_results.py --out results/summary.csv --plots results/plots
Reads runs/*/results_final.json (neural decoders) and results/*.json (classical / re-evaluations).
"""
import argparse
import csv
import glob
import json
import os


def load_all(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception as e:  # noqa
                print("skip", path, e)
                continue
            if "results" not in d:
                continue
            for p, r in d["results"].items():
                if p.startswith("_"):
                    continue
                rows.append(dict(model=d.get("model"), code=d.get("code"), noise=d.get("noise"), err_types=d.get("err_types"),
                                 n=d.get("n"), k=d.get("k"), params=d.get("params", ""), p=float(p),
                                 ler=r.get("ler"), ler_strict=r.get("ler_strict"), ber=r.get("ber"),
                                 ler_x=r.get("ler_x", ""), ler_z=r.get("ler_z", ""), samples=r.get("samples"),
                                 source=path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--plots", default=None, help="directory for LER-vs-p plots (needs matplotlib)")
    a = ap.parse_args()
    rows = load_all(["runs/*/results_final.json", "results/*.json"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")
    if a.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(a.plots, exist_ok=True)
        keys = sorted(set((r["code"], r["noise"], r["err_types"]) for r in rows))
        for code, noise, et in keys:
            fig, ax = plt.subplots(figsize=(5, 4))
            for model in sorted(set(r["model"] for r in rows if (r["code"], r["noise"], r["err_types"]) == (code, noise, et))):
                sel = sorted([r for r in rows if (r["code"], r["noise"], r["err_types"], r["model"]) == (code, noise, et, model)], key=lambda r: r["p"])
                ax.semilogy([r["p"] for r in sel], [max(r["ler_strict"], 1e-7) for r in sel], marker="o", ms=3, label=model)
            ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate (strict)")
            ax.set_title(f"{code}, {noise} ({et})"); ax.grid(True, which="both", alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(os.path.join(a.plots, f"ler_{code}_{noise}_{et}.png"), dpi=150); plt.close(fig)
        print("plots ->", a.plots)


if __name__ == "__main__":
    main()
