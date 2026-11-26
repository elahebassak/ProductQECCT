"""Figures and LaTeX tables for the computational-cost section, generated from the result files only:
results/benchmark_inference.csv, results/benchmark_training.csv and results/all_results.csv.
Writes docs/figs/bench_*.pdf and docs/figs/bench_tables.tex.

  python docs/make_benchmark_figures.py [results_dir]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
COL, DCOL = 3.5, 7.16
C = {"product": "#2e9e46", "qecct": "#3a5fd9", "crossqecct": "#d9701a",
     "bposd_corr": "#a63aa6", "bposd": "#1aa0a0", "mwpm": "#a37d22", "bp": "#7a7a7a"}
LBL = {"product": "ProductQECCT (ours)", "qecct": "QECCT", "crossqecct": "CrossQECCT",
       "mwpm": "MWPM", "bposd": "BP+OSD (separate)", "bposd_corr": "BP+OSD (joint)", "bp": "BP"}
NEURAL = ("product", "qecct", "crossqecct")
STY = {m: dict(color=C[m], ls="-" if m in NEURAL else "--", marker="o" if m in NEURAL else "s",
               label=LBL[m], markeredgewidth=0.6, markerfacecolor="white" if m not in NEURAL else C[m]) for m in C}
ORDER = ["mwpm", "bp", "bposd", "bposd_corr", "qecct", "crossqecct", "product"]
NQ = {"toric-L4": 32, "toric-L6": 72, "toric-L8": 128, "toric-L10": 200, "hgp-400": 400, "hgp-asym": 500,
      "hgp-625": 625, "hgp-1600": 1600, "hgp-3600": 3600}
MIDP = {"toric": 0.102941, "hgp": 0.041765}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig); print("wrote", name)


def fam(code):
    return "toric" if code.startswith("toric") else "hgp"


bi = pd.read_csv(os.path.join(RES, "benchmark_inference.csv"))
bt = pd.read_csv(os.path.join(RES, "benchmark_training.csv"))
ar = pd.read_csv(os.path.join(RES, "all_results.csv"))
bi["mid"] = [abs(p - MIDP[fam(c)]) < 1e-4 for p, c in zip(bi.p, bi.code)]
mid = bi[bi.mid]


def lat(code, dec, meas="auto", stat="median_us"):
    m = {"auto": "latency_b1_gpu_e2e_graph" if dec in NEURAL else "latency_b1_cpu1core"}.get(meas, meas)
    r = mid[(mid.code == code) & (mid.decoder == dec) & (mid.measurement == m)]
    return float(r[stat].iloc[0]) if len(r) else np.nan


def thr(code, dec, meas):
    r = mid[(mid.code == code) & (mid.decoder == dec) & (mid.measurement == meas)]
    return float(r.codewords_per_s.iloc[0]) if len(r) else np.nan


def best_gpu_thr(code, dec):
    r = mid[(mid.code == code) & (mid.decoder == dec) & mid.measurement.str.startswith("throughput_gpu")]
    return float(r.codewords_per_s.max()) if len(r) else np.nan


def best_cpu_thr(code, dec):
    r = mid[(mid.code == code) & (mid.decoder == dec) & mid.measurement.isin(["throughput_cpu_workers4", "throughput_cpu_workers8"])]
    return float(r.codewords_per_s.max()) if len(r) else np.nan


# ------------------------------------------------------------------ Fig: batch-1 latency vs n (HGP codes), and throughput vs batch
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.1))
ax = axs[0]
HGP = {c: n for c, n in NQ.items() if c.startswith("hgp")}
for dec in ORDER:
    xs, ys = [], []
    for code, n in HGP.items():
        v = lat(code, dec)
        if np.isfinite(v):
            xs.append(n); ys.append(v / 1e3)
    if xs:
        ax.plot(xs, ys, **STY[dec])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("qubits $n$"); ax.set_ylabel("latency per codeword (ms), batch 1")
ax.set_xticks(list(HGP.values())); ax.set_xticklabels([str(v) for v in HGP.values()])
ax.minorticks_off(); ax.grid(True, which="major"); ax.set_title("(a) single-codeword latency, HGP codes, $p=0.042$")
ax = axs[1]
code = "hgp-625"
Bs = [1, 8, 32, 128, 512]
for dec in NEURAL:
    ys = [thr(code, dec, f"throughput_gpu_e2e_B{B}") for B in Bs]
    ax.plot(Bs, ys, **STY[dec])
cl = []
for dec, meas, ls, tag in [("bposd", "throughput_cpu_workers1", ":", "1 core"), ("bposd", "throughput_cpu_workers8", "--", "8 cores"),
                           ("bposd_corr", "throughput_cpu_workers1", ":", "1 core"), ("bposd_corr", "throughput_cpu_workers8", "--", "8 cores")]:
    v = thr(code, dec, meas)
    ax.axhline(v, color=C[dec], ls=ls, lw=1.0)
    cl.append(Line2D([], [], color=C[dec], ls=ls, lw=1.0, label=f"{LBL[dec]}, {tag}"))
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("GPU batch size $B$"); ax.set_ylabel("codewords / s")
ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs]); ax.minorticks_off(); ax.grid(True, which="major")
ax.set_title("(b) throughput on $[[625,25]]$, $p=0.042$")
h = [Line2D([], [], **STY[d]) for d in ORDER] + cl
fig.legend(h, [LBL[d] for d in ORDER] + [c.get_label() for c in cl], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.40), handlelength=2.2, columnspacing=1.2)
save(fig, "bench_latency_throughput")

# ------------------------------------------------------------------ Fig: accuracy vs latency
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=False)
for ax, metric, title in [(axs[0], "ler", r"$\mathrm{LER}$"), (axs[1], "ler_strict", r"$\mathrm{LER}_{\mathrm{strict}}$")]:
    for code, mk in [("toric-L8", "^"), ("hgp-625", "o")]:
        sel = ar[(ar.code == code) & (ar.noise == "depolarizing") & (abs(ar.p - MIDP[fam(code)]) < 1e-4)]
        for dec in ORDER:
            keys = [dec] if dec in NEURAL or dec in ("mwpm", "bp") else [k for k in sel.decoder.unique() if k.startswith(dec + "[") or k == dec]
            rows = sel[sel.decoder.isin(keys)]
            if not len(rows):
                continue
            acc = rows[metric].min()                      # best BP setting per code, as in the tables
            v = lat(code, dec)
            if np.isfinite(v):
                ax.plot(v / 1e3, acc, marker=mk, color=C[dec], ls="none", markersize=4.5 if mk == "o" else 5,
                        markerfacecolor=C[dec] if dec in NEURAL else "white", markeredgewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both"); ax.set_xlabel("latency (ms), batch 1"); ax.set_title(title)
h = [Line2D([], [], color=C[d], marker="o", ls="none", markerfacecolor=C[d] if d in NEURAL else "white", label=LBL[d]) for d in ORDER]
h += [Line2D([], [], color="k", marker="^", ls="none", markerfacecolor="none", label="toric $L{=}8$, $p{=}0.103$"),
      Line2D([], [], color="k", marker="o", ls="none", markerfacecolor="none", label="$[[625,25]]$, $p{=}0.042$")]
fig.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.5), handlelength=1.2, columnspacing=0.8)
save(fig, "bench_accuracy_latency")

# ------------------------------------------------------------------ Fig: toric LER vs L from the CSV (both noise models)
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.7))
for ax, noise, p0 in [(axs[0], "depolarizing", 0.102941), (axs[1], "independent", 0.054706)]:
    for dec in ORDER:
        xs, ys = [], []
        for L in (4, 6, 8, 10):
            sel = ar[(ar.code == f"toric-L{L}") & (ar.noise == noise) & (abs(ar.p - p0) < 1e-4)]
            keys = [k for k in sel.decoder.unique() if k == dec or k.startswith(dec + "[")]
            if keys:
                xs.append(L); ys.append(sel[sel.decoder.isin(keys)].ler.min())
        if xs:
            ax.plot(xs, ys, **STY[dec])
    ax.set_yscale("log"); ax.set_xticks([4, 6, 8, 10]); ax.set_xlabel("$L$"); ax.grid(True, which="both")
    ax.set_title(f"{noise}, $p={p0:.3f}$")
axs[0].set_ylabel(r"$\mathrm{LER}$")
h = [Line2D([], [], **STY[d]) for d in ORDER if d != "bp"]
fig.legend(h, [LBL[d] for d in ORDER if d != "bp"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.3), handlelength=2.2)
save(fig, "toric_L_from_csv")

# ------------------------------------------------------------------ LaTeX tables
def fmt(v, d=0):
    return "--" if not np.isfinite(v) else (f"{v:,.{d}f}".replace(",", "\\,"))


tex = []
# training-cost table, nine codes
tex.append("% training cost, one A100-40GB, 50 timed steps (tools/benchmark_training_step.py)")
tex.append(r"\begin{tabular}{lrrrrrrrrr}" + "\n" + r"\toprule" + "\n" + r" & \multicolumn{3}{c}{parameters} & \multicolumn{3}{c}{s / epoch} & \multicolumn{3}{c}{peak GiB}\\" + "\n"
           + r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" + "\n" + r"code & Ours & QECCT & Cross & Ours & QECCT & Cross & Ours & QECCT & Cross\\" + "\n" + r"\midrule")
for code in NQ:
    row = [code.replace("toric-L", "toric $L{=}$").replace("hgp-", "HGP ")]
    cells = {}
    for m in NEURAL:
        r = bt[(bt.model == m) & (bt.code == code)]
        ok = r[r.error.isna()] if "error" in r else r
        cells[m] = ok.iloc[0] if len(ok) else None
    for key, d in [("params", 0), ("s_per_epoch_1000", 0), ("peak_mem_gib", 1)]:
        for m in NEURAL:
            c = cells[m]
            if c is None:
                row.append("OOM")
            elif key == "params":
                v = c.params; row.append(f"{v/1e6:.2f}M" if v < 1e8 else (f"{v/1e6:.0f}M" if v < 1e9 else f"{v/1e9:.2f}B"))
            else:
                row.append(fmt(float(c[key]), d) + ("$^\\ast$" if key == "s_per_epoch_1000" and c.amp else ""))
    tex.append(" & ".join(row) + r"\\")
tex.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

# latency / throughput table for four codes
def best_gpu_thr_kind(code, dec, e2e):
    pre = "throughput_gpu_e2e_B" if e2e else "throughput_gpu_B"
    r = mid[(mid.code == code) & (mid.decoder == dec) & mid.measurement.str.startswith(pre)]
    return float(r.codewords_per_s.max()) if len(r) else np.nan


tex.append("% inference: batch-1 latency median [p95] in us; eager; throughput codewords/s: classical 1 core | 8 cores, neural GPU end-to-end | device-resident (best B<=512)")
codes4 = ["toric-L8", "hgp-625", "hgp-1600", "hgp-3600"]
for code in codes4:
    tex.append(f"% ---- {code}, p={MIDP[fam(code)]}")
    for dec in ORDER:
        med = lat(code, dec); p95 = lat(code, dec, stat="p95_us")
        if not np.isfinite(med):
            continue
        if dec in NEURAL:
            eager = lat(code, dec, "latency_b1_gpu_e2e_eager"); t1 = best_gpu_thr_kind(code, dec, True); t2 = best_gpu_thr_kind(code, dec, False)
        else:
            eager = np.nan; t1 = thr(code, dec, "throughput_cpu_workers1"); t2 = thr(code, dec, "throughput_cpu_workers8")
        tex.append(f"{LBL[dec]:22s} & {fmt(med)} [{fmt(p95)}] & {fmt(eager)} & {fmt(t1)} & {fmt(t2)}\\\\")
with open(os.path.join(OUT, "bench_tables.tex"), "w") as f:
    f.write("\n".join(tex))
print("wrote bench_tables.tex")
