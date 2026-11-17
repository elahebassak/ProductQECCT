"""Publication figures for the ProductQECCT paper (IEEE single-column width = 3.5 in, double = 7.16 in).
All numbers are those recorded from the Nibi runs (Oct 2026). Writes PDFs to docs/figs/paper/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "paper")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.35, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
COL = 3.5      # IEEE column width (in)
DCOL = 7.16    # IEEE text width (in)

# Fixed categorical assignment (validated for CVD separation); classical decoders are also dashed.
C = {"product": "#2e9e46", "qecct": "#3a5fd9", "crossqecct": "#d9701a",
     "bposd_corr": "#a63aa6", "bposd": "#1aa0a0", "mwpm": "#a37d22", "bp": "#7a7a7a"}
LBL = {"product": "ProductQECCT (ours)", "qecct": "QECCT", "crossqecct": "CrossQECCT",
       "mwpm": "MWPM", "bposd": "BP+OSD (separate)", "bposd_corr": "BP+OSD (joint)", "bp": "BP"}
NEURAL = ("product", "qecct", "crossqecct")
STY = {m: dict(color=C[m], ls="-" if m in NEURAL else "--", marker="o" if m in NEURAL else "s",
               label=LBL[m], markeredgewidth=0.6, markerfacecolor="white" if m not in NEURAL else C[m])
       for m in C}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig: toric codes, LER vs L
Ls = [4, 6, 8, 10]
toric = {
    "depolarizing noise, $p=0.10$": {
        "mwpm": [2.552e-1, 1.811e-1, 1.342e-1, 9.794e-2],
        "bposd": [2.495e-1, 1.754e-1, 1.278e-1, 9.258e-2],
        "bposd_corr": [1.303e-1, 7.031e-2, 4.179e-2, 2.536e-2],
        "qecct": [1.375e-1, 9.162e-2, 8.162e-2, None],
        "crossqecct": [1.357e-1, 8.716e-2, 6.904e-2, 8.008e-2],
        "product": [1.260e-1, 7.067e-2, 4.934e-2, 4.629e-2]},
    "independent noise, $p=0.05$": {
        "mwpm": [9.315e-2, 4.971e-2, 2.775e-2, 1.559e-2],
        "bposd": [8.954e-2, 4.826e-2, 2.641e-2, 1.424e-2],
        "qecct": [8.424e-2, 4.600e-2, 3.387e-2, 3.385e-2],
        "crossqecct": [8.419e-2, 4.676e-2, 2.997e-2, 3.492e-2],
        "product": [8.379e-2, 4.255e-2, 2.461e-2, 1.867e-2]},
}
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5))
for ax, (title, d) in zip(axes, toric.items()):
    for m in ["mwpm", "bposd", "bposd_corr", "qecct", "crossqecct", "product"]:
        if m not in d:
            continue
        pts = [(l, y) for l, y in zip(Ls, d[m]) if y is not None]
        xs, ys = zip(*pts)
        ax.semilogy(xs, ys, **STY[m], zorder=3 if m == "product" else 2)
    ax.set_title(f"toric codes, {title}")
    ax.set_xlabel("lattice size $L$"); ax.set_xticks(Ls); ax.grid(True, which="major")
axes[0].set_ylabel("logical error rate")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.8)
fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.92))
save(fig, "toric_L")

# ------------------------------------------------------------------ Fig: hgp-400 LER vs p
ps = [0.02, 0.04, 0.06, 0.08]
h400 = {
    "depolarizing": {"bp": [3.766e-2, 1.902e-1, 3.787e-1, 6.608e-1], "bposd": [1.975e-3, 1.986e-2, 6.139e-2, 2.034e-1],
                     "bposd_corr": [6.550e-4, 8.385e-3, 2.755e-2, 1.031e-1], "product": [4.460e-4, 6.894e-3, 2.933e-2, 1.510e-1]},
    "independent": {"bp": [4.895e-2, 2.613e-1, 5.330e-1, 8.569e-1], "bposd": [3.745e-3, 4.226e-2, 1.670e-1, 5.562e-1],
                    "product": [4.698e-3, 6.349e-2, 2.678e-1, 7.266e-1]},
}
fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.3))
for ax, (noise, d) in zip(axes, h400.items()):
    for m in ["bp", "bposd", "bposd_corr", "product"]:
        if m in d:
            ax.semilogy(ps, d[m], **STY[m], zorder=3 if m == "product" else 2)
    ax.set_title(f"$[[400,16]]$ HGP code, {noise} noise")
    ax.set_xlabel("physical error rate $p$"); ax.set_xticks(ps); ax.grid(True, which="major")
    ax.legend(loc="lower right")
axes[0].set_ylabel("logical error rate")
fig.tight_layout(w_pad=1.5)
save(fig, "hgp400_p")

# ------------------------------------------------------------------ Fig: LER vs epoch on HGP codes (p = 0.04)
curves = {
    ("[[400,16]]", "depolarizing"): {
        "product": [(1, 1.66e-1), (20, 3.49e-2), (40, 2.29e-2), (60, 1.94e-2), (80, 1.78e-2), (100, 1.44e-2), (120, 1.28e-2), (140, 1.14e-2), (160, 9.90e-3), (180, 9.20e-3), (200, 8.75e-3), (220, 8.90e-3), (240, 8.70e-3), (260, 7.55e-3), (280, 8.60e-3), (300, 6.75e-3)],
        "qecct": [(1, 9.24e-1), (20, 1.12e-1), (40, 7.91e-2), (60, 6.66e-2)],
        "crossqecct": [(1, 8.89e-1), (20, 7.79e-2), (40, 4.93e-2), (60, 4.13e-2), (80, 3.68e-2), (100, 3.64e-2)]},
    ("[[400,16]]", "independent"): {
        "product": [(1, 4.32e-1), (20, 1.69e-1), (40, 1.40e-1), (60, 1.25e-1), (80, 1.13e-1), (100, 1.13e-1), (120, 1.00e-1), (140, 9.50e-2), (160, 8.95e-2), (180, 8.42e-2), (200, 8.70e-2), (220, 8.60e-2), (240, 7.99e-2), (260, 7.96e-2), (280, 7.98e-2), (300, 7.69e-2)],
        "qecct": [(1, 9.09e-1), (20, 3.52e-1), (40, 2.81e-1), (60, 2.53e-1), (80, 2.36e-1), (100, 2.20e-1), (120, 2.08e-1), (140, 1.99e-1), (160, 1.94e-1), (180, 1.86e-1), (200, 1.85e-1), (220, 1.78e-1)],
        "crossqecct": [(1, 8.93e-1), (20, 2.75e-1), (40, 2.19e-1), (60, 1.88e-1), (80, 1.75e-1), (100, 1.59e-1), (120, 1.57e-1), (140, 1.51e-1), (160, 1.43e-1), (180, 1.40e-1), (200, 1.42e-1), (220, 1.40e-1), (240, 1.31e-1), (260, 1.29e-1), (280, 1.29e-1)]},
    ("[[625,25]]", "depolarizing"): {
        "product": [(1, 1.89e-1), (20, 4.13e-2), (40, 3.35e-2), (60, 2.68e-2), (80, 2.18e-2), (100, 2.07e-2), (120, 1.95e-2), (140, 1.98e-2), (160, 1.67e-2), (180, 1.77e-2), (200, 1.59e-2), (220, 1.55e-2), (240, 1.62e-2)],
        "qecct": [(1, 9.88e-1), (20, 1.85e-1)],
        "crossqecct": [(1, 9.81e-1), (20, 1.28e-1), (40, 9.43e-2)]},
}
fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.2), sharex=True)
for ax, (key, d) in zip(axes, curves.items()):
    for m in ["qecct", "crossqecct", "product"]:
        xs, ys = zip(*d[m])
        ax.semilogy(xs, ys, **STY[m], markersize=2.5)
    ax.set_title(f"{key[0]} HGP, {key[1]}"); ax.set_xlabel("training epoch"); ax.set_xlim(0, 305)
    ax.grid(True, which="major")
axes[0].set_ylabel("logical error rate at $p=0.04$"); axes[0].legend(loc="upper right")
fig.tight_layout(w_pad=1.2)
save(fig, "hgp_epochs")

# ------------------------------------------------------------------ Fig: LER vs LER_strict (toric, depolarizing)
# (LER, LER_strict) at p = 0.10
strict = {
    "L6": {"qecct": (9.162e-2, 0.987), "crossqecct": (8.716e-2, 0.981), "product": (7.067e-2, 0.358)},
    "L8": {"qecct": (8.162e-2, 0.925), "crossqecct": (6.904e-2, 0.927), "product": (4.934e-2, 0.532)},
}
fig, axes = plt.subplots(1, 2, figsize=(COL, 2.05), sharey=False)
w = 0.26
for ax, (metric, idx) in zip(axes, [("LER", 0), ("LER$_\\mathrm{strict}$", 1)]):
    for gi, (code, d) in enumerate(strict.items()):
        for mi, m in enumerate(NEURAL):
            v = d[m][idx]
            ax.bar(gi + (mi - 1) * w, v, width=w - 0.03, color=C[m], label=LBL[m] if gi == 0 else None)
            ax.text(gi + (mi - 1) * w, v, f"{v:.2f}" if idx else f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(strict))); ax.set_xticklabels([f"toric {c}" for c in strict])
    ax.set_title(metric); ax.set_ylim(0, 1.12 if idx else 0.115)
    ax.grid(True, axis="y")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
fig.suptitle("toric codes, depolarizing noise, $p=0.10$", y=0.88, fontsize=8)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.84))
save(fig, "strict")

# ------------------------------------------------------------------ Fig: cost scaling (attention scores / layer / head, parameters)
codes = ["toric-L4", "toric-L6", "toric-L8", "toric-L10", "hgp-400", "hgp-asym", "hgp-625", "hgp-1600", "hgp-3600"]
n = np.array([32, 72, 128, 200, 400, 500, 625, 1600, 3600])
att_q = np.array([9216, 46656, 147456, 360000, 1.40e6, 2.19e6, 3.42e6, 22.4e6, 113.6e6])
att_p = np.array([512, 1728, 4096, 8000, 21504, 30240, 42000, 172032, 580608])
par_q = np.array([1.34e6, 1.92e6, 3.47e6, 6.71e6, 23.0e6, 35.3e6, 54.4e6, 349e6, 1.76e9])
par_p = np.array([1.21e6, 1.21e6, 1.21e6, 1.21e6, 1.21e6, 1.21e6, 1.22e6, 1.23e6, 1.26e6])
fig, axes = plt.subplots(1, 2, figsize=(COL, 2.05))
for ax, (yq, yp, ttl) in zip(axes, [(att_q, att_p, "attention scores / layer / head"), (par_q, par_p, "parameters")]):
    ax.loglog(n, yq, color=C["qecct"], marker="o", label="QECCT")
    ax.loglog(n, yp, color=C["product"], marker="o", label="ProductQECCT")
    ax.set_xlabel("qubits $n$"); ax.set_title(ttl); ax.grid(True, which="major")
    ax.set_xticks([32, 100, 400, 1600, 3600]); ax.set_xticklabels(["32", "100", "400", "1600", "3600"])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
for ax, (yq, yp) in zip(axes, [(att_q, att_p), (par_q, par_p)]):
    ax.annotate(f"{yq[-1] / yp[-1]:.0f}$\\times$", xy=(n[-1], np.sqrt(yq[-1] * yp[-1])), ha="right", va="center", fontsize=6.5, xytext=(-30, 0), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.9))
save(fig, "cost")

# ------------------------------------------------------------------ Fig: the HGP grid for the toric code (L = 3) with one X-check's support
L = 3
H = np.zeros((L, L), dtype=int)  # cyclic repetition code: H[a, i] = 1 iff i in {a, a+1}
for a in range(L):
    H[a, a] = 1; H[a, (a + 1) % L] = 1
R = Cc = 2 * L
fig, ax = plt.subplots(figsize=(COL * 0.7, COL * 0.7))
fill = {"Q1": "#eaf3ec", "SZ": "#eef1fb", "SX": "#fdf1e6", "Q2": "#eaf3ec"}
for r in range(R):
    for c in range(Cc):
        blk = ("Q1" if c < L else "SZ") if r < L else ("SX" if c < L else "Q2")
        ax.add_patch(Rectangle((c, R - 1 - r), 1, 1, facecolor=fill[blk], edgecolor="white", lw=1.2))
# highlight S_X(a, j) and its support
a, j = 1, 1
ax.add_patch(Rectangle((j, R - 1 - (L + a)), 1, 1, facecolor=C["crossqecct"], edgecolor="white", lw=1.2))
for i in range(L):
    if H[a, i]:
        ax.add_patch(Rectangle((j, R - 1 - i), 1, 1, facecolor=C["product"], edgecolor="white", lw=1.2))
for b in range(L):
    if H[b, j]:
        ax.add_patch(Rectangle((L + b, R - 1 - (L + a)), 1, 1, facecolor=C["product"], edgecolor="white", lw=1.2))
# block separators and labels
ax.plot([L, L], [0, R], color="black", lw=1.0); ax.plot([0, Cc], [L, L], color="black", lw=1.0)
ax.add_patch(Rectangle((0, 0), Cc, R, fill=False, edgecolor="black", lw=1.0))
for (x, y, t) in [(L / 2, R - L / 2, "$Q_1$"), (L + L / 2, R - L / 2, "$S_Z$"), (L / 2, L / 2, "$S_X$"), (L + L / 2, L / 2, "$Q_2$")]:
    ax.text(x, y, t, ha="center", va="center", fontsize=11, color="black", alpha=0.55)
ax.text(j + 0.5, R - 1 - (L + a) + 0.5, "$S_X(a,j)$", ha="center", va="center", fontsize=6, color="white")
ax.text(L / 2, R + 0.2, "data cols $[n_2]$", ha="center", va="bottom", fontsize=6.5)
ax.text(L + L / 2, R + 0.2, "check cols $[r_2]$", ha="center", va="bottom", fontsize=6.5)
ax.text(-0.2, R - L / 2, "data rows $[n_1]$", ha="right", va="center", rotation=90, fontsize=6.5)
ax.text(-0.2, L / 2, "check rows $[r_1]$", ha="right", va="center", rotation=90, fontsize=6.5)
# arrows: column coupling via H1, row coupling via H2
ax.annotate("", xy=(j + 0.5, R - 1 - a + 0.15), xytext=(j + 0.5, R - 1 - (L + a) + 0.85), arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
ax.text(j + 0.62, R - 1 - a - 0.9, "$H_1$", fontsize=7)
ax.annotate("", xy=(L + j + 0.85, R - 1 - (L + a) + 0.5), xytext=(j + 0.85, R - 1 - (L + a) + 0.5), arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8))
ax.text(L + 0.1, R - 1 - (L + a) + 0.62, "$H_2$", fontsize=7)
ax.set_xlim(-0.6, Cc + 0.1); ax.set_ylim(-0.1, R + 0.9); ax.set_aspect("equal"); ax.axis("off")
save(fig, "toric_grid")
print("done")
