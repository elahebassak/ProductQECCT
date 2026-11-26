"""Fig. 1: the HGP grid, the four cross-attention processes A-D and the shared per-column mask.
Consistent with the process equations of the paper:
  A: columns, data rows <- check rows   (Q1 <- SX in data columns, SZ <- Q2 in check columns), mask g(H1^T)
  B: rows,    data cols <- check cols   (Q1 <- SZ in data rows,    SX <- Q2 in check rows),    mask g(H2^T)
  C: columns, check rows <- data rows   (SX <- Q1, Q2 <- SZ),                                  mask g(H1)
  D: rows,    check cols <- data cols   (SZ <- Q1, Q2 <- SX),                                  mask g(H2)
Writes docs/figs/grid_processes.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "pdf.fonttype": 42})

QUBIT, CHECK = "#d6e4f7", "#d9efd9"          # block fills: qubit blocks, stabilizer blocks
QEDGE, CEDGE = "#4a6fa5", "#4f8a4f"
COL_A, COL_C = "#1f4fd0", "#7a3fd0"           # column processes (A: check->qubit, C: qubit->check)
COL_B, COL_D = "#d9701a", "#b02a2a"           # row processes
INK = "#222222"

W, H = 7.16, 2.55
fig = plt.figure(figsize=(W, H))


def grid(ax, x0, y0, w, h, split=(0.6, 0.62), labels=True, lw=0.9, fs=8, dims=True):
    """2x2 block grid; split = (fraction of width for data columns, fraction of height for data rows).
    Returns block rectangles as dict name -> (x, y, w, h)."""
    fx, fy = split
    wd, wc = w * fx, w * (1 - fx)
    hd, hc = h * fy, h * (1 - fy)
    blocks = {"Q1": (x0, y0 + hc, wd, hd), "SZ": (x0 + wd, y0 + hc, wc, hd),
              "SX": (x0, y0, wd, hc), "Q2": (x0 + wd, y0, wc, hc)}
    fill = {"Q1": (QUBIT, QEDGE), "Q2": (QUBIT, QEDGE), "SX": (CHECK, CEDGE), "SZ": (CHECK, CEDGE)}
    for k, (x, y, bw, bh) in blocks.items():
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=fill[k][0], edgecolor=fill[k][1], lw=lw))
    if labels:
        tex = {"Q1": r"$\mathbf{Q_1}$", "SZ": r"$\mathbf{S_Z}$", "SX": r"$\mathbf{S_X}$", "Q2": r"$\mathbf{Q_2}$"}
        dim = {"Q1": r"$n_1\times n_2$", "SZ": r"$n_1\times r_2$", "SX": r"$r_1\times n_2$", "Q2": r"$r_1\times r_2$"}
        for k, (x, y, bw, bh) in blocks.items():
            ax.text(x + bw / 2, y + bh / 2 + (0.09 if dims else 0), tex[k], ha="center", va="center", fontsize=fs + 1, color=INK)
            if dims:
                ax.text(x + bw / 2, y + bh / 2 - 0.13, dim[k], ha="center", va="center", fontsize=fs - 1.5, color="#444444")
    return blocks


def arrow(ax, p, q, color, lw=1.4, ms=9, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=5))


# ------------------------------------------------------------------ (a) the grid
ax = fig.add_axes([0.0, 0.0, 0.33, 1.0]); ax.set_xlim(0, 3.3); ax.set_ylim(0, 2.55); ax.axis("off")
gx, gy, gw, gh = 0.62, 0.38, 2.05, 1.72
B = grid(ax, gx, gy, gw, gh)
# axis labels
ax.text(gx + B["Q1"][2] / 2, gy + gh + 0.08, r"data cols $n_2$", ha="center", va="bottom", fontsize=7)
ax.text(gx + B["Q1"][2] + B["SZ"][2] / 2, gy + gh + 0.08, r"check cols $r_2$", ha="center", va="bottom", fontsize=7)
ax.text(gx - 0.08, gy + B["SX"][3] + B["Q1"][3] / 2, r"data rows $n_1$", ha="right", va="center", fontsize=7, rotation=90)
ax.text(gx - 0.08, gy + B["SX"][3] / 2, r"check rows $r_1$", ha="right", va="center", fontsize=7, rotation=90)
# one column and one row highlighted
cx = gx + 0.22; ax.add_patch(Rectangle((cx - 0.07, gy), 0.14, gh, facecolor="none", edgecolor=COL_A, lw=1.1, ls=(0, (2, 1.5)), zorder=4))
ax.text(cx, gy - 0.06, r"column $\gamma$", ha="center", va="top", fontsize=7, color=COL_A)
ry = gy + gh - 0.22; ax.add_patch(Rectangle((gx, ry - 0.07), gw, 0.14, facecolor="none", edgecolor=COL_B, lw=1.1, ls=(0, (2, 1.5)), zorder=4))
ax.text(gx + gw + 0.05, ry, r"row $\rho$", ha="left", va="center", fontsize=7, color=COL_B)
ax.text(gx + gw / 2, 0.04, r"(a) HGP grid, $(n_1{+}r_1)\times(n_2{+}r_2)$ tokens", ha="center", va="bottom", fontsize=7)

# ------------------------------------------------------------------ (b) the four processes
ax = fig.add_axes([0.335, 0.0, 0.40, 1.0]); ax.set_xlim(0, 4.0); ax.set_ylim(0, 2.55); ax.axis("off")
sw, sh = 1.3, 0.86
pos = {"A": (0.35, 1.32), "B": (2.15, 1.32), "C": (0.35, 0.22), "D": (2.15, 0.22)}
info = {"A": (r"A $\cdot$ columns $\cdot$ $g(H_1^{\top})$", r"check rows $\to$ data rows", COL_A),
        "B": (r"B $\cdot$ rows $\cdot$ $g(H_2^{\top})$", r"check cols $\to$ data cols", COL_B),
        "C": (r"C $\cdot$ columns $\cdot$ $g(H_1)$", r"data rows $\to$ check rows", COL_C),
        "D": (r"D $\cdot$ rows $\cdot$ $g(H_2)$", r"data cols $\to$ check cols", COL_D)}
for k, (x, y) in pos.items():
    b = grid(ax, x, y, sw, sh, labels=True, lw=0.7, fs=6.5, dims=False)
    c = info[k][2]
    Q1, SZ, SX, Q2 = b["Q1"], b["SZ"], b["SX"], b["Q2"]
    if k == "A":      # up, in two columns: SX -> Q1 and Q2 -> SZ
        for xx in (Q1[0] + 0.14, Q1[0] + Q1[2] - 0.14):
            arrow(ax, (xx, SX[1] + 0.08), (xx, Q1[1] + Q1[3] - 0.08), c)
        for xx in (SZ[0] + SZ[2] - 0.12,):
            arrow(ax, (xx, Q2[1] + 0.08), (xx, SZ[1] + SZ[3] - 0.08), c)
    elif k == "C":    # down
        for xx in (Q1[0] + 0.14, Q1[0] + Q1[2] - 0.14):
            arrow(ax, (xx, Q1[1] + Q1[3] - 0.08), (xx, SX[1] + 0.08), c)
        for xx in (SZ[0] + SZ[2] - 0.12,):
            arrow(ax, (xx, SZ[1] + SZ[3] - 0.08), (xx, Q2[1] + 0.08), c)
    elif k == "B":    # left, in two rows: SZ -> Q1 and Q2 -> SX
        for yy in (Q1[1] + 0.09, Q1[1] + Q1[3] - 0.09):
            arrow(ax, (SZ[0] + SZ[2] - 0.06, yy), (Q1[0] + 0.06, yy), c)
        for yy in (SX[1] + 0.07,):
            arrow(ax, (Q2[0] + Q2[2] - 0.06, yy), (SX[0] + 0.06, yy), c)
    else:             # right
        for yy in (Q1[1] + 0.09, Q1[1] + Q1[3] - 0.09):
            arrow(ax, (Q1[0] + 0.06, yy), (SZ[0] + SZ[2] - 0.06, yy), c)
        for yy in (SX[1] + 0.07,):
            arrow(ax, (SX[0] + 0.06, yy), (Q2[0] + Q2[2] - 0.06, yy), c)
    ax.text(x, y + sh + 0.15, info[k][0], ha="left", va="bottom", fontsize=6.5, color=c, fontweight="bold")
    ax.text(x, y + sh + 0.03, info[k][1], ha="left", va="bottom", fontsize=6.2, color=INK)
ax.text(2.0, 0.04, r"(b) the four processes of one layer (context $\to$ target)", ha="center", va="bottom", fontsize=7)

# ------------------------------------------------------------------ (c) the mask of one column
ax = fig.add_axes([0.745, 0.0, 0.255, 1.0]); ax.set_xlim(0, 2.55); ax.set_ylim(0, 2.55); ax.axis("off")
H1 = np.array([[1, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 0]])          # r1 x n1 toy example (3 x 5)
n1, r1 = H1.shape[1], H1.shape[0]
tx, ty0, dy = 0.55, 0.78, 0.27                                            # targets: Q1(i, gamma), i in [n1]
cxp, cy0, dyc = 1.75, 1.0, 0.44                                          # contexts: SX(a, gamma), a in [r1]
tys = [ty0 + i * dy for i in range(n1)][::-1]
cys = [cy0 + a * dyc for a in range(r1)][::-1]
for i, yy in enumerate(tys):
    ax.add_patch(Rectangle((tx - 0.11, yy - 0.09), 0.22, 0.18, facecolor=QUBIT, edgecolor=QEDGE, lw=0.7, zorder=3))
for a, yy in enumerate(cys):
    ax.add_patch(Rectangle((cxp - 0.11, yy - 0.09), 0.22, 0.18, facecolor=CHECK, edgecolor=CEDGE, lw=0.7, zorder=3))
for a in range(r1):
    for i in range(n1):
        if H1[a, i]:
            ax.plot([cxp - 0.11, tx + 0.11], [cys[a], tys[i]], color=COL_A, lw=0.9, alpha=0.9, zorder=2)
ax.text(tx, tys[0] + 0.2, "targets\n" + r"$\mathbf{Q_1}(\cdot,\gamma)$", ha="center", va="bottom", fontsize=6.6, linespacing=1.1)
ax.text(tx, tys[-1] - 0.17, r"$n_1$ data rows", ha="center", va="top", fontsize=6.3, color="#444444")
ax.text(cxp, cys[0] + 0.2, "contexts\n" + r"$\mathbf{S_X}(\cdot,\gamma)$", ha="center", va="bottom", fontsize=6.6, linespacing=1.1)
ax.text(cxp, cys[-1] - 0.17, r"$r_1$ check rows", ha="center", va="top", fontsize=6.3, color="#444444")
ax.text(1.15, 0.36, r"$(i,a)$ attends iff $H_1[a,i]=1$", ha="center", va="center", fontsize=6.8, color=COL_A)
ax.text(1.15, 0.04, r"(c) process A inside column $\gamma$", ha="center", va="bottom", fontsize=7)
ax.text(1.15, 0.2, r"(check column: $\mathbf{S_Z}\leftarrow\mathbf{Q_2}$, same mask)", ha="center", va="bottom", fontsize=6.2, color="#444444")

fig.savefig(os.path.join(OUT, "grid_processes.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(OUT, "grid_processes.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote grid_processes.pdf")
