"""
Quantum code constructions.

* Hypergraph product (HGP) codes  (Tillich & Zemor):
      H_X = [ H1 (x) I_{n2} | I_{r1} (x) H2^T ]
      H_Z = [ I_{n1} (x) H2 | H1^T (x) I_{r2} ]
  with n = n1*n2 + r1*r2 qubits (two sectors) and k = k1*k2 + k1T*k2T.

* The toric code is the HGP of two cyclic repetition codes (H1 = H2 = cyclic rep. PCM).

The 2-D *grid* view used by ProductQECCT: rows indexed by [n1] U [r1], columns by [n2] U [r2]

        +---------------------+---------------------+
        |  Q1  (n1 x n2)      |  S_Z (n1 x r2)      |      Q1 : sector-1 qubits   (i, j)
        +---------------------+---------------------+      S_Z: Z-checks          (i, b)
        |  S_X (r1 x n2)      |  Q2  (r1 x r2)      |      S_X: X-checks          (a, j)
        +---------------------+---------------------+      Q2 : sector-2 qubits   (a, b)

  X-check (a, j) touches Q1(i, j) iff H1[a, i] = 1 (down its column, via H1) and
                          Q2(a, b) iff H2[b, j] = 1 (along its row, via H2^T).
  Z-check (i, b) touches Q1(i, j) iff H2[b, j] = 1 (along its row) and
                          Q2(a, b) iff H1[a, i] = 1 (down its column).
  => every constraint is a column-constraint under H1 and a row-constraint under H2,
     exactly like the row/column parity constraints of a classical product code.

Flat index conventions (used everywhere, including H_X / H_Z columns):
    qubit  Q1(i, j) -> i*n2 + j                 (i in [n1], j in [n2])
    qubit  Q2(a, b) -> n1*n2 + a*r2 + b         (a in [r1], b in [r2])
    X-check (a, j)  -> a*n2 + j                 (rows of H_X)
    Z-check (i, b)  -> i*r2 + b                 (rows of H_Z)
"""
import os
import re
import numpy as np
from . import gf2


# ----------------------------------------------------------------------------
# Classical parity-check matrices
# ----------------------------------------------------------------------------
def repetition_pcm(L, cyclic=True):
    """Parity-check matrix of the length-L repetition code. Cyclic version has L rows (rank L-1)."""
    if cyclic:
        H = np.zeros((L, L), dtype=np.uint8)
        for i in range(L):
            H[i, i] = 1
            H[i, (i + 1) % L] = 1
    else:
        H = np.zeros((L - 1, L), dtype=np.uint8)
        for i in range(L - 1):
            H[i, i] = 1
            H[i, i + 1] = 1
    return H


def random_regular_ldpc(r, n, dv, dc, seed=0, max_tries=1000):
    """Random (dv, dc)-regular LDPC parity-check matrix of size r x n (needs n*dv == r*dc).

    Configuration model with rejection of multi-edges. 4-cycles are allowed (they are
    unavoidable in HGP codes anyway). Deterministic given the seed.
    """
    assert n * dv == r * dc, f"regularity requires n*dv == r*dc (got n={n}, dv={dv}, r={r}, dc={dc})"
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        stubs_v = np.repeat(np.arange(n), dv)
        stubs_c = np.repeat(np.arange(r), dc)
        rng.shuffle(stubs_v)
        H = np.zeros((r, n), dtype=np.uint8)
        ok = True
        for c, v in zip(stubs_c, stubs_v):
            if H[c, v]:
                ok = False
                break
            H[c, v] = 1
        if ok and np.all(H.sum(0) == dv) and np.all(H.sum(1) == dc):
            return H
    raise RuntimeError("could not sample a simple regular bipartite graph; change seed")


def load_alist(path):
    """Read an .alist file (MacKay format) into a dense uint8 parity-check matrix (r x n)."""
    with open(path) as f:
        tokens = f.read().split()
    it = iter(tokens)
    n = int(next(it)); r = int(next(it))
    next(it); next(it)  # max degrees
    col_deg = [int(next(it)) for _ in range(n)]
    row_deg = [int(next(it)) for _ in range(r)]
    H = np.zeros((r, n), dtype=np.uint8)
    for j in range(n):
        # alist pads entries with zeros up to max col degree in some files; read exactly col_deg[j]
        for _ in range(col_deg[j]):
            i = int(next(it))
            if i > 0:
                H[i - 1, j] = 1
    return H


# ----------------------------------------------------------------------------
# CSS code container
# ----------------------------------------------------------------------------
class CSSCode:
    def __init__(self, Hx, Hz, name="css", grid=None, compute_logicals=True):
        self.Hx = (np.asarray(Hx) % 2).astype(np.uint8)
        self.Hz = (np.asarray(Hz) % 2).astype(np.uint8)
        assert self.Hx.shape[1] == self.Hz.shape[1]
        self.n = self.Hx.shape[1]
        self.rx = self.Hx.shape[0]   # number of X-checks (detect Z errors)
        self.rz = self.Hz.shape[0]   # number of Z-checks (detect X errors)
        self.name = name
        self.grid = grid             # dict(n1, n2, r1, r2, H1, H2) for HGP codes, else None
        if compute_logicals:
            self.Lx, self.Lz = gf2.css_logicals(self.Hx, self.Hz)
            self.k = self.Lx.shape[0]
        else:
            self.Lx = self.Lz = None
            self.k = None

    def __repr__(self):
        g = ""
        if self.grid is not None:
            g = f", grid rows={self.grid['n1']}+{self.grid['r1']} cols={self.grid['n2']}+{self.grid['r2']}"
        return f"CSSCode({self.name}: [[{self.n},{self.k}]], rx={self.rx}, rz={self.rz}{g})"

    def is_hgp(self):
        return self.grid is not None

    def summary(self, distance_trials=0):
        s = repr(self)
        if distance_trials > 0 and self.Lx is not None:
            dx = gf2.min_weight_upper_bound(self.Lx, self.Hx, trials=distance_trials)
            dz = gf2.min_weight_upper_bound(self.Lz, self.Hz, trials=distance_trials)
            s += f", d <= min({dx},{dz}) (randomized upper bound)"
        return s


def hgp(H1, H2, name=None):
    """Hypergraph product of two classical codes with PCMs H1 (r1 x n1), H2 (r2 x n2)."""
    H1 = (np.asarray(H1) % 2).astype(np.uint8)
    H2 = (np.asarray(H2) % 2).astype(np.uint8)
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    I = np.eye
    Hx = np.concatenate([np.kron(H1, I(n2, dtype=np.uint8)), np.kron(I(r1, dtype=np.uint8), H2.T)], axis=1)
    Hz = np.concatenate([np.kron(I(n1, dtype=np.uint8), H2), np.kron(H1.T, I(r2, dtype=np.uint8))], axis=1)
    grid = dict(n1=n1, n2=n2, r1=r1, r2=r2, H1=H1, H2=H2)
    if name is None:
        name = f"hgp({r1}x{n1},{r2}x{n2})"
    return CSSCode(Hx, Hz, name=name, grid=grid)


def toric_code(L):
    """Toric code with 2L^2 qubits as HGP of two cyclic repetition codes."""
    H = repetition_pcm(L, cyclic=True)
    return hgp(H, H, name=f"toric-L{L}")


# ----------------------------------------------------------------------------
# Registry / name parser
# ----------------------------------------------------------------------------
# Named presets used in the experiments (classical component codes are random (3,4)-regular LDPC
# codes generated with a fixed seed, so every machine builds identical codes).
PRESETS = {
    # name           : (r1, n1, dv, dc, seed1, r2, n2, dv, dc, seed2)
    "hgp-400":  (12, 16, 3, 4, 4, 12, 16, 3, 4, 4),    # [[400, 16]]  (16^2 + 12^2), full-rank H
    "hgp-625":  (15, 20, 3, 4, 7, 15, 20, 3, 4, 7),    # [[625, 25]]  (20^2 + 15^2), full-rank H
    "hgp-1600": (24, 32, 3, 4, 0, 24, 32, 3, 4, 0),    # [[1600, 64]] (32^2 + 24^2), full-rank H
    "hgp-3600": (36, 48, 3, 4, 0, 36, 48, 3, 4, 0),    # [[3600, 144]] (48^2 + 36^2), full-rank H
    "hgp-asym": (12, 16, 3, 4, 4, 15, 20, 3, 4, 7),    # [[500, 20]]  (16*20 + 12*15): non-square grid
}


def _classical_from_spec(spec):
    """spec: 'rep{L}' | 'ldpc-r{r}-n{n}-dv{dv}-dc{dc}-s{seed}' | 'alist:<path>'"""
    if spec.startswith("alist:"):
        return load_alist(spec[len("alist:"):])
    m = re.fullmatch(r"rep(\d+)", spec)
    if m:
        return repetition_pcm(int(m.group(1)), cyclic=True)
    m = re.fullmatch(r"ldpc-r(\d+)-n(\d+)-dv(\d+)-dc(\d+)-s(\d+)", spec)
    if m:
        r, n, dv, dc, s = map(int, m.groups())
        return random_regular_ldpc(r, n, dv, dc, seed=s)
    raise ValueError(f"unknown classical code spec '{spec}'")


def get_code(name, cache_dir=None):
    """Build a code from its name.

      toric-L{L}                      toric code, 2L^2 qubits
      hgp-400 | hgp-625 | hgp-1600 | hgp-3600 | hgp-asym   presets (see PRESETS)
      hgp:<spec1>:<spec2>             general HGP, e.g. hgp:ldpc-r12-n16-dv3-dc4-s0:rep5
                                      or hgp:alist:/path/H1.alist:alist:/path/H2.alist
    """
    m = re.fullmatch(r"toric-L(\d+)", name)
    if m:
        return toric_code(int(m.group(1)))
    if name in PRESETS:
        r1, n1, dv1, dc1, s1, r2, n2, dv2, dc2, s2 = PRESETS[name]
        H1 = random_regular_ldpc(r1, n1, dv1, dc1, seed=s1)
        H2 = random_regular_ldpc(r2, n2, dv2, dc2, seed=s2)
        return hgp(H1, H2, name=name)
    if name.startswith("hgp:"):
        rest = name[4:]
        # split on ':' but keep 'alist:<path>' tokens together
        parts = []
        toks = rest.split(":")
        i = 0
        while i < len(toks):
            if toks[i] == "alist":
                parts.append("alist:" + toks[i + 1]); i += 2
            else:
                parts.append(toks[i]); i += 1
        assert len(parts) == 2, f"hgp needs two component codes, got {parts}"
        return hgp(_classical_from_spec(parts[0]), _classical_from_spec(parts[1]), name=name)
    raise ValueError(f"unknown code name '{name}'")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1:] or ["toric-L4", "hgp-400", "hgp-asym"]):
        c = get_code(nm)
        print(c.summary(distance_trials=2000))
