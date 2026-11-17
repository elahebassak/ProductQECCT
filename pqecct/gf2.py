"""
GF(2) linear algebra utilities (dense numpy, uint8).

Used to build logical operators of CSS codes and to check code parameters.
"""
import numpy as np


def _as_u8(M):
    return (np.asarray(M) % 2).astype(np.uint8)


def row_reduce(M):
    """Reduced row echelon form over GF(2).

    Returns (R, pivots) where R is the RREF and pivots the pivot column list.
    """
    R = _as_u8(M).copy()
    rows, cols = R.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = np.nonzero(R[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        # eliminate all other rows
        others = np.nonzero(R[:, c])[0]
        others = others[others != r]
        R[others] ^= R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(M):
    return len(row_reduce(M)[1])


def nullspace(M):
    """Basis (rows) of {x : M x = 0 over GF(2)}. Returns array (dim, n)."""
    M = _as_u8(M)
    R, pivots = row_reduce(M)
    n = M.shape[1]
    free = [c for c in range(n) if c not in set(pivots)]
    basis = []
    for f in free:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for i, p in enumerate(pivots):
            if R[i, f]:
                x[p] = 1
        basis.append(x)
    if len(basis) == 0:
        return np.zeros((0, n), dtype=np.uint8)
    return np.stack(basis)


def independent_rows(M):
    """Return indices of a maximal set of linearly independent rows (greedy, in order)."""
    M = _as_u8(M)
    keep = []
    basis = np.zeros((0, M.shape[1]), dtype=np.uint8)
    cur_rank = 0
    for i in range(M.shape[0]):
        cand = np.vstack([basis, M[i:i + 1]])
        rk = rank(cand)
        if rk > cur_rank:
            keep.append(i)
            basis = cand
            cur_rank = rk
    return keep


def quotient_basis(K, S):
    """Basis of span(K) / span(S): rows of K that are independent modulo span(S).

    K: (m, n) candidate vectors (e.g. a kernel basis); S: (s, n) subspace rows.
    Returns array (q, n).
    """
    K = _as_u8(K)
    S = _as_u8(S)
    base_rank = rank(S)
    chosen = []
    cur = S.copy()
    cur_rank = base_rank
    for i in range(K.shape[0]):
        cand = np.vstack([cur, K[i:i + 1]])
        rk = rank(cand)
        if rk > cur_rank:
            chosen.append(K[i])
            cur = cand
            cur_rank = rk
    if len(chosen) == 0:
        return np.zeros((0, K.shape[1]), dtype=np.uint8)
    return np.stack(chosen)


def css_logicals(Hx, Hz):
    """Logical operators of a CSS code.

    Lx: basis of ker(Hz) / rowspace(Hx)   (X-type logicals, k x n)
    Lz: basis of ker(Hx) / rowspace(Hz)   (Z-type logicals, k x n)
    """
    Hx = _as_u8(Hx)
    Hz = _as_u8(Hz)
    assert not np.any((Hx.astype(np.int64) @ Hz.T.astype(np.int64)) % 2), "Hx Hz^T != 0: not a CSS code"
    Lx = quotient_basis(nullspace(Hz), Hx)
    Lz = quotient_basis(nullspace(Hx), Hz)
    assert Lx.shape[0] == Lz.shape[0], (Lx.shape, Lz.shape)
    return Lx, Lz


def min_weight_upper_bound(L, S, trials=200, seed=0):
    """Heuristic upper bound on the minimum weight of a nontrivial logical operator.

    For random nonzero combinations of logical basis rows, greedily add stabilizer rows (S)
    while the weight decreases (hill climbing). Cheap; used only for reporting.
    """
    rng = np.random.default_rng(seed)
    L = _as_u8(L); S = _as_u8(S)
    if L.shape[0] == 0:
        return None
    best = L.shape[1]
    kL = L.shape[0]
    starts = [L[i] for i in range(kL)]
    for _ in range(trials):
        cl = rng.integers(0, 2, size=kL)
        if not cl.any():
            cl[rng.integers(0, kL)] = 1
        starts.append((cl.astype(np.int64) @ L.astype(np.int64) % 2).astype(np.uint8))
    for v in starts:
        v = v.copy()
        improved = True
        while improved:
            improved = False
            w = v.sum()
            cand = (S ^ v[None, :]).sum(1)
            j = int(np.argmin(cand))
            if cand[j] < w:
                v ^= S[j]
                improved = True
        best = min(best, int(v.sum()))
    return best
