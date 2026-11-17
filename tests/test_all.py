"""
Quick correctness tests.  Run from the project root:   python -m pytest -q tests   (or  python tests/test_all.py)
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pqecct.codes import get_code, hgp, repetition_pcm  # noqa: E402
from pqecct import gf2  # noqa: E402
from pqecct.noise import NoiseSampler  # noqa: E402
from pqecct.models import build_model  # noqa: E402
from pqecct.models.common import decoder_loss, decode_metrics, soft_parity, hard_parity  # noqa: E402


def test_hgp_is_css_and_logicals():
    for name in ["toric-L4", "toric-L5", "hgp-400", "hgp-asym"]:
        c = get_code(name)
        assert not ((c.Hx.astype(int) @ c.Hz.T.astype(int)) % 2).any()
        assert not ((c.Lx.astype(int) @ c.Hz.T.astype(int)) % 2).any()
        assert not ((c.Lz.astype(int) @ c.Hx.T.astype(int)) % 2).any()
        # Lx and Lz pair non-degenerately: Lx Lz^T has full rank k
        assert gf2.rank((c.Lx.astype(int) @ c.Lz.T.astype(int)) % 2) == c.k
        if name.startswith("toric"):
            assert c.k == 2 and c.n == 2 * c.grid["n1"] ** 2


def test_hgp_parameters():
    # k = k1*k2 + k1T*k2T
    H1 = repetition_pcm(5); H2 = repetition_pcm(7)
    c = hgp(H1, H2)
    k1 = 5 - gf2.rank(H1); k1T = 5 - gf2.rank(H1)
    k2 = 7 - gf2.rank(H2); k2T = 7 - gf2.rank(H2)
    assert c.k == k1 * k2 + k1T * k2T
    assert c.n == 5 * 7 + 5 * 7


def test_grid_adjacency_matches_pcm():
    """The row/column masks used by ProductQECCT reproduce the HGP Tanner graph exactly."""
    c = get_code("hgp-asym")
    g = c.grid
    n1, n2, r1, r2, H1, H2 = g["n1"], g["n2"], g["r1"], g["r2"], g["H1"], g["H2"]
    # X-check (a,j) vs Q1 (i,j): H1[a,i]; vs Q2 (a,b): H2[b,j]
    for a in range(r1):
        for j in range(n2):
            row = c.Hx[a * n2 + j]
            for i in range(n1):
                assert row[i * n2 + j] == H1[a, i]
            for b in range(r2):
                assert row[n1 * n2 + a * r2 + b] == H2[b, j]
    # Z-check (i,b) vs Q1 (i,j): H2[b,j]; vs Q2 (a,b): H1[a,i]
    for i in range(n1):
        for b in range(r2):
            row = c.Hz[i * r2 + b]
            for j in range(n2):
                assert row[i * n2 + j] == H2[b, j]
            for a in range(r1):
                assert row[n1 * n2 + a * r2 + b] == H1[a, i]


def test_noise_and_syndromes():
    c = get_code("toric-L4")
    s = NoiseSampler(c, "depolarizing", "xz")
    e_x, e_z, s_x, s_z = s.sample_batch(0.1, 64)
    assert torch.equal(s_z, torch.remainder(e_x @ s.HzT, 2))
    assert torch.equal(s_x, torch.remainder(e_z @ s.HxT, 2))
    # numpy sampler has the same marginals (statistically)
    rng = np.random.default_rng(0)
    ex, ez, sx, sz = s.sample_batch_numpy(0.1, 20000, rng)
    assert abs(ex.mean() - 2 * 0.1 / 3) < 0.01 and abs(ez.mean() - 2 * 0.1 / 3) < 0.01
    s2 = NoiseSampler(c, "independent", "x")
    e_x, e_z, *_ = s2.sample_batch(0.1, 64)
    assert e_z.sum() == 0


def test_soft_parity_matches_hard():
    c = get_code("toric-L4")
    s = NoiseSampler(c, "depolarizing", "xz")
    e_x, e_z, _, _ = s.sample_batch(0.1, 32)
    sp = soft_parity(s.LzT, e_x)
    hp = hard_parity(s.LzT, e_x)
    assert torch.allclose(sp, hp)


def test_models_forward_backward():
    torch.manual_seed(0)
    for cname in ["toric-L4", "hgp-asym"]:
        code = get_code(cname)
        for et in ["x", "xz"]:
            samp = NoiseSampler(code, "depolarizing", et)
            for m in ["product", "qecct", "crossqecct"]:
                a = argparse.Namespace(model=m, d_model=32, h=4, N_dec=2, no_share=False, d_g=16, N_g=1, no_g=0,
                                       dropout=0.0, no_out_mix=False, err_types=et, g_hidden_mult=2, no_mask=0)
                model = build_model(a, code)
                e_x, e_z, s_x, s_z = samp.sample_batch(0.05, 4)
                logits, g = model(s_x, s_z)
                assert logits.shape == (4, 2 * code.n)
                loss, _ = decoder_loss(logits, g, e_x, e_z, samp, et)
                loss.backward()
                assert torch.isfinite(loss)
                met = decode_metrics(logits, e_x, e_z, s_x, s_z, samp, et)
                assert met["samples"] == 4


def test_perfect_prediction_gives_zero_ler():
    code = get_code("hgp-400")
    samp = NoiseSampler(code, "depolarizing", "xz")
    e_x, e_z, s_x, s_z = samp.sample_batch(0.05, 16)
    logits = torch.cat([2 * e_x - 1, 2 * e_z - 1], dim=1) * 5
    met = decode_metrics(logits, e_x, e_z, s_x, s_z, samp, "xz")
    assert met["ler"] == 0 and met["ler_strict"] == 0 and met["bit_errors"] == 0
    # a stabilizer added to the truth is still a success; a logical added is a failure
    stab = torch.tensor(code.Hx[0], dtype=torch.float32)
    e2 = torch.remainder(e_x + stab, 2)
    logits2 = torch.cat([2 * e2 - 1, 2 * e_z - 1], dim=1) * 5
    met2 = decode_metrics(logits2, e_x, e_z, s_x, s_z, samp, "xz")
    assert met2["ler_strict"] == 0
    log = torch.tensor(code.Lx[0], dtype=torch.float32)
    e3 = torch.remainder(e_x + log, 2)
    logits3 = torch.cat([2 * e3 - 1, 2 * e_z - 1], dim=1) * 5
    met3 = decode_metrics(logits3, e_x, e_z, s_x, s_z, samp, "xz")
    assert met3["ler"] == 16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
