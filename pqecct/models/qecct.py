"""
QECCT baseline: re-implementation of "Deep Quantum Error Correction" (Choukroun & Wolf, AAAI 2024)
in the common interface.  Global masked self-attention over [g_omega(s) | s] tokens, ECCT mask
derived from the parity-check matrix, dense MLP noise estimator, dense output layer.

For err_types='x'  : tokens = [n noise tokens (e_x) | rz syndrome tokens (s_z)],  H = H_Z
    err_types='z'  : tokens = [n (e_z) | rx (s_x)],                               H = H_X
    err_types='xz' : tokens = [2n (e_x, e_z) | rz + rx (s_z, s_x)], H = blockdiag(H_Z, H_X)
(the last one is the DQEC "depolarization" configuration: H = [H_Z 0; 0 H_X]).
"""
import numpy as np
import torch
import torch.nn as nn
from .common import SelfAttentionBlock, clones


def active_pcm(code, err_types):
    """Returns H (r_tot x n_tot) over the active error types, with DQEC ordering."""
    if err_types == "x":
        return code.Hz
    if err_types == "z":
        return code.Hx
    from scipy.linalg import block_diag
    return block_diag(code.Hz, code.Hx).astype(np.uint8)


def ecct_mask(H):
    """ECCT/DQEC mask: identity + bit-bit pairs sharing a check + bit-check incidences. True = blocked."""
    r, n = H.shape
    size = n + r
    mask = np.eye(size, dtype=bool)
    for ii in range(r):
        idx = np.nonzero(H[ii])[0]
        mask[np.ix_(idx, idx)] = True
        mask[n + ii, idx] = True
        mask[idx, n + ii] = True
    return ~mask


class DenseNoiseEstimator(nn.Module):
    """DQEC g_omega: MLP r -> 5n -> 5n -> n (GELU)."""

    def __init__(self, r, n, hidden_mult=5):
        super().__init__()
        hdim = hidden_mult * n
        self.net = nn.Sequential(nn.Linear(r, hdim), nn.GELU(), nn.Linear(hdim, hdim), nn.GELU(), nn.Linear(hdim, n))

    def forward(self, s):
        return self.net(s)


class QECCT(nn.Module):
    def __init__(self, code, err_types="xz", d_model=128, h=8, N_dec=6, d_ff=None, g_hidden_mult=5,
                 no_g=False, no_mask=False, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.code = code
        self.n = code.n
        self.err_types = err_types
        H = active_pcm(code, err_types)
        self.r_tot, self.n_tot = H.shape
        self.no_g = no_g
        self.g = None if no_g else DenseNoiseEstimator(self.r_tot, self.n_tot, g_hidden_mult)
        self.src_embed = nn.Parameter(torch.empty(self.n_tot + self.r_tot, d_model))
        self.layers = clones(SelfAttentionBlock(d_model, h, d_ff, dropout), N_dec)
        self.norm = nn.LayerNorm(d_model)
        self.oned = nn.Linear(d_model, 1)
        self.out_fc = nn.Linear(self.n_tot + self.r_tot, self.n_tot)
        mask = torch.tensor(ecct_mask(H))[None, None]
        self.register_buffer("mask", None if no_mask else mask)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _pack_syndromes(self, s_x, s_z):
        if self.err_types == "x":
            return s_z
        if self.err_types == "z":
            return s_x
        return torch.cat([s_z, s_x], dim=1)

    def _unpack_logits(self, out):
        B = out.shape[0]
        n = self.n
        zeros = torch.zeros(B, n, device=out.device, dtype=out.dtype)
        if self.err_types == "x":
            return torch.cat([out, zeros], dim=1)
        if self.err_types == "z":
            return torch.cat([zeros, out], dim=1)
        return out  # already [e_x | e_z]

    def forward(self, s_x, s_z):
        s = self._pack_syndromes(s_x, s_z)
        if self.no_g:
            g = torch.ones(s.shape[0], self.n_tot, device=s.device)
            g_logits = None
        else:
            g = self.g(s)
            g_logits = self._unpack_logits(g)
        values = torch.cat([g, 1.0 - 2.0 * s], dim=1).unsqueeze(-1)   # (B, n_tot + r_tot, 1)
        x = self.src_embed.unsqueeze(0) * values
        for layer in self.layers:
            x = layer(x, self.mask)
        x = self.norm(x)
        out = self.out_fc(self.oned(x).squeeze(-1))
        return self._unpack_logits(out), g_logits
