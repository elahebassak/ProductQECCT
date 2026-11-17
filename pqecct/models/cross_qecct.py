"""
CrossQECCT baseline: CrossMPT (Park et al., ICLR 2025) adapted to the quantum setting exactly the
way DQEC adapts ECCT: the magnitude input is replaced by the dense noise estimate g_omega(s), and the
DQEC losses are used.  Global *cross*-attention between the noise-token sequence (n_tot) and the
syndrome-token sequence (r_tot) with masks H^T (VN side) and H (CN side).

This isolates the contribution of cross-attention (CrossMPT) from that of the product
factorization (ProductQECCT): same tokens, same masks (globally), no row/column structure.
"""
import torch
import torch.nn as nn
from .common import CrossAttentionBlock, clones
from .qecct import active_pcm, DenseNoiseEstimator


class CrossQECCT(nn.Module):
    def __init__(self, code, err_types="xz", d_model=128, h=8, N_dec=6, d_ff=None, g_hidden_mult=5,
                 no_g=False, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.code = code
        self.n = code.n
        self.err_types = err_types
        H = active_pcm(code, err_types)
        self.r_tot, self.n_tot = H.shape
        self.no_g = no_g
        self.g = None if no_g else DenseNoiseEstimator(self.r_tot, self.n_tot, g_hidden_mult)
        self.embed_VN = nn.Parameter(torch.empty(self.n_tot, d_model))
        self.embed_CN = nn.Parameter(torch.empty(self.r_tot, d_model))
        # CrossMPT shares one block for both directions inside a layer
        self.layers = clones(CrossAttentionBlock(d_model, h, d_ff, dropout), N_dec)
        self.norm = nn.LayerNorm(d_model)
        self.oned = nn.Linear(d_model, 1)
        self.out_fc = nn.Linear(self.n_tot + self.r_tot, self.n_tot)
        Hb = torch.tensor(H, dtype=torch.bool)
        self.register_buffer("mask_VN", (~Hb.T)[None, None])   # VN (n_tot) attends CN (r_tot)
        self.register_buffer("mask_CN", (~Hb)[None, None])     # CN attends VN
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
        zeros = torch.zeros(B, self.n, device=out.device, dtype=out.dtype)
        if self.err_types == "x":
            return torch.cat([out, zeros], dim=1)
        if self.err_types == "z":
            return torch.cat([zeros, out], dim=1)
        return out

    def forward(self, s_x, s_z):
        s = self._pack_syndromes(s_x, s_z)
        if self.no_g:
            g = torch.ones(s.shape[0], self.n_tot, device=s.device)
            g_logits = None
        else:
            g = self.g(s)
            g_logits = self._unpack_logits(g)
        VN = self.embed_VN.unsqueeze(0) * g.unsqueeze(-1)
        CN = self.embed_CN.unsqueeze(0) * (1.0 - 2.0 * s).unsqueeze(-1)
        for layer in self.layers:
            VN = layer(VN, CN, self.mask_VN)
            CN = layer(CN, VN, self.mask_CN)
        x = torch.cat([self.norm(VN), self.norm(CN)], dim=1)
        out = self.out_fc(self.oned(x).squeeze(-1))
        return self._unpack_logits(out), g_logits
