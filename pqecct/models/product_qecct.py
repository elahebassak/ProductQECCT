"""
ProductQECCT: product-structured message-passing transformer decoder for hypergraph-product
(and toric/surface) quantum codes.  (Proposed method.)

Tokens live on the HGP grid (see pqecct/codes.py docstring):

        rows  [n1] | [r1]          cols  [n2] | [r2]
        +-----------+-----------+
        |  Q1       |  S_Z      |   <- "data rows"  (target of Process A, context of Process C)
        +-----------+-----------+
        |  S_X      |  Q2       |   <- "check rows" (context of Process A, target of Process C)
        +-----------+-----------+
          data cols   check cols
          (target of B, ctx of D)  (ctx of B, target of D)

Each decoder layer performs four masked cross-attention processes (ProductMPT, Fig. 2b):
    A: column domain, data rows  <- check rows,  mask H1^T  (n1 x r1)   [applied to every column]
    B: row domain,    data cols  <- check cols,  mask H2^T  (n2 x r2)   [applied to every row]
    C: column domain, check rows <- data rows,   mask H1    (r1 x n1)
    D: row domain,    check cols <- data cols,   mask H2    (r2 x n2)
These masks are *exactly* the HGP Tanner-graph adjacency restricted to a row / a column:
    X-check (a,j) ~ Q1(i,j) iff H1[a,i]=1 ;  X-check (a,j) ~ Q2(a,b) iff H2[b,j]=1
    Z-check (i,b) ~ Q1(i,j) iff H2[b,j]=1 ;  Z-check (i,b) ~ Q2(a,b) iff H1[a,i]=1
so the attention pattern is the full Tanner graph, but computed as dense attention over
short rows / columns: cost O(n1 n2 (r1 + r2) + r1 r2 (n1 + n2)) per layer instead of
O((2n + r)^2) for global masked self-attention (QECCT).

Measurement collapse is handled as in DQEC: a (here factorized) noise estimator g_omega(s)
produces an initial per-qubit error estimate, which the decoder refines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import CrossAttentionBlock, clones


class GridProcessLayer(nn.Module):
    """One decoder layer = processes A, B, C, D (optionally with shared parameters)."""

    def __init__(self, d_model, h, d_ff, share=True, dropout=0.0, softcap=30.0):
        super().__init__()
        if share:
            blk = CrossAttentionBlock(d_model, h, d_ff, dropout, softcap)
            self.blocks = nn.ModuleList([blk, blk, blk, blk])
        else:
            self.blocks = clones(CrossAttentionBlock(d_model, h, d_ff, dropout, softcap), 4)

    @staticmethod
    def _col_domain(block, phi, n_t, target_first, mask):
        """Attend across rows inside each column. phi: (B, R, C, d).
        target rows = [:n_t] if target_first else [n_t:] (then n_t is the split point)."""
        B, R, C, d = phi.shape
        if target_first:
            t, c = phi[:, :n_t], phi[:, n_t:]
        else:
            t, c = phi[:, n_t:], phi[:, :n_t]
        Lt, Lc = t.shape[1], c.shape[1]
        t2 = t.permute(0, 2, 1, 3).reshape(B * C, Lt, d)
        c2 = c.permute(0, 2, 1, 3).reshape(B * C, Lc, d)
        t2 = block(t2, c2, mask)
        t = t2.reshape(B, C, Lt, d).permute(0, 2, 1, 3)
        return torch.cat([t, c], dim=1) if target_first else torch.cat([c, t], dim=1)

    @staticmethod
    def _row_domain(block, phi, n_t, target_first, mask):
        """Attend across columns inside each row."""
        B, R, C, d = phi.shape
        if target_first:
            t, c = phi[:, :, :n_t], phi[:, :, n_t:]
        else:
            t, c = phi[:, :, n_t:], phi[:, :, :n_t]
        Lt, Lc = t.shape[2], c.shape[2]
        t2 = t.reshape(B * R, Lt, d)
        c2 = c.reshape(B * R, Lc, d)
        t2 = block(t2, c2, mask)
        t = t2.reshape(B, R, Lt, d)
        return torch.cat([t, c], dim=2) if target_first else torch.cat([c, t], dim=2)

    def forward(self, phi, n1, n2, masks):
        mA, mB, mC, mD = masks
        phi = self._col_domain(self.blocks[0], phi, n1, True, mA)    # A: data rows  <- check rows
        phi = self._row_domain(self.blocks[1], phi, n2, True, mB)    # B: data cols  <- check cols
        phi = self._col_domain(self.blocks[2], phi, n1, False, mC)   # C: check rows <- data rows
        phi = self._row_domain(self.blocks[3], phi, n2, False, mD)   # D: check cols <- data cols
        return phi


class GridEncoder(nn.Module):
    """Embeds (qubit features, syndromes) onto the grid, runs N GridProcessLayers, returns grid states."""

    def __init__(self, code, d_model, h, N, d_ff, q_feat_dim, share=True, dropout=0.0):
        super().__init__()
        g = code.grid
        self.n1, self.n2, self.r1, self.r2 = g["n1"], g["n2"], g["r1"], g["r2"]
        self.R, self.C = self.n1 + self.r1, self.n2 + self.r2
        self.d = d_model
        self.q_proj = nn.Linear(q_feat_dim, d_model)
        self.s_proj = nn.Linear(1, d_model)
        self.row_emb = nn.Parameter(torch.zeros(self.R, d_model))
        self.col_emb = nn.Parameter(torch.zeros(self.C, d_model))
        self.type_emb = nn.Parameter(torch.zeros(4, d_model))  # Q1, SZ, SX, Q2
        nn.init.normal_(self.row_emb, std=0.02); nn.init.normal_(self.col_emb, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        self.embed_norm = nn.LayerNorm(d_model)
        self.layers = clones(GridProcessLayer(d_model, h, d_ff, share, dropout), N)
        self.norm = nn.LayerNorm(d_model)
        H1 = torch.tensor(g["H1"], dtype=torch.bool)
        H2 = torch.tensor(g["H2"], dtype=torch.bool)
        # True = blocked
        self.register_buffer("mask_A", (~H1.T)[None, None])   # (1,1,n1,r1)
        self.register_buffer("mask_B", (~H2.T)[None, None])   # (1,1,n2,r2)
        self.register_buffer("mask_C", (~H1)[None, None])     # (1,1,r1,n1)
        self.register_buffer("mask_D", (~H2)[None, None])     # (1,1,r2,n2)

    def embed(self, q_feat, s_x, s_z):
        """q_feat: (B, n, f) qubit features in flat qubit order; s_x: (B, rx), s_z: (B, rz)."""
        B = q_feat.shape[0]
        n1, n2, r1, r2, d = self.n1, self.n2, self.r1, self.r2, self.d
        q = self.q_proj(q_feat)
        Q1 = q[:, :n1 * n2].reshape(B, n1, n2, d) + self.type_emb[0]
        Q2 = q[:, n1 * n2:].reshape(B, r1, r2, d) + self.type_emb[3]
        SZ = self.s_proj((1.0 - 2.0 * s_z).unsqueeze(-1)).reshape(B, n1, r2, d) + self.type_emb[1]
        SX = self.s_proj((1.0 - 2.0 * s_x).unsqueeze(-1)).reshape(B, r1, n2, d) + self.type_emb[2]
        top = torch.cat([Q1, SZ], dim=2)
        bot = torch.cat([SX, Q2], dim=2)
        phi = torch.cat([top, bot], dim=1)
        phi = phi + self.row_emb[None, :, None, :] + self.col_emb[None, None, :, :]
        return self.embed_norm(phi)

    def forward(self, q_feat, s_x, s_z):
        phi = self.embed(q_feat, s_x, s_z)
        masks = (self.mask_A, self.mask_B, self.mask_C, self.mask_D)
        for layer in self.layers:
            phi = layer(phi, self.n1, self.n2, masks)
        return self.norm(phi)

    def qubit_cells(self, grid_vals):
        """grid_vals: (B, R, C, c) -> (B, n, c) in flat qubit order [Q1 row-major | Q2 row-major]."""
        B = grid_vals.shape[0]
        Q1 = grid_vals[:, :self.n1, :self.n2].reshape(B, self.n1 * self.n2, -1)
        Q2 = grid_vals[:, self.n1:, self.n2:].reshape(B, self.r1 * self.r2, -1)
        return torch.cat([Q1, Q2], dim=1)


class ProductQECCT(nn.Module):
    def __init__(self, code, d_model=128, h=8, N_dec=6, d_ff=None, share_processes=True,
                 d_g=32, N_g=1, no_g=False, dropout=0.0, out_mix=True):
        super().__init__()
        assert code.is_hgp(), "ProductQECCT requires an HGP (grid-structured) code"
        d_ff = d_ff or 4 * d_model
        self.code = code
        self.n = code.n
        self.no_g = no_g
        # dense PCMs for violated-check counts (kept as float buffers; exact small integers)
        self.register_buffer("HxT", torch.tensor(code.Hx.T, dtype=torch.float32))
        self.register_buffer("HzT", torch.tensor(code.Hz.T, dtype=torch.float32))
        self.deg_x = float(code.Hx.sum(0).max())
        self.deg_z = float(code.Hz.sum(0).max())
        # factorized noise estimator g_omega: a tiny grid encoder on count features
        if not no_g:
            self.g_enc = GridEncoder(code, d_g, max(1, d_g // 16), N_g, 2 * d_g, q_feat_dim=2,
                                     share=share_processes, dropout=0.0)
            self.g_out = nn.Linear(d_g, 2)
        # main decoder
        self.enc = GridEncoder(code, d_model, h, N_dec, d_ff, q_feat_dim=4, share=share_processes,
                               dropout=dropout)
        self.readout = nn.Linear(d_model, 2)
        self.out_mix = out_mix
        if out_mix:
            R, C = self.enc.R, self.enc.C
            # factorized output mixing  G + A G B^T  (O(R^2 + C^2) params instead of O(n^2)).
            # A, B start at zero so the head is the plain per-cell readout at init and mixing is learned.
            self.A = nn.Parameter(torch.zeros(2, R, R))
            self.Bm = nn.Parameter(torch.zeros(2, C, C))

    def count_features(self, s_x, s_z):
        c_x = (s_x @ self.HxT.T) / self.deg_x   # (B, n): fraction of violated X-checks on each qubit
        c_z = (s_z @ self.HzT.T) / self.deg_z
        return torch.stack([c_x, c_z], dim=-1)  # (B, n, 2)

    def forward(self, s_x, s_z):
        counts = self.count_features(s_x, s_z)
        if self.no_g:
            g_logits = None
            g_feat = torch.zeros_like(counts)
        else:
            phi_g = self.g_enc(counts, s_x, s_z)
            g_cells = self.g_out(self.enc_qubit_cells(self.g_enc, phi_g))  # (B, n, 2)
            g_logits = torch.cat([g_cells[..., 0], g_cells[..., 1]], dim=1)  # (B, 2n)
            g_feat = torch.tanh(g_cells / 4.0)   # bounded feature for the main decoder (logit magnitude keeps growing)
        q_feat = torch.cat([g_feat, counts], dim=-1)   # (B, n, 4) = [g_x, g_z, c_x, c_z]
        phi = self.enc(q_feat, s_x, s_z)
        G = self.readout(phi)                           # (B, R, C, 2)
        if self.out_mix:
            Gm = G.permute(0, 3, 1, 2)                  # (B, 2, R, C)
            Gm = Gm + torch.matmul(torch.matmul(self.A.unsqueeze(0), Gm), self.Bm.transpose(1, 2).unsqueeze(0))
            G = Gm.permute(0, 2, 3, 1)
        cells = self.enc.qubit_cells(G)                 # (B, n, 2)
        logits = torch.cat([cells[..., 0], cells[..., 1]], dim=1)
        return logits, g_logits

    @staticmethod
    def enc_qubit_cells(enc, grid):
        return enc.qubit_cells(grid)
