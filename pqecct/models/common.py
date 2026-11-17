"""
Shared building blocks: multi-head (cross-)attention, FFN, pre-norm residual blocks,
differentiable GF(2) parity (DQEC's logical-error loss), losses and metrics.

Model interface (all decoders):
    logits, g_logits = model(s_x, s_z)
        s_x: (B, rx) float {0,1}   X-check syndrome (detects Z errors)
        s_z: (B, rz) float {0,1}   Z-check syndrome (detects X errors)
        logits:   (B, 2n)  [logit(e_x) | logit(e_z)]   (positive = error)
        g_logits: (B, 2n)  initial noise estimate (or None)
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_VALUE = -1e9


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    """Standard multi-head attention; `mask` is a boolean tensor broadcastable to
    (B, h, Lq, Lk) with True = *blocked* (same convention as ECCT/CrossMPT code)."""

    def __init__(self, h, d_model, dropout=0.0, softcap=None):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.softcap = softcap   # attention-logit soft-capping: s <- c * tanh(s / c); bounds logits, prevents collapse

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        q, k, v = [l(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        if self.softcap is None:
            # memory-efficient fused kernel: identical math, never materializes the (Lq x Lk) score
            # matrix on GPU (needed for QECCT on HGP codes with ~2000 tokens). Bool mask: True = keep.
            drop = self.dropout.p if (self.dropout is not None and self.training) else 0.0
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else ~mask, dropout_p=drop)
            x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
            return self.linears[-1](x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.softcap:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        if mask is not None:
            scores = scores.masked_fill(mask, MASK_VALUE)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Pre-norm residual: x + sublayer(norm(x))."""

    def __init__(self, size, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class CrossAttentionBlock(nn.Module):
    """Masked cross-attention block (CrossMPT / ProductMPT 'cross-attention module'):
        x  <- x + Attn(norm(x) -> norm_c(ctx))   (query from target, key/value from context)
        x  <- x + FFN(norm(x))
    The context is LayerNorm'd before the K/V projections.  Without it (as in the original
    CrossMPT code) K/V scale with the un-normalized residual stream of the other token set, and the
    two streams feed each other: in the DQEC setting this loop blows up and the model collapses to
    a constant output (observed on toric L6/L8 for both CrossQECCT and ProductQECCT).
    """

    def __init__(self, d_model, h, d_ff, dropout=0.0, softcap=None):
        super().__init__()
        self.attn = MultiHeadedAttention(h, d_model, dropout, softcap)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub = clones(SublayerConnection(d_model, dropout), 2)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.size = d_model

    def forward(self, x, ctx, mask):
        c = self.ctx_norm(ctx)
        x = self.sub[0](x, lambda t: self.attn(t, c, c, mask))
        return self.sub[1](x, self.ff)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadedAttention(h, d_model, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sub = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        x = self.sub[0](x, lambda t: self.attn(t, t, t, mask))
        return self.sub[1](x, self.ff)


# ----------------------------------------------------------------------------
# Differentiable GF(2) products (DQEC Eq. 9/10)
# ----------------------------------------------------------------------------
def soft_parity(LT, x_prob):
    """Soft version of (x @ L^T) mod 2.

    LT: (n, k) float {0,1} (transposed logical matrix); x_prob: (B, n) in [0,1].
    Returns (B, k) probabilities that each logical is flipped, computed via
        phi^{-1}( prod_j phi(L_ij * x_j) ),   phi(u) = 1 - 2u.
    (For logicals of very large weight the product can underflow to 0, i.e. parity 1/2 with
    zero gradient, which is harmless; the --ler_warmup_epochs option covers the large codes.)
    """
    t = 1.0 - 2.0 * (x_prob.unsqueeze(-1) * LT.unsqueeze(0))      # (B, n, k): 1 - 2 L_ij x_j
    prod = torch.prod(t, dim=1)
    return 0.5 * (1.0 - prod)


def hard_parity(LT, x_bin):
    return torch.remainder(x_bin @ LT, 2.0)


# ----------------------------------------------------------------------------
# Loss and metrics
# ----------------------------------------------------------------------------
def decoder_loss(logits, g_logits, e_x, e_z, sampler, err_types, lam_ber=0.5, lam_ler=1.0, lam_g=0.5):
    """DQEC objective: lam_ber * BCE(e) + lam_ler * BCE(soft logical parity) + lam_g * BCE(g, e)."""
    n = e_x.shape[1]
    lx, lz = logits[:, :n], logits[:, n:]
    loss_ber = 0.0
    loss_ler = 0.0
    loss_g = 0.0
    cnt = 0
    if "x" in err_types:
        loss_ber = loss_ber + F.binary_cross_entropy_with_logits(lx, e_x)
        pl = soft_parity(sampler.LzT, torch.sigmoid(lx)).clamp(1e-6, 1 - 1e-6)
        loss_ler = loss_ler + F.binary_cross_entropy(pl, hard_parity(sampler.LzT, e_x))
        if g_logits is not None:
            loss_g = loss_g + F.binary_cross_entropy_with_logits(g_logits[:, :n], e_x)
        cnt += 1
    if "z" in err_types:
        loss_ber = loss_ber + F.binary_cross_entropy_with_logits(lz, e_z)
        pl = soft_parity(sampler.LxT, torch.sigmoid(lz)).clamp(1e-6, 1 - 1e-6)
        loss_ler = loss_ler + F.binary_cross_entropy(pl, hard_parity(sampler.LxT, e_z))
        if g_logits is not None:
            loss_g = loss_g + F.binary_cross_entropy_with_logits(g_logits[:, n:], e_z)
        cnt += 1
    loss_ber = loss_ber / cnt
    loss_ler = loss_ler / cnt
    loss_g = loss_g / cnt if g_logits is not None else torch.zeros((), device=logits.device)
    total = lam_ber * loss_ber + lam_ler * loss_ler + lam_g * loss_g
    parts = dict(ber=loss_ber.detach().item(), ler=loss_ler.detach().item(), g=loss_g.detach().item())
    return total, parts


@torch.no_grad()
def decode_metrics(logits, e_x, e_z, s_x, s_z, sampler, err_types):
    """Returns per-batch counts needed for BER / LER statistics.

    LER (DQEC-style): a logical operator of the predicted error differs from the truth
                      (equivalently the residual error acts as a nontrivial logical).
    LER_strict     : logical flip OR residual syndrome nonzero (prediction inconsistent with
                      the measured syndrome). For syndrome-matching decoders (BP+OSD, MWPM)
                      the two coincide. This is the primary metric of the proposal.
    """
    n = e_x.shape[1]
    B = e_x.shape[0]
    ex_hat = (logits[:, :n] > 0).float()
    ez_hat = (logits[:, n:] > 0).float()
    out = dict(samples=B)
    fail_any = torch.zeros(B, dtype=torch.bool, device=logits.device)
    fail_strict_any = torch.zeros_like(fail_any)
    bit_err = 0.0
    nbits = 0
    if "x" in err_types:
        rx = ex_hat != e_x
        bit_err += rx.float().sum().item(); nbits += n
        lflip = torch.remainder((ex_hat + e_x) @ sampler.LzT, 2.0).any(1)
        sres = torch.remainder((ex_hat + e_x) @ sampler.HzT, 2.0).any(1)
        out["ler_x"] = lflip.float().sum().item()
        out["ler_x_strict"] = (lflip | sres).float().sum().item()
        out["synd_mismatch_x"] = sres.float().sum().item()
        fail_any |= lflip
        fail_strict_any |= (lflip | sres)
    if "z" in err_types:
        rz = ez_hat != e_z
        bit_err += rz.float().sum().item(); nbits += n
        lflip = torch.remainder((ez_hat + e_z) @ sampler.LxT, 2.0).any(1)
        sres = torch.remainder((ez_hat + e_z) @ sampler.HxT, 2.0).any(1)
        out["ler_z"] = lflip.float().sum().item()
        out["ler_z_strict"] = (lflip | sres).float().sum().item()
        out["synd_mismatch_z"] = sres.float().sum().item()
        fail_any |= lflip
        fail_strict_any |= (lflip | sres)
    out["ler"] = fail_any.float().sum().item()
    out["ler_strict"] = fail_strict_any.float().sum().item()
    out["bit_errors"] = bit_err
    out["bits"] = nbits * B
    return out


def accumulate(acc, part):
    for k, v in part.items():
        acc[k] = acc.get(k, 0.0) + v
    return acc


def finalize(acc):
    S = max(acc.get("samples", 0), 1)
    res = {}
    for k, v in acc.items():
        if k in ("samples", "bits", "bit_errors"):
            continue
        res[k] = v / S
    res["ber"] = acc.get("bit_errors", 0.0) / max(acc.get("bits", 1), 1)
    res["samples"] = int(S)
    return res
