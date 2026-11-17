"""
Noise models and syndrome computation (batched, torch, device-resident).

Error vectors are represented as two binary vectors e_x, e_z in {0,1}^n:
    e_x[q] = 1  <=> an X (or Y) error on qubit q      -> detected by Z-checks:  s_z = H_Z e_x
    e_z[q] = 1  <=> a  Z (or Y) error on qubit q      -> detected by X-checks:  s_x = H_X e_z

Noise models
  independent  : X and Z flips i.i.d. with probability p each (as in DQEC / Krastanov & Jiang)
  depolarizing : X, Y, Z each with probability p/3

`err_types` restricts which error types are *generated and decoded*:
  'x'  : only e_x (classical-like single-type decoding problem; the usual way DQEC reports
         "independent noise" numbers), 'z' : only e_z, 'xz' : both jointly.
"""
import numpy as np
import torch


class NoiseSampler:
    def __init__(self, code, noise_model="depolarizing", err_types="xz", device="cpu"):
        assert noise_model in ("independent", "depolarizing")
        assert err_types in ("x", "z", "xz")
        self.code = code
        self.noise_model = noise_model
        self.err_types = err_types
        self.device = device
        self.n = code.n
        # dense float parity-check matrices for exact integer matmul on GPU
        self.HxT = torch.tensor(code.Hx.T, dtype=torch.float32, device=device)  # n x rx
        self.HzT = torch.tensor(code.Hz.T, dtype=torch.float32, device=device)  # n x rz
        self.LxT = torch.tensor(code.Lx.T, dtype=torch.float32, device=device)  # n x k
        self.LzT = torch.tensor(code.Lz.T, dtype=torch.float32, device=device)  # n x k

    # ------------------------------------------------------------------
    def sample_errors(self, p, batch_size, generator=None):
        """p: float or tensor (batch_size,) of physical error rates. Returns e_x, e_z (B, n) float {0,1}."""
        if not torch.is_tensor(p):
            p = torch.full((batch_size,), float(p), device=self.device)
        p = p.to(self.device).view(-1, 1)
        u = torch.rand(batch_size, self.n, device=self.device, generator=generator)
        if self.noise_model == "independent":
            e_x = (u < p).float()
            u2 = torch.rand(batch_size, self.n, device=self.device, generator=generator)
            e_z = (u2 < p).float()
        else:  # depolarizing: X w.p. p/3, Y w.p. p/3, Z w.p. p/3
            q = p / 3.0
            x_only = (u < q)
            y_both = (u >= q) & (u < 2 * q)
            z_only = (u >= 2 * q) & (u < 3 * q)
            e_x = (x_only | y_both).float()
            e_z = (z_only | y_both).float()
        if self.err_types == "x":
            e_z = torch.zeros_like(e_z)
        elif self.err_types == "z":
            e_x = torch.zeros_like(e_x)
        return e_x, e_z

    def syndromes(self, e_x, e_z):
        """s_z = H_Z e_x (B, rz), s_x = H_X e_z (B, rx), as float {0,1}."""
        s_z = torch.remainder(e_x @ self.HzT, 2.0)
        s_x = torch.remainder(e_z @ self.HxT, 2.0)
        return s_x, s_z

    def logical_flips(self, e_x, e_z):
        """Lz e_x (B,k) and Lx e_z (B,k): which logical operators the error flips."""
        lz = torch.remainder(e_x @ self.LzT, 2.0)
        lx = torch.remainder(e_z @ self.LxT, 2.0)
        return lz, lx

    def sample_batch(self, p, batch_size, generator=None):
        e_x, e_z = self.sample_errors(p, batch_size, generator)
        s_x, s_z = self.syndromes(e_x, e_z)
        return e_x, e_z, s_x, s_z

    def sample_batch_numpy(self, p, batch_size, rng):
        """Numpy version (same distribution) for classical decoders. Returns uint8 arrays."""
        n = self.n
        u = rng.random((batch_size, n))
        if self.noise_model == "independent":
            e_x = (u < p)
            e_z = (rng.random((batch_size, n)) < p)
        else:
            q = p / 3.0
            e_x = (u < q) | ((u >= q) & (u < 2 * q))
            e_z = ((u >= q) & (u < 2 * q)) | ((u >= 2 * q) & (u < 3 * q))
        e_x = e_x.astype(np.uint8); e_z = e_z.astype(np.uint8)
        if self.err_types == "x":
            e_z[:] = 0
        elif self.err_types == "z":
            e_x[:] = 0
        s_z = (e_x.astype(np.int64) @ self.code.Hz.T.astype(np.int64)) % 2
        s_x = (e_z.astype(np.int64) @ self.code.Hx.T.astype(np.int64)) % 2
        return e_x, e_z, s_x.astype(np.uint8), s_z.astype(np.uint8)
