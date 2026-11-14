"""Action-conditioned latent predictor (V-JEPA 2-AC style).

Per macro step t the sequence holds [action token, state token, N frame tokens]. Attention is block-causal over time
(step t sees steps <= t), and the outputs at the frame tokens of step t predict the frame tokens of step t+1:

    z_hat[:, t] = P(z[:, :t+1], s[:, :t+1], a[:, :t+1])  ~  z[:, t+1]

Training (train.py): teacher forcing over a window + a short autoregressive rollout, L1 in latent space.
Planning (planner.py): roll out from the current frame for candidate action sequences, compare with a goal latent.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .common import A_DIM


class ACPredictor(nn.Module):
    def __init__(self, z_dim, n_tokens, dim=384, depth=6, heads=6, max_steps=8, dropout=0.0, a_dim=A_DIM,
                 s_dim=A_DIM):
        super().__init__()
        self.z_dim, self.n_tokens, self.max_steps = z_dim, n_tokens, max_steps
        self.hparams = dict(z_dim=z_dim, n_tokens=n_tokens, dim=dim, depth=depth, heads=heads, max_steps=max_steps,
                            dropout=dropout, a_dim=a_dim, s_dim=s_dim)
        self.z_in = nn.Linear(z_dim, dim)
        self.a_in = nn.Linear(a_dim, dim)
        self.s_in = nn.Linear(s_dim, dim)
        self.pos_space = nn.Parameter(torch.zeros(1, 1, n_tokens + 2, dim))
        self.pos_time = nn.Parameter(torch.zeros(1, max_steps, 1, dim))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, z_dim)
        self._masks = {}

    def _mask(self, T, device):
        key = (T, device)
        if key not in self._masks:
            per = self.n_tokens + 2
            step = torch.arange(T, device=device).repeat_interleave(per)
            self._masks[key] = step[None, :] > step[:, None]  # True = blocked (future step)
        return self._masks[key]

    def forward(self, z, s, a):
        """z (B,T,N,Dz), s (B,T,16), a (B,T,16) -> (B,T,N,Dz) predictions of the next step's tokens."""
        B, T, N, _ = z.shape
        if T > self.max_steps:
            raise ValueError(f"sequence of {T} steps > max_steps {self.max_steps}")
        x = torch.cat([self.a_in(a)[:, :, None], self.s_in(s)[:, :, None], self.z_in(z)], 2)  # (B,T,N+2,D)
        x = x + self.pos_space + self.pos_time[:, :T]
        x = self.blocks(x.flatten(1, 2), mask=self._mask(T, z.device))
        x = self.norm(x).view(B, T, N + 2, -1)[:, :, 2:]
        return self.out(x)

    @torch.no_grad()
    def rollout(self, z0, s0, actions, step_state):
        """Autoregressive rollout. z0 (B,N,D) or (B,T0,N,D) context, s0 matching, actions (B,H,16).
        step_state(s, a) -> next state. Returns predicted tokens (B,H,N,D)."""
        z = z0[:, None] if z0.dim() == 3 else z0
        s = s0[:, None] if s0.dim() == 2 else s0
        T0 = z.shape[1]
        a_hist = torch.zeros(z.shape[0], T0 - 1, actions.shape[-1], device=z.device, dtype=z.dtype)
        preds = []
        for h in range(actions.shape[1]):
            a = torch.cat([a_hist, actions[:, : h + 1]], 1)
            zz, ss, aa = z[:, -self.max_steps:], s[:, -self.max_steps:], a[:, -self.max_steps:]
            p = self.forward(zz, ss, aa)[:, -1]
            preds.append(p)
            z = torch.cat([z, p[:, None]], 1)
            s = torch.cat([s, step_state(s[:, -1], actions[:, h])[:, None]], 1)
        return torch.stack(preds, 1)


def step_state(s, a):
    """Planning-space state update: arm += delta, grip := commanded grip."""
    return torch.cat([s[..., :14] + a[..., :14], a[..., 14:]], -1)


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), "cfg": cfg, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ACPredictor(**ck["hparams"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck
