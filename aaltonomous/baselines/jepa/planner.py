"""Cross-entropy-method MPC in latent space (V-JEPA 2-AC planning).

Energy of an action sequence a_1..a_H from the current tokens z0 and state s0:

    E(a) = | P_rollout(z0, s0, a)[H] - z_goal |_1   (+ every step if planner.energy_all_steps)
           + smooth_weight * sum |delta arm|

CEM: Gaussian over (H,16) actions, `samples` draws per iteration, refit to the `elites` best, `iters` times; the
mean of the final distribution is returned. Arm deltas are bounded per step (ActionSpace.max_step) and the joint
limits (with margin) are respected along the whole sequence; grips stay in [0,1]. Only the first step is executed.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import ActionSpace
from .predictor import ACPredictor, step_state


class CEMPlanner:
    def __init__(self, model: ACPredictor, space: ActionSpace, cfg, device):
        self.model, self.space, self.pc = model, space, cfg["planner"]
        self.device = torch.device(device)
        self.lo = torch.as_tensor(space.arm_lo, device=self.device, dtype=torch.float32)
        self.hi = torch.as_tensor(space.arm_hi, device=self.device, dtype=torch.float32)
        self.prev = None   # warm start: last plan shifted by one step

    def reset(self):
        self.prev = None

    def _bound(self, a, s0):
        """Clamp a (S,H,16) in place: per-step arm bound, cumulative joint limits, grip in [0,1]."""
        m = self.space.max_step
        a[..., :14].clamp_(-m, m)
        arm = s0[:14].expand(a.shape[0], 14).clone()
        for h in range(a.shape[1]):
            nxt = torch.minimum(torch.maximum(arm + a[:, h, :14], self.lo), self.hi)
            a[:, h, :14] = nxt - arm
            arm = nxt
        a[..., 14:].clamp_(0.0, 1.0)
        return a

    @torch.no_grad()
    def plan(self, z0, s0, z_goal):
        """z0 (N,D), s0 (16,), z_goal (N,D) -> (first action (16,) numpy, info)."""
        pc, dev = self.pc, self.device
        H, S, K = int(pc["horizon"]), int(pc["samples"]), int(pc["elites"])
        z0 = z0.to(dev).float()
        s0 = torch.as_tensor(s0, device=dev, dtype=torch.float32)
        zg = z_goal.to(dev).float()
        m = self.space.max_step
        mu = torch.zeros(H, 16, device=dev)
        mu[:, 14:] = s0[14:]                                  # keep the current grip by default
        if self.prev is not None:
            mu[:-1] = self.prev[1:]
            mu[-1] = self.prev[-1]
        std = torch.full((H, 16), float(pc["init_std"]) * m, device=dev)
        std[:, 14:] = float(pc["grip_std"])
        best_e = None
        for _ in range(int(pc["iters"])):
            a = mu + std * torch.randn(S, H, 16, device=dev)
            a[0] = mu                                         # always score the current mean
            a = self._bound(a, s0)
            pred = self.model.rollout(z0.expand(S, *z0.shape), s0.expand(S, 16), a, step_state)  # (S,H,N,D)
            if pc["energy_all_steps"]:
                e = (pred - zg).abs().mean((2, 3)).sum(1)
            else:
                e = (pred[:, -1] - zg).abs().mean((1, 2))
            e = e + float(pc["smooth_weight"]) * a[..., :14].abs().sum((1, 2)) / max(m, 1e-6)
            idx = e.topk(K, largest=False).indices
            el = a[idx]
            mu = (1 - pc["momentum"]) * el.mean(0) + pc["momentum"] * mu
            std = (1 - pc["momentum"]) * el.std(0) + pc["momentum"] * std
            best_e = float(e[idx[0]])
        mu = self._bound(mu[None].clone(), s0)[0]
        self.prev = mu
        return mu[0].cpu().numpy().astype(np.float64), {"energy": best_e}
