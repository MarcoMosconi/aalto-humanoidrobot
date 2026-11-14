"""Sub-goals for the latent planner, retrieved from a bank of demo latents.

V-JEPA 2-AC plans towards goal *images*. A sorting episode has no single goal image the robot can reach in one short
plan, so the goal is a moving sub-goal taken from demonstrations:

    1. match the current (tokens, joint state) to its nearest demo frame (cosine of mean-pooled tokens + proprio),
    2. the goal is the frame `lookahead` macro steps later in that same demo.

The bank is the latent cache (encode.py) of successful episodes. On rung 2 the bank prefers demos with the same
colour -> bin mapping as the instruction (parse_mapping). `done` is reported when the nearest demo frame has been
near the end of its demo for a few looks in a row.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aaltonomous.interface.tasks import BINS, COLORS


def parse_mapping(text):
    """'Put the red objects in the left bin, the yellow ...' -> {'red': 'left', ...}; None if not every colour has
    exactly one bin. Handles the training templates in interface/tasks.py (one colour + one bin per clause)."""
    if not text:
        return None
    out = {}
    for clause in re.split(r"[,.;:]| and ", text.lower()):
        cs = [c for c in COLORS if re.search(rf"\b{c}\b", clause)]
        bs = [b for b in BINS if re.search(rf"\b{b}\b", clause)]
        if len(cs) == 1 and len(bs) == 1:
            out[cs[0]] = bs[0]
    return out if set(out) == set(COLORS) else None


class GoalBank:
    def __init__(self, cfg, latents_dir=None, episodes=None, device="cpu"):
        gc = cfg["goals"]
        self.cfg, self.device = gc, torch.device(device)
        if episodes is None:
            episodes = [torch.load(p, weights_only=False) for p in sorted(Path(latents_dir).glob("*.pt"))]
        if gc.get("only_success", True):
            ok = [e for e in episodes if e["meta"].get("success") is not False]
            episodes = ok or episodes
        z, keys, st, ep, pos, frac = [], [], [], [], [], []
        self.mappings = []
        stride = max(1, int(np.ceil(sum(len(e["z"]) for e in episodes) / gc["max_bank_frames"])))
        for i, e in enumerate(episodes):
            n = len(e["z"])
            idx = np.arange(0, n, stride)
            z.append(e["z"])                                   # full episode kept for the goal lookup
            keys.append(F.normalize(e["z"][idx].float().mean(1), dim=-1))
            st.append(e["state"][idx].float())
            ep += [i] * len(idx)
            pos += list(idx)
            frac += list(idx / max(n - 1, 1))
            self.mappings.append(e["meta"].get("mapping"))
        if not z:
            raise ValueError("goal bank is empty")
        self.z = z
        self.keys = torch.cat(keys).to(self.device)
        self.state = torch.cat(st).to(self.device)
        self.ep = torch.tensor(ep, device=self.device)
        self.pos = torch.tensor(pos, device=self.device)
        self.frac = torch.tensor(frac, device=self.device, dtype=torch.float32)
        self.reset()

    def __len__(self):
        return len(self.keys)

    def reset(self, instruction=None):
        self.n_done = 0
        mp = parse_mapping(instruction) if self.cfg.get("match_instruction", True) else None
        allow = torch.ones(len(self.mappings), dtype=torch.bool)
        if mp is not None:
            same = torch.tensor([m == mp for m in self.mappings])
            if same.any():
                allow = same
        self.allow = allow.to(self.device)[self.ep]
        self.last = None

    @torch.no_grad()
    def goal(self, z_now, s_now):
        """z_now (N,D), s_now (16,) -> (goal tokens (N,D), info dict). info['done'] once the demo is (nearly) over."""
        q = F.normalize(z_now.float().mean(0), dim=-1).to(self.device)
        sim = self.keys @ q
        ds = (self.state[:, :14] - s_now[:14].to(self.device)).norm(dim=-1)
        score = sim - self.cfg["proprio_weight"] * ds
        score = torch.where(self.allow, score, torch.full_like(score, -1e9))
        j = int(score.argmax())
        e, p = int(self.ep[j]), int(self.pos[j])
        zz = self.z[e]
        g = min(p + int(self.cfg["lookahead"]), len(zz) - 1)
        near_end = float(self.frac[j]) >= self.cfg["done_frac"]
        self.n_done = self.n_done + 1 if near_end else 0
        self.last = {"episode": e, "frame": p, "goal_frame": g, "sim": float(sim[j]), "dq": float(ds[j]),
                     "frac": float(self.frac[j]), "done": self.n_done >= int(self.cfg["done_patience"])}
        return zz[g].float(), self.last
