"""Train the action-conditioned predictor on cached latents (encode.py).

    python -m aaltonomous.baselines.jepa.train --latents runs/jepa/latents --out runs/jepa/ckpt

Loss (V-JEPA 2-AC): L1 between predicted and encoded next-step tokens, (1) teacher forcing over a window of
`predictor.context` steps and (2) an autoregressive rollout of `train.rollout_steps` steps from the first frame.
Episodes are split train/val by episode. Checkpoints: `<out>/last.pt`, `<out>/best.pt` (lowest val loss).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_config
from .predictor import ACPredictor, save_checkpoint, step_state


class Windows(torch.utils.data.Dataset):
    """Windows of T+1 consecutive samples: z (T+1,N,D), state (T+1,16), action (T,16)."""

    def __init__(self, eps, T):
        self.eps, self.T = eps, T
        self.index = [(i, s) for i, e in enumerate(eps) for s in range(len(e["z"]) - T)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, s = self.index[j]
        e, T = self.eps[i], self.T
        return e["z"][s:s + T + 1].float(), e["state"][s:s + T + 1], e["action"][s:s + T]


def load_latents(root):
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    eps = [torch.load(p, weights_only=False) for p in sorted(root.glob("*.pt"))]
    return man, eps


def losses(model, z, s, a, rollout_steps):
    pred = model(z[:, :-1], s[:, :-1], a)
    l_tf = (pred - z[:, 1:]).abs().mean()
    l_ro = torch.zeros((), device=z.device)
    if rollout_steps > 0:
        zz, ss = z[:, :1], s[:, :1]
        for h in range(min(rollout_steps, a.shape[1])):
            p = model(zz, ss, a[:, : h + 1])[:, -1]
            l_ro = l_ro + (p - z[:, h + 1]).abs().mean()
            zz = torch.cat([zz, p[:, None]], 1)
            ss = torch.cat([ss, step_state(ss[:, -1], a[:, h])[:, None]], 1)
        l_ro = l_ro / min(rollout_steps, a.shape[1])
    return l_tf, l_ro


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latents", default="runs/jepa/latents")
    ap.add_argument("--out", default="runs/jepa/ckpt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", default=None, help='json overrides, e.g. \'{"train.steps": 2000}\'')
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, json.loads(a.set) if a.set else None)
    tc, pc = cfg["train"], cfg["predictor"]
    torch.manual_seed(a.seed)
    man, eps = load_latents(a.latents)
    T = int(pc["context"])
    eps = [e for e in eps if len(e["z"]) > T]
    if not eps:
        raise SystemExit(f"no episode in {a.latents} longer than {T} samples")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(eps))
    n_val = int(round(tc["val_frac"] * len(eps))) if len(eps) > 1 else 0
    val_eps, tr_eps = [eps[i] for i in order[:n_val]], [eps[i] for i in order[n_val:]]
    tr, va = Windows(tr_eps, T), Windows(val_eps, T) if val_eps else None
    print(f"[train] {len(tr_eps)} train / {len(val_eps)} val episodes, {len(tr)} windows, tokens {man['n_tokens']}x{man['dim']}")
    dl = torch.utils.data.DataLoader(tr, batch_size=tc["batch"], shuffle=True, drop_last=len(tr) >= tc["batch"],
                                     num_workers=tc["num_workers"], persistent_workers=tc["num_workers"] > 0)
    model = ACPredictor(man["dim"], man["n_tokens"], pc["dim"], pc["depth"], pc["heads"],
                        max_steps=max(T, cfg["planner"]["horizon"]) + 1, dropout=pc["dropout"]).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps, warm = int(tc["steps"]), int(tc["warmup"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = {"encoder": man["encoder"], "macro_ticks": man["macro_ticks"]}
    best, step, t0 = float("inf"), 0, time.monotonic()
    use_amp = a.device.startswith("cuda")
    while step < steps:
        for z, s, act in dl:
            z, s, act = z.to(a.device), s.to(a.device), act.to(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                l_tf, l_ro = losses(model, z, s, act, tc["rollout_steps"])
                loss = l_tf + tc["rollout_weight"] * l_ro
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % tc["log_every"] == 0 or step == steps:
                msg = f"[train] step {step}/{steps} tf {l_tf.item():.4f} rollout {l_ro.item():.4f} ({time.monotonic() - t0:.0f} s)"
                if va is not None and len(va):
                    v = evaluate(model, va, tc, a.device)
                    msg += f" | val tf {v[0]:.4f} rollout {v[1]:.4f}"
                    if v[0] + v[1] < best:
                        best = v[0] + v[1]
                        save_checkpoint(out / "best.pt", model, cfg, dict(extra, step=step, val=v))
                print(msg, flush=True)
            if step % tc["save_every"] == 0 or step == steps:
                save_checkpoint(out / "last.pt", model, cfg, dict(extra, step=step))
            if step >= steps:
                break
    if not (out / "best.pt").exists():
        save_checkpoint(out / "best.pt", model, cfg, dict(extra, step=step))
    print(f"[train] done: {out}/best.pt")


@torch.no_grad()
def evaluate(model, ds, tc, device, max_batches=20):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=tc["batch"], shuffle=False)
    tf = ro = n = 0
    for i, (z, s, a) in enumerate(dl):
        if i >= max_batches:
            break
        l_tf, l_ro = losses(model, z.to(device), s.to(device), a.to(device), tc["rollout_steps"])
        tf, ro, n = tf + l_tf.item(), ro + l_ro.item(), n + 1
    model.train()
    return tf / max(n, 1), ro / max(n, 1)


if __name__ == "__main__":
    main()
