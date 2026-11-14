"""Encode episode images once with the frozen encoder and cache the latents (training never runs the ViT).

    python -m aaltonomous.baselines.jepa.encode --episodes runs/jepa/episodes --out runs/jepa/latents

Per episode `<name>.pt`: z (N, n_tokens, D) float16, state (N,16), action (N-1,16), meta (sidecar). Plus
`manifest.json` with the encoder config: train.py and the policy refuse latents made with another encoder setup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import ActionSpace, load_config
from .data import list_episodes, load_episode
from .encoder import make_encoder


def encode_episode(enc, space, arrays):
    z = enc.encode({c: arrays[c] for c in enc.cameras}).to(torch.float16).cpu()
    # commanded pose = the state the planner rolls forward with step_state (state + delta)
    state = torch.from_numpy(space.state(arrays["arm_cmd"], arrays["hand_cmd"]))
    action = torch.from_numpy(space.actions_from_commands(arrays["arm_cmd"], arrays["hand_cmd"]))
    return z, state, action


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", default=["runs/jepa/episodes"])
    ap.add_argument("--out", default="runs/jepa/latents")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", default=None, help='json overrides, e.g. \'{"encoder.kind": "tiny"}\'')
    ap.add_argument("--force", action="store_true", help="re-encode episodes that are already cached")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, json.loads(a.set) if a.set else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = out / "manifest.json"
    if man.exists() and json.loads(man.read_text())["encoder"] != cfg["encoder"] and not a.force:
        raise SystemExit(f"{man}: cached with another encoder config; use a new --out or --force")
    enc = make_encoder(cfg["encoder"])
    space = ActionSpace(cfg)
    man.write_text(json.dumps({"encoder": cfg["encoder"], "macro_ticks": cfg["macro_ticks"], "dim": enc.dim,
                               "n_tokens": enc.n_tokens}, indent=1))
    paths = [p for root in a.episodes for p in list_episodes(root)]
    for p in paths:
        dst = out / f"{p.name}.pt"
        if dst.exists() and not a.force:
            continue
        arrays, meta = load_episode(p)
        missing = [c for c in enc.cameras if c not in arrays]
        if missing:
            print(f"[encode] skip {p.name}: no {missing}")
            continue
        if int(meta.get("macro_ticks", cfg["macro_ticks"])) != int(cfg["macro_ticks"]):
            print(f"[encode] skip {p.name}: macro_ticks {meta.get('macro_ticks')} != {cfg['macro_ticks']}")
            continue
        z, s, act = encode_episode(enc, space, arrays)
        torch.save({"z": z, "state": s, "action": act, "meta": meta, "name": p.name}, dst)
        print(f"[encode] {p.name}: {tuple(z.shape)} -> {dst}  (|a| mean {np.abs(act.numpy()[:, :14]).mean():.3f})")


if __name__ == "__main__":
    main()
