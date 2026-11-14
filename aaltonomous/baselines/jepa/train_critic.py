"""Train the grasp critic's probe on *imagined* lift-end latents, with episode-level cross-validation.

    python -m aaltonomous.baselines.jepa.train_critic --picks runs/jepa/grasps/picks \
        --ckpt runs/jepa/ckpt_grasp/best.pt --out runs/jepa/critic.pt

For each labelled pick (collect_grasps.py): z0 = V-JEPA 2(look image), the plan up to its lift end -> macro actions,
predictor rollout -> ẑ_lift. The probe (grasp.HeldProbe) learns ẑ_lift -> held. Reported per fold (grouped by
episode, so no layout is in train and test):
  * probe AUC / accuracy on imagined latents (what the policy uses),
  * the non-JEPA baseline: each variant's mean success rate in the training folds,
  * and, as an upper bound, the same probe on the *real* encoded frame at the lift end.
The final probe is refit on all picks and saved with the predictor/encoder references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import ActionSpace, load_config
from .data import load_episode
from .encoder import make_encoder
from .grasp import GraspCritic, HeldProbe, traj_actions
from .predictor import load_checkpoint


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    r = s.argsort().argsort() + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def fit_probe(Z, y, dim, device, epochs=300, lr=1e-3, wd=1e-2, seed=0):
    torch.manual_seed(seed)
    p = HeldProbe(dim).to(device)
    opt = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=wd)
    Z, yt = Z.to(device), torch.as_tensor(y, dtype=torch.float32, device=device)
    pos = max(float(yt.mean()), 1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    for _ in range(epochs):
        idx = torch.randperm(len(Z), device=device)[:64]
        z = Z[idx] + 0.05 * torch.randn_like(Z[idx])           # light noise: robust to rollout drift
        loss = lossf(p(z), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return p.eval()


@torch.no_grad()
def predict(p, Z, device):
    return torch.sigmoid(p(Z.to(device))).cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", default="runs/jepa/grasps/picks")
    ap.add_argument("--frames-latents", default=None, help="latents of the episodes' frames (encode.py) for the real-"
                    "latent upper bound; default <picks>/../latents")
    ap.add_argument("--ckpt", default="runs/jepa/ckpt_grasp/best.pt")
    ap.add_argument("--out", default="runs/jepa/critic.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    model, ck = load_checkpoint(a.ckpt, a.device)
    cfg = load_config()
    cfg["encoder"], cfg["macro_ticks"] = ck["encoder"], ck["macro_ticks"]
    enc = make_encoder(cfg["encoder"], a.device)
    space = ActionSpace(cfg)
    critic = GraspCritic(model, None, enc, space, a.device)
    lat_dir = Path(a.frames_latents or Path(a.picks).parent / "latents")

    Zi, Zr, y, ep, names = [], [], [], [], []
    for f in sorted(Path(a.picks).glob("*.npz")):
        arrays, meta = load_episode(f.with_suffix(""))
        fl = lat_dir / f"{f.stem}.pt"
        frames = torch.load(fl, weights_only=False) if fl.exists() else None
        for pk in meta["picks"]:
            i = pk["i"]
            z0 = critic.encode({"head_rgb": arrays[f"img_{i}"]})
            s0 = space.state(arrays[f"arm0_{i}"], arrays[f"hand0_{i}"])
            acts = traj_actions(space, arrays[f"arm_{i}"], arrays[f"hand_{i}"], arrays[f"arm0_{i}"], arrays[f"hand0_{i}"])
            Zi.append(critic.imagine(z0, s0, [acts])[0].cpu())
            if frames is not None and "tick0" in pk:
                j = min((pk["tick0"] + pk["lift_end"]) // space.k, len(frames["z"]) - 1)
                Zr.append(frames["z"][j].float())
            y.append(bool(pk["held"]))
            ep.append(f.stem)
            names.append(pk["variant"]["name"])
    if not y:
        raise SystemExit(f"no picks in {a.picks}")
    Zi = torch.stack(Zi)
    y = np.array(y)
    eps = sorted(set(ep))
    print(f"[critic] {len(y)} picks from {len(eps)} episodes, held rate {y.mean():.2f}")
    rng = np.random.default_rng(0)
    fold_of = {e: i % a.folds for i, e in enumerate(rng.permutation(eps))}
    folds = np.array([fold_of[e] for e in ep])
    s_img, s_prior, s_real = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    use_real = len(Zr) == len(y)
    Zr = torch.stack(Zr) if use_real else None
    for k in range(a.folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        p = fit_probe(Zi[tr], y[tr], Zi.shape[-1], a.device)
        s_img[te] = predict(p, Zi[te], a.device)
        rate = {n: y[tr][np.array(names)[tr] == n].mean() for n in set(np.array(names)[tr])}
        s_prior[te] = [rate.get(n, y[tr].mean()) for n in np.array(names)[te]]
        if use_real:
            pr = fit_probe(Zr[tr], y[tr], Zr.shape[-1], a.device)
            s_real[te] = predict(pr, Zr[te], a.device)
    rep = {"n": int(len(y)), "held_rate": float(y.mean()),
           "auc_imagined": auc(y, s_img), "acc_imagined": float(((s_img > 0.5) == y).mean()),
           "auc_variant_prior": auc(y, s_prior),
           "auc_real_lift_frame": auc(y, s_real) if use_real else None}
    print("[critic] cross-val:", json.dumps(rep))
    p = fit_probe(Zi, y, Zi.shape[-1], a.device)
    torch.save({"probe": p.state_dict(), "dim": Zi.shape[-1], "ckpt": str(a.ckpt), "report": rep}, a.out)
    print(f"[critic] saved {a.out}")
    return rep


def load_critic(critic_path, device=None):
    """-> GraspCritic (encoder + predictor + probe) from a train_critic.py file."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cp = torch.load(critic_path, map_location=device, weights_only=False)
    model, ck = load_checkpoint(cp["ckpt"], device)
    cfg = load_config()
    cfg["encoder"], cfg["macro_ticks"] = ck["encoder"], ck["macro_ticks"]
    probe = HeldProbe(cp["dim"]).to(device).eval()
    probe.load_state_dict(cp["probe"])
    return GraspCritic(model, probe, make_encoder(cfg["encoder"], device), ActionSpace(cfg), device)


if __name__ == "__main__":
    main()
