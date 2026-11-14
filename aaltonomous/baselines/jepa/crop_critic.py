"""Object-centred JEPA grasp critic: the world model works on V-JEPA 2 latents of a crop around the grasp.

Why: on the full head frame (8x8 tokens) whether an object is held after the lift is barely readable (AUC ~0.68
even on the *real* lift-end frame), on a crop around the hand it is (AUC ~0.90). So per pick the critic uses a fixed
crop window centred between the grasp point and the lift point (FK of the planned trajectory, projected with the
head calibration), and:

    build:  for every collected pick, crop its macro-step frames tick0 .. tick0+lift_end -> V-JEPA 2 -> latents,
            saved per pick in encode.py's format (+ manifest) so train.py trains the predictor on them unchanged
    probe:  predictor rollout of the plan's macro actions from the tick0 crop -> imagined lift-end latent ->
            standardised mean token -> logistic regression -> p(held); cross-validated by episode
    policy: CropGraspCritic.score_plans(obs, arm, hand, plans) -> p(held) per candidate (grasp.JepaGraspPolicy)

    python -m aaltonomous.baselines.jepa.crop_critic build --grasps runs/jepa/grasps --out runs/jepa/crop_latents
    python -m aaltonomous.baselines.jepa.train --latents runs/jepa/crop_latents --out runs/jepa/ckpt_crop --set '{...}'
    python -m aaltonomous.baselines.jepa.crop_critic probe --grasps runs/jepa/grasps --latents runs/jepa/crop_latents \
        --ckpt runs/jepa/ckpt_crop/best.pt --out runs/jepa/crop_critic.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aaltonomous.interface.calibration import Calibration

from .common import ActionSpace, load_config
from .data import load_episode
from .encoder import make_encoder
from .grasp import lift_end, traj_actions
from .predictor import load_checkpoint, step_state

FRAME_HW = (240, 320)   # frames are stored (and runtime looks are downscaled) at this size
HALF_FULL = 120         # crop half size in full-resolution (480x640) pixels


def _kin():
    from aaltonomous.baselines.classical.config import load_config as lc
    from aaltonomous.baselines.classical.kinematics import ArmKinematics
    return ArmKinematics(lc(profile="sim"))


def window_center(kin, side, arm_traj, upto=None):
    """Pelvis point between the grasp point (lowest along the approach) and the hand at the lift end."""
    arm = np.asarray(arm_traj)[: upto or len(arm_traj)]
    pts = [kin.grasp_pose(side, q)[0] for q in arm[::4]] + [kin.grasp_pose(side, arm[-1])[0]]
    gp = pts[int(np.argmin([p[2] for p in pts]))]
    return 0.5 * (gp + pts[-1])


def crop(cal, img, p, half_full=HALF_FULL, out=256):
    """img (H,W,3) of the head camera at any scale; square crop centred on p's projection, resized to out."""
    u, v, _ = cal.project("head", p)
    s = img.shape[0] / 480.0
    h = int(round(half_full * s))
    u, v = int(round(u * s)), int(round(v * s))
    pad = np.pad(img, ((h, h), (h, h), (0, 0)))
    c = np.ascontiguousarray(pad[v:v + 2 * h, u:u + 2 * h])
    t = torch.as_tensor(c).permute(2, 0, 1)[None].float()
    return F.interpolate(t, size=(out, out), mode="bilinear", antialias=True, align_corners=False)[0] \
        .permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def downscale(img):
    if img.shape[:2] == FRAME_HW:
        return img
    import cv2
    return cv2.resize(img, (FRAME_HW[1], FRAME_HW[0]), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------ build
def build(grasps, out, cfg):
    grasps, out = Path(grasps), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    enc = make_encoder(cfg["encoder"])
    space = ActionSpace(cfg)
    kin, cal = _kin(), Calibration.load()
    k = space.k
    (out / "manifest.json").write_text(json.dumps({"encoder": cfg["encoder"], "macro_ticks": k, "dim": enc.dim,
                                                   "n_tokens": enc.n_tokens, "crop_half_full": HALF_FULL}, indent=1))
    n = 0
    for pj in sorted((grasps / "picks").glob("*.json")):
        fr = grasps / "frames" / pj.with_suffix(".npz").name
        if not fr.exists():
            continue
        pa, pm = load_episode(pj.with_suffix(""))
        fa, fm = load_episode(fr.with_suffix(""))
        for p in pm["picks"]:
            dst = out / f"{pj.stem}_pick{p['i']:02d}.pt"
            if dst.exists() or p.get("tick0") is None:
                continue
            i = p["i"]
            c = window_center(kin, p["side"], pa[f"arm_{i}"])
            j0 = p["tick0"] // k
            j1 = min((p["tick0"] + p["lift_end"]) // k, len(fa["t"]) - 1)
            if j1 - j0 < 2:
                continue
            imgs = np.stack([crop(cal, fa["head_rgb"][j], c) for j in range(j0, j1 + 1)])
            z = enc.encode({enc.cameras[0]: imgs}).to(torch.float16).cpu()
            arm, hand = fa["arm_cmd"][j0:j1 + 1], fa["hand_cmd"][j0:j1 + 1]
            torch.save({"z": z, "state": torch.from_numpy(space.state(arm, hand)),
                        "action": torch.from_numpy(space.actions_from_commands(arm, hand)),
                        "meta": {"episode": pj.stem, "pick": i, "held": p["held"], "variant": p["variant"],
                                 "success": None, "center": c.tolist()}, "name": dst.stem}, dst)
            n += 1
    print(f"[crop] {n} new pick sequences -> {out}")


# ------------------------------------------------------------------------------------------------ probe
def _logreg(X, y, lam=0.05, iters=300):
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    t = torch.as_tensor(y, dtype=torch.float32, device=X.device)
    opt = torch.optim.LBFGS([w, b], max_iter=iters)

    def clo():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(X @ w + b, t) + lam * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(clo)
    return w.detach(), b.detach()


class LinearProbe:
    """Standardised mean token -> logistic regression."""

    def __init__(self, mu, sd, w, b):
        self.mu, self.sd, self.w, self.b = mu, sd, w, b

    @classmethod
    def fit(cls, Z, y, lam=0.05):
        X = Z.float().mean(1)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        w, b = _logreg((X - mu) / sd, y, lam)
        return cls(mu, sd, w, b)

    def __call__(self, Z):
        X = (Z.float().mean(1).to(self.mu.device) - self.mu) / self.sd
        return torch.sigmoid(X @ self.w + self.b)

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("mu", "sd", "w", "b")}


def auc(y, s):
    from .train_critic import auc as _auc
    return _auc(y, s)


@torch.no_grad()
def imagine(model, z0, s0, acts, device):
    """z0 (C,N,D), s0 (C,16), acts list of (M_i,16) -> imagined final latents (C,N,D)."""
    M = max(len(a) for a in acts)
    A = torch.zeros(len(acts), M, 16, device=device)
    for i, a in enumerate(acts):
        A[i, :len(a)] = torch.as_tensor(a, dtype=torch.float32)
        A[i, len(a):, 14:] = A[i, len(a) - 1, 14:]
    pred = model.rollout(z0.to(device).float(), s0.to(device).float(), A, step_state)
    last = torch.tensor([len(a) - 1 for a in acts], device=device)
    return pred[torch.arange(len(acts), device=device), last]


def probe(grasps, latents, ckpt, out, device, folds=5, lam=0.05):
    model, ck = load_checkpoint(ckpt, device)
    cfg = load_config()
    cfg["macro_ticks"] = ck["macro_ticks"]
    space = ActionSpace(cfg)
    Z0, S0, ACT, ZR, y, ep = [], [], [], [], [], []
    cache = {}
    for f in sorted(Path(latents).glob("*_pick*.pt")):
        d = torch.load(f, weights_only=False)
        m = d["meta"]
        if m["episode"] not in cache:
            cache = {m["episode"]: load_episode(Path(grasps) / "picks" / m["episode"])}
        pa, _ = cache[m["episode"]]
        i = m["pick"]
        Z0.append(d["z"][0])
        S0.append(d["state"][0])
        # imagine from the *plan* (what the policy has at decision time), not the executed commands
        ACT.append(traj_actions(space, pa[f"arm_{i}"], pa[f"hand_{i}"], pa[f"arm0_{i}"], pa[f"hand0_{i}"]))
        ZR.append(d["z"][-1])
        y.append(bool(m["held"]))
        ep.append(m["episode"])
    y = np.array(y)
    Zi = torch.cat([imagine(model, torch.stack(Z0[s:s + 64]), torch.stack(S0[s:s + 64]), ACT[s:s + 64], device).cpu()
                    for s in range(0, len(y), 64)])
    Z0, ZR = torch.stack(Z0), torch.stack(ZR)
    eps = sorted(set(ep))
    fo = np.array([{e: i % folds for i, e in enumerate(np.random.default_rng(0).permutation(eps))}[e] for e in ep])
    res = {}
    for name, Z in (("imagined_lift", Zi), ("look_only", Z0), ("real_lift", ZR)):
        s = np.zeros(len(y))
        for k in range(folds):
            tr, te = fo != k, fo == k
            p = LinearProbe.fit(Z[tr].to(device), y[tr], lam)
            s[te] = p(Z[te].to(device)).cpu().numpy()
        res[name] = round(auc(y, s), 3)
    # the decision the critic makes: rank candidates for the same object. Pairwise accuracy among picks of the same
    # object (episode + colour seen at the same look) is not available offline, so report AUC only.
    rep = {"n": int(len(y)), "episodes": len(eps), "held_rate": round(float(y.mean()), 3), "auc": res}
    print("[crop critic]", json.dumps(rep))
    p = LinearProbe.fit(Zi.to(device), y, lam)
    torch.save({"probe": p.state(), "ckpt": str(ckpt), "report": rep, "half_full": HALF_FULL}, out)
    print(f"[crop critic] saved {out}")
    return rep


# ------------------------------------------------------------------------------------------------ outcome-aware training
def _load_picks(grasps, latents, space):
    items, cache = [], {}
    for f in sorted(Path(latents).glob("*_pick*.pt")):
        d = torch.load(f, weights_only=False)
        m = d["meta"]
        if m["episode"] not in cache:
            cache = {m["episode"]: load_episode(Path(grasps) / "picks" / m["episode"])}
        pa, _ = cache[m["episode"]]
        i = m["pick"]
        d["plan_act"] = torch.as_tensor(traj_actions(space, pa[f"arm_{i}"], pa[f"hand_{i}"], pa[f"arm0_{i}"],
                                                     pa[f"hand0_{i}"]), dtype=torch.float32)
        d["held"] = bool(m["held"])
        d["episode"] = m["episode"]
        items.append(d)
    return items


def train_outcome(items, cfg, device, steps=1500, w_out=0.5, lam=0.05, seed=0, log=None):
    """Predictor from scratch on `items`: teacher-forced + rollout L1 (train.py) and, per pick, the full rollout of
    its plan from the first crop must be read as `held` by a probe fitted on the REAL lift-end latents."""
    from .predictor import ACPredictor
    from .train import Windows, losses
    pc, tc = cfg["predictor"], cfg["train"]
    torch.manual_seed(seed)
    T = int(pc["context"])
    eps = [{"z": d["z"], "state": d["state"], "action": d["action"]} for d in items if len(d["z"]) > T]
    ds = Windows(eps, T)
    dl = torch.utils.data.DataLoader(ds, batch_size=tc["batch"], shuffle=True, drop_last=True)
    real = LinearProbe.fit(torch.stack([d["z"][-1] for d in items]).to(device), np.array([d["held"] for d in items]), lam)
    z_dim, n_tok = items[0]["z"].shape[-1], items[0]["z"].shape[-2]
    model = ACPredictor(z_dim, n_tok, pc["dim"], pc["depth"], pc["heads"], max_steps=T + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=steps, pct_start=0.1)
    yb = torch.tensor([d["held"] for d in items], dtype=torch.float32, device=device)
    step, it = 0, iter(dl)
    while step < steps:
        try:
            z, s, a = next(it)
        except StopIteration:
            it = iter(dl)
            z, s, a = next(it)
        z, s, a = z.to(device), s.to(device), a.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            l_tf, l_ro = losses(model, z, s, a, tc["rollout_steps"])
            loss = l_tf + l_ro
        if w_out > 0:   # outcome term: full plan rollouts of a batch of picks (gradient through the rollout)
            idx = np.random.choice(len(items), min(16, len(items)), replace=False)
            acts = [items[i]["plan_act"] for i in idx]
            M = max(len(x) for x in acts)
            A = torch.zeros(len(idx), M, 16, device=device)
            for r, x in enumerate(acts):
                A[r, :len(x)] = x.to(device)
                A[r, len(x):, 14:] = x[-1, 14:].to(device)
            zz = torch.stack([items[i]["z"][0] for i in idx]).to(device).float()[:, None]
            ss = torch.stack([items[i]["state"][0] for i in idx]).to(device)[:, None]
            outs = []
            for h in range(M):
                aa = torch.cat([torch.zeros(len(idx), zz.shape[1] - 1, 16, device=device), A[:, :h + 1]], 1) \
                    if zz.shape[1] > h + 1 else A[:, :h + 1]
                aa = aa[:, -model.max_steps:]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    p = model(zz[:, -model.max_steps:], ss[:, -model.max_steps:], aa)[:, -1].float()
                outs.append(p)
                zz = torch.cat([zz, p[:, None]], 1)
                ss = torch.cat([ss, step_state(ss[:, -1], A[:, h])[:, None]], 1)
            last = torch.tensor([len(x) - 1 for x in acts], device=device)
            zl = torch.stack(outs, 1)[torch.arange(len(idx), device=device), last]
            logit = ((zl.mean(1) - real.mu) / real.sd) @ real.w + real.b
            loss = loss + w_out * F.binary_cross_entropy_with_logits(logit, yb[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if log and step % 250 == 0:
            log(f"  step {step}/{steps} loss {loss.item():.4f}")
    return model.eval()


@torch.no_grad()
def imagine_items(model, items, device):
    out = []
    for s in range(0, len(items), 64):
        b = items[s:s + 64]
        out.append(imagine(model, torch.stack([d["z"][0] for d in b]), torch.stack([d["state"][0] for d in b]),
                           [d["plan_act"] for d in b], device).cpu())
    return torch.cat(out)


def cv_outcome(grasps, latents, device, folds=5, steps=1500, w_out=0.5, lam=0.05, out=None):
    cfg = load_config(None, {"predictor.context": 8, "train.rollout_steps": 4})
    items = _load_picks(grasps, latents, ActionSpace(cfg))
    y = np.array([d["held"] for d in items])
    eps = sorted({d["episode"] for d in items})
    fmap = {e: i % folds for i, e in enumerate(np.random.default_rng(0).permutation(eps))}
    fo = np.array([fmap[d["episode"]] for d in items])
    s_img, s_look = np.zeros(len(y)), np.zeros(len(y))
    for k in range(folds):
        tr = [d for d, f in zip(items, fo) if f != k]
        te = [d for d, f in zip(items, fo) if f == k]
        m = train_outcome(tr, cfg, device, steps, w_out, lam, seed=k)
        p = LinearProbe.fit(imagine_items(m, tr, device).to(device), y[fo != k], lam)
        s_img[fo == k] = p(imagine_items(m, te, device).to(device)).cpu().numpy()
        pl = LinearProbe.fit(torch.stack([d["z"][0] for d in tr]).to(device), y[fo != k], lam)
        s_look[fo == k] = pl(torch.stack([d["z"][0] for d in te]).to(device)).cpu().numpy()
        print(f"[outcome cv] fold {k}: imagined AUC {auc(y[fo == k], s_img[fo == k]):.3f}", flush=True)
    rep = {"n": int(len(y)), "episodes": len(eps), "w_out": w_out, "steps": steps,
           "auc_imagined": round(auc(y, s_img), 3), "auc_look_only": round(auc(y, s_look), 3)}
    print("[outcome cv]", json.dumps(rep), flush=True)
    if out:   # final model on everything
        m = train_outcome(items, cfg, device, steps, w_out, lam, seed=0)
        from .predictor import save_checkpoint
        man = json.loads((Path(latents) / "manifest.json").read_text())
        ck = str(Path(out).with_suffix("")) + "_predictor.pt"
        save_checkpoint(ck, m, cfg, {"encoder": man["encoder"], "macro_ticks": man["macro_ticks"]})
        p = LinearProbe.fit(imagine_items(m, items, device).to(device), y, lam)
        torch.save({"probe": p.state(), "ckpt": ck, "report": rep, "half_full": HALF_FULL}, out)
        print(f"[outcome] saved {out} (+ {ck})")
    return rep


# ------------------------------------------------------------------------------------------------ runtime
class CropGraspCritic:
    def __init__(self, path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cp = torch.load(path, map_location=self.device, weights_only=False)
        self.model, ck = load_checkpoint(cp["ckpt"], self.device)
        cfg = load_config()
        cfg["encoder"], cfg["macro_ticks"] = ck["encoder"], ck["macro_ticks"]
        self.enc = make_encoder(cfg["encoder"], self.device)
        self.space = ActionSpace(cfg)
        self.probe = LinearProbe(**{k: v.to(self.device) for k, v in cp["probe"].items()})
        self.kin, self.cal = _kin(), Calibration.load()

    @torch.no_grad()
    def score_plans(self, obs, arm_q, hand_q, plans):
        img = downscale(np.asarray(obs["head_rgb"]))
        crops, acts = [], []
        for p in plans:
            le = lift_end(p)
            crops.append(crop(self.cal, img, window_center(self.kin, p.side, p.arm, le)))
            acts.append(traj_actions(self.space, p.arm, p.hand, arm_q, hand_q, le))
        z0 = self.enc.encode({self.enc.cameras[0]: np.stack(crops)})
        s0 = torch.as_tensor(self.space.state(arm_q, hand_q)).expand(len(plans), 16)
        sc = self.probe(imagine(self.model, z0, s0, acts, self.device)).cpu().numpy()
        self.last = {"crops": crops, "scores": sc}   # for viewers (live_demo.py)
        return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "probe", "outcome"])
    ap.add_argument("--grasps", default="runs/jepa/grasps")
    ap.add_argument("--latents", default="runs/jepa/crop_latents")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default="runs/jepa/ckpt_crop/best.pt")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--w-out", type=float, default=0.5, help="outcome loss weight (0 = plain L1 JEPA)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.grasps, a.out or a.latents, load_config())
    elif a.cmd == "outcome":
        cv_outcome(a.grasps, a.latents, a.device, a.folds, a.steps, a.w_out, a.lam, a.out)
    else:
        probe(a.grasps, a.latents, a.ckpt, a.out or "runs/jepa/crop_critic.pt", a.device, lam=a.lam)


if __name__ == "__main__":
    main()
