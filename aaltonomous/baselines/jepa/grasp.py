"""JEPA grasp critic on top of the classical rung-1 sorter (the classical code is reused, not changed).

The classical baseline loses most objects in the grasp (slips / spins, mostly triangles). Here the classical planner
proposes several grasp *variants* per object (finger direction incl. yaw-aligned, xy offset, height; candidate 0 is
the stock classical plan), and the V-JEPA 2 world model imagines each one:

    z0 = V-JEPA2(head image at the look)                         (frozen encoder)
    ẑ_lift = predictor rollout of the plan's macro actions up to the end of its `lift` phase
    p(held) = probe(ẑ_lift)                                       (trained on imagined latents, labelled in sim)

and the policy executes the candidate with the highest p(held) (ties -> shortest). Labels come from the sim scene
state at the lift end (privileged, training only).

    ExploreGraspPolicy   classical + a random grasp variant per pick (data collection, collect_grasps.py)
    JepaGraspPolicy      classical + critic-selected variant (evaluation / submission)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from aaltonomous.baselines.classical.policy import ClassicalPolicy

from .common import ActionSpace


# --------------------------------------------------------------------------------------------- grasp variants
@dataclass
class Variant:
    hint_deg: float | None = None    # finger direction in the pelvis xy plane, LEFT-arm convention (right mirrors);
    hint_weight: float | None = None  # None = classical finger_hints loop with its weight
    dxy: tuple = (0.0, 0.0)          # grasp point shift (pelvis frame, m)
    height_frac: float | None = None
    name: str = "classical"

    def as_dict(self):
        return dict(hint_deg=self.hint_deg, hint_weight=self.hint_weight, dxy=list(self.dxy),
                    height_frac=self.height_frac, name=self.name)


def make_variants(det, rng=None, n=6):
    """Candidate grasps for one detection. Candidate 0 = stock classical. Yaw-aligned finger directions use the
    perception's top-face yaw (rad, pelvis) folded into the reachable fan [-90, +30] deg (left-arm convention)."""
    out = [Variant()]
    fan = []
    if det.yaw is not None:
        for k in range(4):
            a = np.degrees(det.yaw) + 90.0 * k
            a = (a + 180.0) % 360.0 - 180.0
            if -90.0 <= a <= 30.0:
                fan.append(a)
        for k in (1, 2):  # triangles: faces every 60 deg (shape unknown to the perception: cheap to add)
            a = (np.degrees(det.yaw) + 60.0 * k + 180.0) % 360.0 - 180.0
            if -90.0 <= a <= 30.0:
                fan.append(a)
    fan += [0.0, -30.0, -60.0]
    seen = set()
    for a in fan:
        key = round(a / 10.0)
        if key in seen:
            continue
        seen.add(key)
        for hf in (0.5, 0.35):
            out.append(Variant(hint_deg=float(a), hint_weight=1.0, height_frac=hf, name=f"hint{a:+.0f}_h{hf}"))
    for d in ((0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006)):
        out.append(Variant(dxy=d, name=f"shift{d[0]*1000:+.0f},{d[1]*1000:+.0f}"))
    if rng is not None:
        rest = list(rng.permutation(len(out) - 1) + 1)
        out = [out[0]] + [out[i] for i in rest]
    return out[:n]


def plan_variant(planner, det, v: Variant, arm_q, hand_q, obstacles=()):
    """Plan `det` with variant `v` using the classical Planner (its config / IK hint temporarily changed)."""
    if v.hint_deg is None and v.height_frac is None and v.dxy == (0.0, 0.0):
        return planner.plan(det, arm_q, hand_q, obstacles=obstacles)
    g = planner.cfg["grasp"]
    saved = (planner.kin.hint.copy(), planner.kin.w_hint, g["height_frac"], copy.deepcopy(g.get("finger_hints")))
    try:
        if v.height_frac is not None:
            g["height_frac"] = v.height_frac
        if v.hint_deg is not None:
            r = np.radians(v.hint_deg)
            g["finger_hints"] = [[float(np.cos(r)), float(np.sin(r)), 0.0]]
            planner.kin.w_hint = float(v.hint_weight if v.hint_weight is not None else planner.kin.w_hint)
        d = copy.copy(det)
        d.top = np.asarray(det.top, float) + np.r_[v.dxy, 0.0]
        p = planner.plan(d, arm_q, hand_q, obstacles=obstacles, relocate=False)
        if p.ok and p.target_bin == "counter":
            p.ok, p.reason = False, "variant: relocation"
        return p
    finally:
        planner.kin.hint, planner.kin.w_hint, g["height_frac"] = saved[0], saved[1], saved[2]
        if saved[3] is None:
            g.pop("finger_hints", None)
        else:
            g["finger_hints"] = saved[3]


def lift_end(plan):
    for name, s, e in plan.phases:
        if name == "lift":
            return e
    return None


def plan_actions(space: ActionSpace, plan, arm_now, hand_now, upto=None):
    """Plan ticks [0, upto) -> macro-step actions (M,16) in the world model's space (commanded targets every k)."""
    return traj_actions(space, plan.arm, plan.hand, arm_now, hand_now, upto)


def traj_actions(space: ActionSpace, arm_traj, hand_traj, arm_now, hand_now, upto=None):
    k = space.k
    upto = len(arm_traj) if upto is None else upto
    idx = np.arange(k - 1, upto, k)
    if len(idx) == 0 or idx[-1] != upto - 1:
        idx = np.r_[idx, upto - 1]
    arm = np.vstack([np.asarray(arm_now)[None], np.asarray(arm_traj)[idx]])
    hand = np.vstack([np.asarray(hand_now)[None], np.asarray(hand_traj)[idx]])
    return space.actions_from_commands(arm, hand)


# --------------------------------------------------------------------------------------------- probe + critic
class HeldProbe(nn.Module):
    """Attention-pooled linear probe on (imagined) latent tokens -> logit of 'object held after the lift'."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z):
        p, _ = self.attn(self.q.expand(z.shape[0], -1, -1), z, z)
        return self.head(p[:, 0])[:, 0]


class GraspCritic:
    def __init__(self, model, probe, encoder, space: ActionSpace, device, max_steps=None):
        self.model, self.probe, self.enc, self.space = model, probe, encoder, space
        self.device = torch.device(device)
        self.max_steps = max_steps

    @torch.no_grad()
    def encode(self, obs):
        return self.enc.encode({c: np.asarray(obs[c])[None] for c in self.enc.cameras})[0]

    @torch.no_grad()
    def imagine(self, z0, s0, actions_list):
        """z0 (N,D), s0 (16,), list of (M_i,16) -> imagined final latents (C,N,D) (one rollout per candidate)."""
        from .predictor import step_state
        M = max(len(a) for a in actions_list)
        A = torch.zeros(len(actions_list), M, 16, device=self.device)
        for i, a in enumerate(actions_list):
            A[i, :len(a)] = torch.as_tensor(a, dtype=torch.float32)
            A[i, len(a):, :14] = 0.0                     # pad = hold (arm still, grip unchanged)
            A[i, len(a):, 14:] = A[i, len(a) - 1, 14:]
        z = z0.to(self.device).float()
        s = torch.as_tensor(s0, device=self.device, dtype=torch.float32)
        pred = self.model.rollout(z.expand(len(actions_list), *z.shape), s.expand(len(actions_list), 16), A, step_state)
        last = torch.tensor([len(a) - 1 for a in actions_list], device=self.device)
        return pred[torch.arange(len(actions_list), device=self.device), last]

    @torch.no_grad()
    def score(self, z0, s0, actions_list):
        return torch.sigmoid(self.probe(self.imagine(z0, s0, actions_list))).cpu().numpy()


# --------------------------------------------------------------------------------------------- policies
class _VariantPolicy(ClassicalPolicy):
    """ClassicalPolicy whose per-object plan goes through `choose(det, arm, hand, obstacles)`; logs each started pick."""

    def __init__(self, *a, n_variants=6, seed=0, **kw):
        super().__init__(*a, **kw)
        self.n_variants = n_variants
        self.rng = np.random.default_rng(seed)
        self.picks = []
        self._look_obs = None
        outer = self

        class _Hook:  # stands in for self.planner inside ClassicalPolicy._look only
            def __getattr__(self, k):
                return getattr(outer._inner, k)

            def plan(self, det, arm_q, hand_q, obstacles=()):
                return outer.choose(det, arm_q, hand_q, obstacles)

        self._inner = self.planner
        self._hook = _Hook()

    def reset(self, info=None):
        super().reset(info)
        self.picks = []

    def _look(self, obs):
        self._look_obs = obs
        self.planner = self._hook
        try:
            return super()._look(obs)
        finally:
            self.planner = self._inner

    def _start(self, plan, t):
        self.picks.append({"t": t, "plan": plan, "variant": getattr(plan, "variant", {"name": "classical"}),
                           "obs": self._look_obs, "arm0": self.arm_cmd.copy(), "hand0": self.hand_cmd.copy(),
                           "lift_end": lift_end(plan), "scores": getattr(plan, "scores", None)})
        return super()._start(plan, t)

    def choose(self, det, arm_q, hand_q, obstacles):
        raise NotImplementedError


class ExploreGraspPolicy(_VariantPolicy):
    """Data collection: with prob. p_classical the stock plan, else a random valid variant."""

    def __init__(self, *a, p_classical=0.3, **kw):
        super().__init__(*a, **kw)
        self.p_classical = p_classical

    def choose(self, det, arm_q, hand_q, obstacles):
        vs = make_variants(det, self.rng, n=self.n_variants)
        if self.rng.random() < self.p_classical:
            vs = vs[:1]
        else:
            vs = vs[1:] + vs[:1]   # a random variant first, the classical plan as fallback
        last = None
        for v in vs:
            p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
            if p.ok:
                p.variant = v.as_dict()
                return p
            last = p
        return last


class JepaGraspPolicy(_VariantPolicy):
    """Evaluation: plan every variant, imagine each grasp with the world model, run the most likely to hold."""

    def __init__(self, *a, critic: GraspCritic, margin=0.05, **kw):
        super().__init__(*a, **kw)
        self.critic, self.margin = critic, margin

    def choose(self, det, arm_q, hand_q, obstacles):
        plans, last = [], None
        for v in make_variants(det, None, n=self.n_variants):
            p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
            if p.ok:
                p.variant = v.as_dict()
                plans.append(p)
            else:
                last = last or p
        if not plans:
            return last
        grasp = [p for p in plans if lift_end(p) is not None]
        if len(grasp) <= 1:
            return plans[0]
        if hasattr(self.critic, "score_plans"):          # object-centred critic (crop_critic.CropGraspCritic)
            sc = np.asarray(self.critic.score_plans(self._look_obs, arm_q, hand_q, grasp))
        else:
            z0 = self.critic.encode(self._look_obs)
            s0 = self.critic.space.state(arm_q, hand_q)
            acts = [plan_actions(self.critic.space, p, arm_q, hand_q, lift_end(p)) for p in grasp]
            sc = self.critic.score(z0, s0, acts)
        for p, s in zip(grasp, sc):
            p.scores = [round(float(x), 3) for x in sc]
        # stock plan unless a variant is clearly more likely to hold (margin), then the shortest among the best
        best = float(sc.max())
        pick = 0 if (grasp[0].variant["name"] == "classical" and sc[0] >= best - self.margin) else int(sc.argmax())
        self._note(float(self._look_obs.get("t", 0.0)), f"grasp critic {det.color}: " + ", ".join(
            f"{p.variant['name']}={s:.2f}" for p, s in zip(grasp, sc)) + f" -> {grasp[pick].variant['name']}")
        return grasp[pick]
