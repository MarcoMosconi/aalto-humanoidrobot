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
                           "lift_end": lift_end(plan), "scores": getattr(plan, "scores", None),
                           "shape": getattr(plan, "shape_info", None)})
        return super()._start(plan, t)

    def choose(self, det, arm_q, hand_q, obstacles):
        raise NotImplementedError


class ExploreGraspPolicy(_VariantPolicy):
    """Data collection: with prob. p_classical the stock plan, else a random valid variant."""

    def __init__(self, *a, p_classical=0.3, prefer=None, p_prefer=0.7, **kw):
        super().__init__(*a, **kw)
        self.p_classical = p_classical
        self.prefer, self.p_prefer = prefer, p_prefer   # e.g. "h0.35": oversample risky grasp families

    def choose(self, det, arm_q, hand_q, obstacles):
        vs = make_variants(det, self.rng, n=64 if self.prefer else self.n_variants)
        r = self.rng.random()
        if r < self.p_classical:
            vs = vs[:1]
        elif self.prefer and r < self.p_classical + (1 - self.p_classical) * self.p_prefer:
            pref = [v for v in vs[1:] if self.prefer in v.name]
            vs = pref + [v for v in vs[1:] if v not in pref] + vs[:1]
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


class TriangleProbePolicy(_VariantPolicy):
    """Data collection for triangle grasp orientation: ShapePerception (shapes.py) labels each detection; triangles get
    a random finger direction over the reachable fan (hint weight 1), everything else the stock plan. Each pick logs
    det.shape / det.tri_yaw (plan.shape_info) so held can be read against the hand's angle to the triangle."""

    def __init__(self, *a, fan=(-90.0, 30.0), **kw):
        super().__init__(*a, **kw)
        from .shapes import ShapePerception
        self.perception = ShapePerception(self.cfg, self.scene, self.cal)
        self.fan = fan

    def choose(self, det, arm_q, hand_q, obstacles):
        info = {"shape": getattr(det, "shape", None), "tri_yaw": getattr(det, "tri_yaw", None),
                "feat": getattr(det, "shape_feat", None)}
        vs = []
        if info["shape"] == "triangle":
            vs = [Variant(hint_deg=float(h), hint_weight=1.0, name=f"tri{h:+.0f}")
                  for h in self.rng.uniform(*self.fan, size=6)]
        last = None
        for v in vs + [Variant()]:
            p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
            if p.ok:
                p.variant, p.shape_info = v.as_dict(), info
                return p
            last = p
        return last


TRI_MODES = {"natural": None, "apex": 0.0, "face": 60.0, "across": 30.0}   # finger direction vs the apex direction


def tri_hint_deg(tri_yaw, offset_deg, side, fan=(-90.0, 30.0)):
    """Finger direction for a triangle (pelvis angle = apex direction + offset, any of the 3 symmetric copies) as a
    LEFT-convention hint inside the reachable fan (the planner mirrors y for the right arm). None if no copy fits."""
    best = None
    for k in range(3):
        a = np.degrees(tri_yaw) + offset_deg + 120.0 * k
        a = (a + 180.0) % 360.0 - 180.0
        h = a if side == "left" else -a
        if fan[0] <= h <= fan[1]:
            mid = 0.5 * (fan[0] + fan[1])
            if best is None or abs(h - mid) < abs(best - mid):
                best = h
    return best


class TriangleBenchPolicy(_VariantPolicy):
    """Triangle grasp bench: every triangle pick gets a random (finger-direction mode, grasp height) from TRI_MODES x
    heights; the variant name records both ("tri_<mode>_h<frac>"). Non-triangles: stock plan."""

    def __init__(self, *a, heights=(0.5, 0.35), **kw):
        super().__init__(*a, **kw)
        from .shapes import ShapePerception
        self.perception = ShapePerception(self.cfg, self.scene, self.cal)
        self.heights = heights

    def _variant(self, det, mode, h, side):
        off = TRI_MODES[mode]
        if off is None:
            return Variant(height_frac=h, name=f"tri_{mode}_h{h}")
        hd = tri_hint_deg(det.tri_yaw, off, side)
        return None if hd is None else Variant(hint_deg=hd, hint_weight=1.0, height_frac=h, name=f"tri_{mode}_h{h}")

    def choose(self, det, arm_q, hand_q, obstacles):
        info = {"shape": getattr(det, "shape", None), "tri_yaw": getattr(det, "tri_yaw", None)}
        if info["shape"] == "triangle" and info["tri_yaw"] is not None:
            for _ in range(6):
                mode = list(TRI_MODES)[int(self.rng.integers(len(TRI_MODES)))]
                h = float(self.heights[int(self.rng.integers(len(self.heights)))])
                for side in ("left", "right"):             # the planner picks the arm; hint depends on it
                    v = self._variant(det, mode, h, side)
                    if v is None:
                        continue
                    p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
                    if p.ok and (v.hint_deg is None or p.side == side):
                        p.variant, p.shape_info = v.as_dict(), {**info, "mode": mode}
                        return p
        p = plan_variant(self._inner, det, Variant(), arm_q, hand_q, obstacles)
        p.variant, p.shape_info = Variant().as_dict(), info
        return p


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
        self._cands = []
        if not plans:
            return last
        grasp = [p for p in plans if lift_end(p) is not None]
        self._cands = grasp                               # for viewers (live_demo.py)
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


class JepaVerifyPolicy(_VariantPolicy):
    """Classical + V-JEPA 2 grasp verifier (+ optional imagination critic).

    Right after a pick's lift phase (the chunk is cut exactly there) the head image is cropped around the grasp and
    the verifier reads p(held). Below its threshold the pick is aborted: the hand opens where it is, the arm goes home,
    and the next look retries the object with a grasp variant not tried on it yet (critic-ranked when a critic is
    given, else the make_variants order). A failed grasp then costs ~3 s instead of a full carry + drop + retreat.
    """

    def __init__(self, *a, verifier, critic=None, margin=0.05, max_attempts=3, prior=None, **kw):
        ov = dict(kw.pop("overrides", None) or {})
        ov.setdefault("policy.max_attempts", max_attempts)
        super().__init__(*a, overrides=ov, **kw)
        self.verifier, self.critic, self.margin = verifier, critic, margin
        self.prior = prior            # {family: {rate}} (fit_variant_prior): try the best-holding grasp families first
        self.reset()

    def reset(self, info=None):
        super().reset(info)
        self.tried = []       # (xy, variant name) of aborted / finished grasps
        self._exec = None
        self.n_aborts = 0
        self.verifications = []

    # ---- choice
    def _tried_here(self, det):
        r = self.cfg["policy"]["same_object_radius"] * 1.5
        return {n for xy, n in self.tried if np.linalg.norm(np.asarray(xy) - det.top[:2]) < r}

    def choose(self, det, arm_q, hand_q, obstacles):
        skip = self._tried_here(det)
        plans, last = [], None
        vs = make_variants(det, None, n=64)
        if self.prior is not None:
            vs = order_by_prior(vs, self.prior)
        for v in vs:
            if v.name in skip:
                continue
            p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
            if p.ok:
                p.variant = v.as_dict()
                plans.append(p)
            else:
                last = last or p
            if self.critic is None and plans:
                break                                   # no critic: first untried variant that plans
            if len(plans) >= self.n_variants:
                break
        self._cands = [p for p in plans if lift_end(p) is not None]
        if not plans:
            return last
        grasp = self._cands
        if self.critic is None or len(grasp) <= 1:
            return plans[0]
        sc = np.asarray(self.critic.score_plans(self._look_obs, arm_q, hand_q, grasp))
        for p in grasp:
            p.scores = [round(float(x), 3) for x in sc]
        best = float(sc.max())
        pick = 0 if (grasp[0].variant["name"] == "classical" and sc[0] >= best - self.margin) else int(sc.argmax())
        self._note(float(self._look_obs.get("t", 0.0)), f"grasp critic {det.color}: " + ", ".join(
            f"{p.variant['name']}={s:.2f}" for p, s in zip(grasp, sc)) + f" -> {grasp[pick].variant['name']}")
        return grasp[pick]

    # ---- execution with the check after the lift
    def _start(self, plan, t):
        out = super()._start(plan, t)
        le = lift_end(plan)
        self._exec = None if le is None or plan.target_bin == "counter" else \
            {"plan": plan, "total": len(plan.arm), "lift_end": le, "checked": False}
        return out

    def _executed(self):
        return self._exec["total"] - (len(self.queue_a) if self.queue_a is not None else 0)

    def _pop(self):
        e = self._exec
        if e is not None and not e["checked"] and self.queue_a is not None and len(self.queue_a):
            left = e["lift_end"] - self._executed()
            if 0 < left < int(self.cfg["motion"]["chunk"]):   # end this chunk exactly at the lift end
                a, h = self.queue_a[:left], self.queue_h[:left]
                self.queue_a, self.queue_h = self.queue_a[left:], self.queue_h[left:]
                self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
                return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}
        return super()._pop()

    def act(self, obs):
        e = self._exec
        if e is not None and not e["checked"] and self.state == "executing" and self._executed() >= e["lift_end"]:
            e["checked"] = True
            plan = e["plan"]
            t = float(obs.get("t", 0.0))
            xy = plan.waypoints["grasp"][:2]
            self.tried.append((xy, plan.variant["name"]))
            rec, abort, why = self._check(obs, plan)
            self.verifications.append({"t": t, "color": plan.color, "variant": plan.variant["name"], **rec,
                                       "abort": abort})
            if abort:
                self.n_aborts += 1
                self._note(t, f"verify {plan.color} ({plan.variant['name']}): {why} -> abort, retry with another grasp")
                self._abort(plan)
            else:
                self._note(t, f"verify {plan.color}: {why} -> carry on")
        return super().act(obs)

    def _check(self, obs, plan):
        """-> (record, abort?, reason). V-JEPA 2 crop verifier only."""
        p = self.verifier.p_held(obs, plan)
        thr = self.verifier.threshold
        return {"p_held": p}, p < thr, f"p(held)={p:.2f} {'<' if p < thr else '>='} {thr:.2f}"

    def _abort(self, plan):
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        sl = ArmKinematics.sl(plan.side)
        hand_rel = self.hand_cmd.copy()
        hand_rel[sl] = self.post[self.cfg["grasp"].get("release_posture", "preshape")][sl]
        hand_open = self.hand_cmd.copy()
        hand_open[sl] = self.post["open"][sl]
        home = self.arm_cmd.copy()
        home[sl] = self.home[sl]
        a1, h1 = self._inner.go_to(self.arm_cmd, self.hand_cmd, self.arm_cmd, hand_rel)   # let go where it is
        hold = int(round(0.3 * 30))
        a1 = np.vstack([a1, np.repeat(self.arm_cmd[None], hold, 0)])
        h1 = np.vstack([h1, np.repeat(hand_rel[None], hold, 0)])
        a2, h2 = self._inner.go_to(self.arm_cmd, hand_rel, home, hand_open)
        self._enqueue(np.vstack([a1, a2]), np.vstack([h1, h2]))
        self._exec = None


class FingerJepaVerifyPolicy(JepaVerifyPolicy):
    """v5: the lift-end check reads the Dex3 finger closure first, the V-JEPA 2 crop verifier second.

    grip = projection of the hand's joint positions on the open->closed line (1 = fully closed on nothing). In the
    sim a held object stops the fingers at 0.92-0.995, an empty hand closes to >= 0.999 (589 picks, no overlap). Clear
    readings decide alone; readings in between (sensor noise / calibration on the real robot) go to the JEPA verifier.
    Retries switch grasp family (finger direction band x height) before reusing one, best delivery rate first, and
    the stock classical grasp is always tried first.
    """

    def __init__(self, *a, grip_empty=0.998, grip_held=0.99, family_prior=None, **kw):
        super().__init__(*a, **kw)
        from .common import ActionSpace, load_config
        self.space = ActionSpace(load_config())
        self.grip_empty, self.grip_held = grip_empty, grip_held
        self.family_prior = family_prior

    def _check(self, obs, plan):
        g = float(self.space.grip_of(np.asarray(obs["hand_q"], float))[0 if plan.side == "left" else 1])
        p = float(self.verifier.p_held(obs, plan))          # always run: logged + shown, decides only if ambiguous
        thr = self.verifier.threshold
        rec = {"grip": round(g, 4), "p_held": p}
        if g >= self.grip_empty:
            return {**rec, "by": "fingers"}, True, f"fingers closed on nothing (grip {g:.3f}), JEPA p(held)={p:.2f}"
        if g <= self.grip_held:
            return {**rec, "by": "fingers"}, False, f"fingers stopped by the object (grip {g:.3f}), JEPA p(held)={p:.2f}"
        return {**rec, "by": "jepa"}, p < thr, f"fingers unsure (grip {g:.3f}) -> JEPA p(held)={p:.2f} vs {thr:.2f}"

    def choose(self, det, arm_q, hand_q, obstacles):
        skip = self._tried_here(det)
        fams = {variant_family({"name": n, "hint_deg": _hint_of(n), "height_frac": _h_of(n)}) for n in skip}
        vs = make_variants(det, None, n=64)
        rest = [v for v in vs[1:] if v.name not in skip]
        if self.family_prior is not None:
            rest = order_by_prior(rest, self.family_prior)
        rest = [v for v in rest if variant_family(v) not in fams] + [v for v in rest if variant_family(v) in fams]
        order = ([vs[0]] if vs[0].name not in skip else []) + rest
        last = None
        for v in order:
            p = plan_variant(self._inner, det, v, arm_q, hand_q, obstacles)
            if p.ok:
                p.variant = v.as_dict()
                self._cands = [p] if lift_end(p) is not None else []
                return p
            last = last or p
        self._cands = []
        return last


def _hint_of(name):
    return float(name[4:].split("_")[0]) if name.startswith("hint") else 0.0


def _h_of(name):
    return float(name.split("_h")[1]) if "_h" in name else 0.5


# --------------------------------------------------------------------------------------------- data-driven variant order
def variant_family(v):
    """Coarse family of a variant dict / Variant: finger direction band x grasp height (or shift / classical)."""
    d = v.as_dict() if isinstance(v, Variant) else v
    n = d["name"]
    if n == "classical":
        return "classical"
    if n.startswith("shift"):
        return "shift"
    h = d["hint_deg"]
    band = "fwd" if h > -20 else "diag" if h > -55 else "inward"
    return f"{band}_h{d['height_frac']}"


def fit_variant_prior(picks_dir, out=None, alpha=2.0, label="held"):
    """Success rate (`label`: held | delivered) per family from collect_grasps.py picks, shrunk towards the overall
    rate (alpha pseudo-counts)."""
    import glob
    import json
    rows = [p for f in glob.glob(f"{picks_dir}/*.json") for p in json.load(open(f))["picks"]
            if p.get(label) is not None]
    base = float(np.mean([bool(p[label]) for p in rows]))
    fam = {}
    for p in rows:
        fam.setdefault(variant_family(p["variant"]), []).append(bool(p[label]))
    prior = {k: {"n": len(v), "rate": round((sum(v) + alpha * base) / (len(v) + alpha), 3)} for k, v in fam.items()}
    prior["_base"] = {"n": len(rows), "rate": round(base, 3)}
    if out:
        with open(out, "w") as f:
            json.dump(prior, f, indent=1)
    return prior


def order_by_prior(variants, prior):
    base = prior.get("_base", {}).get("rate", 0.5)
    return sorted(variants, key=lambda v: -prior.get(variant_family(v), {}).get("rate", base))
