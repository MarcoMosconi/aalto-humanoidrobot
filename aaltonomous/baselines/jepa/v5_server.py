"""v5 (classical + Dex3 finger check + V-JEPA 2 grasp verifier, FingerJepaVerifyPolicy) behind the policy-server protocol.

    python -m aaltonomous.baselines.jepa.v5_server                       # :8000
    python -m aaltonomous.sim.evaluate --policy ws://localhost:8000 --team v5 --trials 8     # sim gate
    mkdir -p jepa_weights && cp runs/jepa/held_verifier_all_fa0.10.pt runs/jepa/variant_prior_dlv.json jepa_weights/
    docker build -f aaltonomous/docker/v5.Dockerfile -t myteam/v5:v1 .

Same policy object as `grasp_eval.py --policy v5` (no forked logic): the classical planner runs threaded (1-tick
holds while it looks / plans), and after each lift the finger closure decides held / empty; the V-JEPA 2 crop verifier
decides the ambiguous band. That check runs inside act() (one ViT-L forward on one crop, ~tens of ms on a GPU).
By default the verifier only runs when the fingers are unsure (`AALTO_V5_ALWAYS_JEPA=1`: on every pick, as in
grasp_eval, for logging); the decisions are the same either way.
Rung 1 only (colour -> bin from scene.yaml), like the classical baseline.
Env: V5_VERIFIER, V5_FAMILY_PRIOR (files), V5_MAX_ATTEMPTS (6), AALTO_TIME_LIMIT (120), AALTO_ARM_VEL, AALTO_PROFILE,
AALTO_V5_ALWAYS_JEPA.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch

from aaltonomous.deploy.policy_server import Policy, serve

from .crop_critic import HeldVerifier
from .grasp import FingerJepaVerifyPolicy

log = logging.getLogger("aaltonomous.v5")


class _LazyJepaV5(FingerJepaVerifyPolicy):
    """FingerJepaVerifyPolicy whose crop verifier only runs when the finger reading is in the ambiguous band."""

    always_jepa = False

    def _check(self, obs, plan):
        if self.always_jepa:
            return super()._check(obs, plan)
        g = float(self.space.grip_of(np.asarray(obs["hand_q"], float))[0 if plan.side == "left" else 1])
        if g >= self.grip_empty:
            return {"grip": round(g, 4), "by": "fingers"}, True, f"fingers closed on nothing (grip {g:.3f})"
        if g <= self.grip_held:
            return {"grip": round(g, 4), "by": "fingers"}, False, f"fingers stopped by the object (grip {g:.3f})"
        return super()._check(obs, plan)


class V5ServerPolicy(Policy):
    name = "v5-finger-jepa"

    def __init__(self, verifier, family_prior=None, max_attempts=6, time_limit=None, arm_vel=None, always_jepa=False):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        ov = {}
        av = arm_vel or os.environ.get("AALTO_ARM_VEL")
        if av:
            ov["motion.arm_max_vel"] = float(av)
        prior = json.load(open(family_prior)) if family_prior else None
        self.pol = _LazyJepaV5(threaded=True, require_confirm=False, time_limit=tl, overrides=ov,
                               log=lambda m: log.info(m), verifier=HeldVerifier(verifier),
                               max_attempts=max_attempts, family_prior=prior)
        self.pol.always_jepa = always_jepa
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])
        self._n_ver = 0
        self._warmup()

    def _warmup(self):
        """One verifier forward so the CUDA kernels / weights are loaded before the first episode (startup budget)."""
        enc = self.pol.verifier.enc
        img = np.full((self.pol.verifier.enc.size,) * 2 + (3,), 127, np.uint8)
        for _ in range(2):
            t0 = time.monotonic()
            self.pol.verifier.probe(enc.encode({enc.cameras[0]: img[None]}))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log.info("verifier forward %.0f ms (%s)", 1e3 * (time.monotonic() - t0), self.pol.verifier.device)

    def reset(self, instruction, task=None, seed=None):
        job, t0 = self.pol._job, time.monotonic()      # let a planning thread from the last episode finish
        while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
            time.sleep(0.02)
        if task not in (None, "sort_fixed"):
            log.warning("task %r: v5 only does rung 1 (colour = bin from scene.yaml)", task)
        self.pol.reset({"instruction": instruction})
        self._n_ver = 0
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)

    def act(self, obs):
        t0 = time.monotonic()
        a = self.pol.act(obs)
        if len(self.pol.verifications) > self._n_ver:
            v = self.pol.verifications[-1]
            log.info("check %s %s: %s abort=%s (act %.0f ms)", v["color"], v["variant"],
                     {k: v[k] for k in ("grip", "p_held", "by") if k in v}, v["abort"], 1e3 * (time.monotonic() - t0))
            self._n_ver = len(self.pol.verifications)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verifier", default=os.environ.get("V5_VERIFIER", "runs/jepa/held_verifier_all_fa0.10.pt"))
    ap.add_argument("--family-prior", default=os.environ.get("V5_FAMILY_PRIOR", "runs/jepa/variant_prior_dlv.json"))
    ap.add_argument("--max-attempts", type=int, default=int(os.environ.get("V5_MAX_ATTEMPTS", 6)))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = V5ServerPolicy(a.verifier, a.family_prior or None, a.max_attempts,
                         always_jepa=os.environ.get("AALTO_V5_ALWAYS_JEPA") == "1")
    log.info("v5 loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=a.port)
