"""The JEPA policy behind the submission protocol (port 8000).

    python -m aaltonomous.baselines.jepa.policy_server --ckpt runs/jepa/ckpt/best.pt --bank runs/jepa/latents
    docker build -f aaltonomous/docker/jepa.Dockerfile -t myteam/jepa:v1 .

Threaded: while it encodes + plans, act() returns `policy.hold_ticks`-tick holds at once, then one macro step
(`macro_ticks` ticks). The V-JEPA 2 weights load (and warm up) before serve(), inside the 120 s startup budget.
Env: JEPA_CKPT, JEPA_BANK (defaults for the flags), AALTO_TIME_LIMIT (s, default 120), HF_HUB_OFFLINE=1 in the image.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from aaltonomous.deploy.policy_server import Policy, serve

from .policy import JepaPolicy

log = logging.getLogger("aaltonomous.jepa")


class JepaServerPolicy(Policy):
    name = "jepa-ac"

    def __init__(self, ckpt, bank, time_limit=None, device=None):
        tl = float(time_limit or os.environ.get("AALTO_TIME_LIMIT", 120.0))
        self.pol = JepaPolicy.from_checkpoint(ckpt, bank, device=device, threaded=True, time_limit=tl,
                                              log=lambda m: log.info(m))
        self.action_horizon = int(self.pol.cfg["macro_ticks"])
        self.name = f"jepa-ac ({self.pol.cfg['encoder']['kind']}, bank {len(self.pol.bank)} frames)"
        self._warmup()

    def _warmup(self):
        """One think on a blank-ish frame so CUDA kernels load before the first episode."""
        img = np.full((480, 640, 3), 127, np.uint8)
        obs = {c: img for c in self.pol.enc.cameras}
        obs.update(arm_q=self.pol.home, hand_q=self.pol.h_open)
        self.pol.reset({"instruction": ""})
        self.pol._think(obs, self.pol.home.copy(), self.pol.h_open.copy())
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset(self, instruction, task=None, seed=None):
        job, t0 = self.pol._job, time.monotonic()
        while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
            time.sleep(0.02)
        self.pol.reset({"instruction": instruction})
        log.info("reset task=%s seed=%s instruction=%r", task, seed, instruction)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("JEPA_CKPT", "runs/jepa/ckpt/best.pt"))
    ap.add_argument("--bank", default=os.environ.get("JEPA_BANK", "runs/jepa/latents"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = JepaServerPolicy(a.ckpt, a.bank)
    log.info("%s loaded in %.1f s", pol.name, time.monotonic() - t0)
    serve(pol, port=a.port)
