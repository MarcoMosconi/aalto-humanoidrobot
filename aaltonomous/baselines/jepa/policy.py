"""JepaPolicy: obs -> action with a frozen V-JEPA 2 encoder, a learned action-conditioned latent predictor and CEM.

    pol = JepaPolicy.from_checkpoint("runs/jepa/ckpt/best.pt", bank="runs/jepa/latents")
    pol.reset({"instruction": obs["instruction"]})
    while True:
        obs, info = env.step(pol.act(obs))

Loop: (optional) go to the home pose -> settle -> think -> execute one macro step (`macro_ticks` ticks) -> think ...
Think = encode the current camera image(s), retrieve a sub-goal from the demo bank (goals.py), CEM-plan `horizon`
macro steps towards it (planner.py), keep the first step. With `threaded=True` thinking runs in a background thread
and act() returns short holds meanwhile, so the command rate stays at 30 Hz. Every macro step is pre-checked on a
shadow SafetyFilter; a step that would clip is halved (up to 3 times) or replaced by a hold. `done` when the goal bank
says the matched demo is over, or shortly before the time limit.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np
import torch

from aaltonomous.interface.safety import JointSpec, SafetyConfig, SafetyFilter

from .common import ActionSpace, classical_pose, load_config
from .encoder import make_encoder
from .goals import GoalBank
from .planner import CEMPlanner
from .predictor import load_checkpoint


class _ShadowFilter(SafetyFilter):
    def _log(self, t, reasons, details):  # pre-checks only: don't spam the clip log
        pass


class JepaPolicy:
    def __init__(self, cfg, model, encoder, bank: GoalBank, device, safety: SafetyConfig | None = None,
                 threaded=False, time_limit=120.0, log=print):
        self.cfg, self.model, self.enc, self.bank = cfg, model, encoder, bank
        self.device = torch.device(device)
        self.safety = safety or SafetyConfig(dt=1.0 / cfg["hz"])
        self.spec = JointSpec()
        self.home, post = classical_pose()
        self.h_open = post["open"]
        self.space = ActionSpace(cfg, self.safety, self.spec, post)
        self.planner = CEMPlanner(model, self.space, cfg, device)
        self.shadow = _ShadowFilter(self.spec, self.safety)
        self.threaded = threaded
        if threaded:
            sys.setswitchinterval(min(sys.getswitchinterval(), 0.001))
        self.time_limit = time_limit
        self.log = log or (lambda *a: None)
        self.reset()

    @classmethod
    def from_checkpoint(cls, ckpt, bank, device=None, overrides=None, **kw):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, ck = load_checkpoint(ckpt, device)
        cfg = load_config(None, overrides)
        cfg["encoder"] = ck["encoder"]                 # the predictor only works with the encoder it was trained on
        cfg["macro_ticks"] = ck["macro_ticks"]
        enc = make_encoder(cfg["encoder"], device)
        return cls(cfg, model, enc, GoalBank(cfg, bank, device=device), device, **kw)

    # ------------------------------------------------------------------ API
    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = info or {}
        self.instruction = info.get("instruction", "")
        self.time_limit = info.get("time_limit", self.time_limit)
        self.bank.reset(self.instruction)
        self.planner.reset()
        self.state = "init"
        self.queue_a = self.queue_h = None
        self.arm_cmd = self.hand_cmd = None
        self._job = None
        self.events, self.trace = [], []

    def act(self, obs):
        if self.arm_cmd is None:
            sp = self.spec
            self.arm_cmd = np.clip(np.asarray(obs["arm_q"], float), sp.arm_lower, sp.arm_upper)
            self.hand_cmd = np.clip(np.asarray(obs["hand_q"], float), sp.hand_lower, sp.hand_upper)
        t = float(obs.get("t", 0.0))
        if self.queue_a is not None and len(self.queue_a):
            return self._pop()
        if self.state == "done":
            return self._hold(done=True)
        if t > self.time_limit - float(self.cfg["policy"]["time_margin_s"]):
            self._note(t, "time limit: done")
            self.state = "done"
            return self._hold(done=True)
        if self.state == "init":
            self.state = "settle"
            if self.cfg["policy"]["home_first"]:
                a, h = self._go_to(self.home, self.h_open)
                if len(a):
                    self._note(t, f"to home ({len(a) / self.cfg['hz']:.1f} s)")
                    self._enqueue(a, h)
                    return self._pop()
        if self.state == "settle":
            n = max(1, int(round(self.cfg["policy"]["settle_s"] * self.cfg["hz"])))
            self._enqueue(np.repeat(self.arm_cmd[None], n, 0), np.repeat(self.hand_cmd[None], n, 0))
            self.state = "think"
            return self._pop()
        # think
        if self.threaded:
            if self._job is None:
                self._job = {"obs": obs, "arm": self.arm_cmd.copy(), "hand": self.hand_cmd.copy(), "result": None}
                threading.Thread(target=self._think_job, args=(self._job,), daemon=True).start()
            if self._job["result"] is None:
                return self._hold()
            res, self._job = self._job["result"], None
        else:
            res = self._think(obs, self.arm_cmd.copy(), self.hand_cmd.copy())
        return self._after_think(res, t)

    # ------------------------------------------------------------------ thinking
    def _think_job(self, job):
        try:
            job["result"] = self._think(job["obs"], job["arm"], job["hand"])
        except Exception as e:  # never kill the control loop
            job["result"] = {"error": repr(e)}

    def _think(self, obs, arm_cmd, hand_cmd):
        t0 = time.monotonic()
        imgs = {}
        for c in self.enc.cameras:
            im = np.asarray(obs[c])
            if not im.any():
                return {"wait": f"no frame yet from {c}"}
            imgs[c] = im[None]
        z = self.enc.encode(imgs)[0]
        # plan from the commanded pose (what the next chunk continues from), grip from the commanded hand
        s = self.space.state(arm_cmd, hand_cmd)
        zg, ginfo = self.bank.goal(z, torch.as_tensor(s))
        if ginfo["done"]:
            return {"done": True, "goal": ginfo}
        a, pinfo = self.planner.plan(z, s, zg)
        a = self.space.clip_action(a, arm_cmd)
        return {"action": a, "goal": ginfo, "plan": pinfo, "ms": 1e3 * (time.monotonic() - t0)}

    def _after_think(self, res, t):
        if "error" in res:
            self._note(t, f"think error {res['error']}: holding")
            return self._hold()
        if "wait" in res:
            return self._hold()
        if res.get("done"):
            self._note(t, f"goal bank: demo over ({res['goal']}) -> done")
            self.state = "done"
            return self._hold(done=True)
        a = res["action"]
        arm, hand = self._safe_chunk(a)
        self.trace.append({"t": round(t, 2), "a": np.round(a, 4).tolist(), "goal": res["goal"],
                           "energy": res["plan"]["energy"], "ms": round(res["ms"], 1), "scaled": arm is None})
        if arm is None:
            return self._hold()
        self._enqueue(arm, hand)
        return self._pop()

    def _safe_chunk(self, a):
        a = np.array(a, float)
        for _ in range(4):
            arm, hand = self.space.chunk(self.arm_cmd, self.hand_cmd, a)
            self.shadow.reset(self.arm_cmd, self.hand_cmd, 0.0)
            ok = True
            for i in range(len(arm)):
                r = self.shadow.filter({"arm_q_target": arm[i], "hand_q_target": hand[i]}, (i + 1) * self.safety.dt)
                if r.clipped:
                    ok = False
                    break
            if ok:
                return arm, hand
            a[:14] *= 0.5
        return None, None

    def _go_to(self, arm_to, hand_to):
        vmax = 0.9 * self.safety.arm_max_vel * self.safety.dt
        n = int(np.ceil(np.abs(arm_to - self.arm_cmd).max() / vmax)) if np.abs(arm_to - self.arm_cmd).max() > 1e-6 else 0
        hn = int(np.ceil(np.abs(hand_to - self.hand_cmd).max() / (0.9 * self.safety.hand_max_vel * self.safety.dt)))
        n = max(n, hn)
        if n == 0:
            return np.zeros((0, 14)), np.zeros((0, 14))
        s = (np.arange(1, n + 1) / n)[:, None]
        return self.arm_cmd + s * (arm_to - self.arm_cmd), self.hand_cmd + s * (hand_to - self.hand_cmd)

    # ------------------------------------------------------------------ action plumbing
    def _enqueue(self, a, h):
        self.queue_a, self.queue_h = np.asarray(a, float), np.asarray(h, float)

    def _pop(self):
        k = int(self.cfg["macro_ticks"])
        a, h = self.queue_a[:k], self.queue_h[:k]
        self.queue_a, self.queue_h = self.queue_a[k:], self.queue_h[k:]
        if not len(a):
            return self._hold()
        self.arm_cmd, self.hand_cmd = a[-1].copy(), h[-1].copy()
        return {"arm_q_target": a.copy(), "hand_q_target": h.copy(), "done": False}

    def _hold(self, done=False):
        k = 1 if done else max(1, int(self.cfg["policy"]["hold_ticks"]))
        return {"arm_q_target": np.repeat(self.arm_cmd[None], k, 0),
                "hand_q_target": np.repeat(self.hand_cmd[None], k, 0), "done": bool(done)}

    def _note(self, t, msg):
        self.events.append((round(t, 2), msg))
        self.log(f"[jepa t={t:6.2f}] {msg}")
