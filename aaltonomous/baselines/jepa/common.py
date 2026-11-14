"""Shared bits of the JEPA policy: config loading and the planning action space.

Planning space per macro step (`macro_ticks` control ticks):
    action = [delta arm_q (14), grip_left, grip_right]          (16,)
    state  = [arm_q (14), grip_left, grip_right]                (16,)
A grip of 0 is the `open` Dex3 posture and 1 the `closed` one (classical config.yaml, mirrored for the right hand);
the hand target is the straight line between them. Real/teleop hand commands are projected onto that line, so demos
with other finger shapes still give a usable grip signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from aaltonomous.interface.safety import JointSpec, SafetyConfig

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
A_DIM = 16


def _set(cfg, key, val):
    d = cfg
    ks = key.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = val


def load_config(path=None, overrides=None):
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for k, v in (overrides or {}).items():
        _set(cfg, k, v)
    return cfg


def classical_pose(profile="sim"):
    """(home arm (14,), {open, closed, ...} hand postures (14,)) from the classical baseline's config (read only)."""
    from aaltonomous.baselines.classical.config import hand_postures, home_arm
    from aaltonomous.baselines.classical.config import load_config as load_classical
    ccfg = load_classical(profile=profile)
    spec = JointSpec()
    return home_arm(ccfg, spec), hand_postures(ccfg, spec)


class ActionSpace:
    """Converts between joint targets and the 16-d planning space, with the per-step bounds."""

    def __init__(self, cfg, safety: SafetyConfig | None = None, spec: JointSpec | None = None, postures=None):
        self.cfg = cfg
        self.spec = spec or JointSpec()
        self.safety = safety or SafetyConfig(dt=1.0 / cfg["hz"])
        self.k = int(cfg["macro_ticks"])
        if postures is None:
            _, postures = classical_pose()
        self.h_open = np.asarray(postures["open"], float)
        self.h_closed = np.asarray(postures["closed"], float)
        self.max_step = float(cfg["action"]["arm_step_frac"]) * self.safety.arm_max_vel * self.k / float(cfg["hz"])
        m = float(cfg["action"]["joint_margin"])
        self.arm_lo, self.arm_hi = self.spec.arm_lower + m, self.spec.arm_upper - m

    # ------------------------------------------------------------- hands
    def grip_of(self, hand_q):
        """(…,14) hand vector -> (…,2) grip in [0,1] per hand (projection on the open->closed line)."""
        h = np.asarray(hand_q, float)
        out = []
        for sl in (slice(0, 7), slice(7, 14)):
            d = self.h_closed[sl] - self.h_open[sl]
            out.append(((h[..., sl] - self.h_open[sl]) @ d) / max(float(d @ d), 1e-9))
        return np.clip(np.stack(out, -1), 0.0, 1.0)

    def hand_of(self, grip):
        g = np.clip(np.asarray(grip, float), 0.0, 1.0)
        left = self.h_open[:7] + g[..., :1] * (self.h_closed[:7] - self.h_open[:7])
        right = self.h_open[7:] + g[..., 1:2] * (self.h_closed[7:] - self.h_open[7:])
        return np.concatenate([left, right], -1)

    # ------------------------------------------------------------- state / action
    def state(self, arm_q, hand_q):
        return np.concatenate([np.asarray(arm_q, float), self.grip_of(hand_q)], -1).astype(np.float32)

    def actions_from_commands(self, arm_cmd, hand_cmd):
        """Per-macro-step actions from commanded targets sampled every macro step: (N,14),(N,14) -> (N-1,16)."""
        d_arm = np.diff(np.asarray(arm_cmd, float), axis=0)
        grip = self.grip_of(hand_cmd)[1:]
        return np.concatenate([d_arm, grip], -1).astype(np.float32)

    def clip_action(self, a, arm_now):
        """Bound the arm step and keep arm_now + step inside the joint limits (margin); grip into [0,1]."""
        a = np.array(a, float)
        a[..., :14] = np.clip(a[..., :14], -self.max_step, self.max_step)
        tgt = np.clip(np.asarray(arm_now)[..., :14] + a[..., :14], self.arm_lo, self.arm_hi)
        a[..., :14] = tgt - np.asarray(arm_now)[..., :14]
        a[..., 14:] = np.clip(a[..., 14:], 0.0, 1.0)
        return a

    def chunk(self, arm_from, hand_from, a):
        """One planned macro action -> (k,14) arm + (k,14) hand targets, linear in time, within the velocity caps."""
        k = self.k
        s = (np.arange(1, k + 1) / k)[:, None]
        arm = np.asarray(arm_from, float)[None] + s * np.asarray(a[:14], float)[None]
        hand_to = self.hand_of(a[14:16])
        hand_from = np.asarray(hand_from, float)
        dh = self.safety.hand_max_vel * self.safety.dt * 0.95  # fingers: move at the cap, then hold
        hand = np.empty((k, 14))
        h = hand_from.copy()
        for i in range(k):
            h = h + np.clip(hand_to - h, -dh, dh)
            hand[i] = h
        return arm, hand
