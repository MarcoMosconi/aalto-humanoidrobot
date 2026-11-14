"""Collect world-model training episodes by rolling out a teacher policy (default: the classical baseline, unchanged).

    # Isaac sim running (TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh), DR on for variety:
    python -m aaltonomous.baselines.jepa.collect --backend sim --seeds 0-49 --out runs/jepa/episodes
    # no sim: the rendered mock world (kinematic, no physics; plumbing / smoke tests only)
    python -m aaltonomous.baselines.jepa.collect --backend mock --seeds 0-3 --out runs/jepa/episodes_mock

The env is stepped one tick at a time (the teacher's chunks are unrolled here), and every `macro_ticks` ticks the
images, measured joints and the filtered command are stored (data.py format). Sim episodes are scored with the scene
service so the goal bank can keep successful ones only. `--explore` adds random arm/grip perturbations on some
macro steps (more varied actions than the teacher alone gives the world model; off by default).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .common import ActionSpace, load_config
from .data import save_episode


def parse_seeds(spec):
    out = []
    for part in spec:
        if "-" in str(part):
            a, b = str(part).split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _downscale(img, size):
    if size is None or img.shape[:2] == tuple(size):
        return img
    import cv2
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def make_env(backend, host, time_limit, seed=0):
    if backend == "mock":
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        return make_mock_env(seed=seed), None
    import ssl  # noqa: F401  before DDS (see sim/evaluate.py)
    from aaltonomous.sim.evaluate import apply_instance_env
    apply_instance_env()
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=seed, time_limit=time_limit, img_server_ip=host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    return env, SceneClient(host)


def run_episode(env, teacher, space, cfg, cameras, size, max_ticks=None, explore=0.0, rng=None):
    k = int(cfg["macro_ticks"])
    obs = env.reset()
    teacher.reset({"instruction": obs["instruction"]})
    rec = {c: [] for c in cameras}
    rec.update(arm_q=[], hand_q=[], arm_cmd=[], hand_cmd=[], t=[])
    queue, tick, info = [], 0, {}
    pert = None
    while True:
        if tick % k == 0:
            for c in cameras:
                rec[c].append(_downscale(np.asarray(obs[c]), size))
            rec["arm_q"].append(np.asarray(obs["arm_q"], np.float32))
            rec["hand_q"].append(np.asarray(obs["hand_q"], np.float32))
            rec["arm_cmd"].append(env.filter.arm_cmd.astype(np.float32))
            rec["hand_cmd"].append(env.filter.hand_cmd.astype(np.float32))
            rec["t"].append(float(obs.get("t", tick / cfg["hz"])))
            pert = None
            if explore > 0 and rng is not None and rng.random() < explore:  # small random detour, then back
                pert = rng.normal(0, 0.3 * space.max_step, 14)
        if not queue:
            act = teacher.act(obs)
            a = np.atleast_2d(act["arm_q_target"])
            h = np.atleast_2d(act["hand_q_target"])
            n = max(len(a), len(h))
            a = np.repeat(a, n, 0) if len(a) == 1 else a
            h = np.repeat(h, n, 0) if len(h) == 1 else h
            queue = [(a[i], h[i], bool(act.get("done")) and i == n - 1) for i in range(n)]
        arm, hand, done = queue.pop(0)
        if pert is not None:
            frac = 1.0 - abs((tick % k) - k / 2) / (k / 2)   # 0 -> 1 -> 0 over the macro step
            arm = np.clip(arm + frac * pert, space.arm_lo, space.arm_hi)
        obs, info = env.step({"arm_q_target": arm, "hand_q_target": hand, "done": done})
        tick += 1
        if info["episode_over"] or (max_ticks is not None and tick >= max_ticks):
            break
    arrays = {key: np.stack(v) for key, v in rec.items() if len(v)}
    return arrays, info, obs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "mock"], default="sim")
    ap.add_argument("--seeds", nargs="+", default=["0-9"])
    ap.add_argument("--out", default="runs/jepa/episodes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--max-ticks", type=int, default=None, help="stop each episode after this many ticks")
    ap.add_argument("--cameras", nargs="+", default=["head_rgb"])
    ap.add_argument("--size", type=int, nargs=2, default=[240, 320], help="stored image H W (0 0 = full res)")
    ap.add_argument("--explore", type=float, default=0.0, help="probability of a random detour per macro step")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    size = None if a.size == [0, 0] else tuple(a.size)
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    env, sc = make_env(a.backend, a.host, a.time_limit)
    space = ActionSpace(cfg, safety=env.filter.cfg)
    teacher = ClassicalPolicy(safety=env.filter.cfg, threaded=(a.backend == "sim"), time_limit=a.time_limit,
                              log=lambda m: None)
    rng = np.random.default_rng(12345)
    try:
        for seed in parse_seeds(a.seeds):
            env.rng = np.random.default_rng(seed)
            t0 = time.monotonic()
            arrays, info, _ = run_episode(env, teacher, space, cfg, a.cameras, size, a.max_ticks, a.explore, rng)
            meta = {"source": f"collect:{a.backend}", "seed": seed, "teacher": "classical", "instruction": env.instruction,
                    "task": env.task, "mapping": env.mapping, "layout": env.layout, "end_reason": info.get("end_reason"),
                    "macro_ticks": cfg["macro_ticks"], "explore": a.explore}
            if sc is not None:
                time.sleep(1.0)
                ev = sc.evaluate("sort_fixed")
                n_ok = sum(1 for o in ev["outcomes"] if o.get("outcome") == "correct")
                meta.update(n_correct=n_ok, n_objects=len(env.layout), success=n_ok == len(env.layout) and n_ok > 0,
                            outcomes=ev["outcomes"])
            elif hasattr(env.backend, "outcomes"):
                outs = env.backend.outcomes(env.mapping)
                n_ok = sum(1 for o in outs if o == "correct")
                meta.update(n_correct=n_ok, n_objects=len(env.layout), success=n_ok == len(env.layout) and n_ok > 0)
            name = f"{a.backend}_seed{seed:04d}"
            save_episode(os.path.join(a.out, name), arrays, meta)
            print(f"[collect] seed {seed}: {len(arrays['t'])} samples, {meta.get('n_correct')}/{meta.get('n_objects')} "
                  f"correct, end={info.get('end_reason')} ({time.monotonic() - t0:.0f} s) -> {a.out}/{name}.npz")
    finally:
        env.close()


if __name__ == "__main__":
    main()
