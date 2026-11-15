"""Collect sim episodes with random grasp variants + a 'held after lift' label per pick (for the grasp critic).

    python -m aaltonomous.baselines.jepa.collect_grasps --seeds 100-159 --out runs/jepa/grasps

Per episode, like collect.py: `<out>/frames/sim_seedNNNN.npz` (macro-step frames, world-model training data) and
`<out>/picks/sim_seedNNNN.npz`: for each started pick the look image, the start command (arm, hand), the planned
trajectory up to the end of its lift phase, and `held` = some active object's centre was >= `--held-z` m above the
counter at the lift end (sim scene state: privileged, training only). Sidecars carry the variants and outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .collect import _downscale, make_env, parse_seeds
from .common import ActionSpace, load_config
from .data import save_episode
from .grasp import ExploreGraspPolicy, TriangleBenchPolicy, TriangleProbePolicy


def lift_xy(pick):
    """Pelvis xy of the hand's grasp point at the lift end (planned)."""
    w = pick["plan"].waypoints
    return w["grasp"][:2]


def held_now(sc, counter_z, dz, near_xy=None, radius=0.08):
    """Some active object's centre >= dz above the counter (and, with near_xy, within `radius` in xy of it: an object
    stacked in a bin or on a rim is not 'held')."""
    objs = [o for o in sc.state()["objects"] if o.get("active")]
    if near_xy is not None:
        objs = [o for o in objs if np.hypot(o["pos"][0] - near_xy[0], o["pos"][1] - near_xy[1]) < radius]
    zs = [o["pos"][2] - counter_z for o in objs]
    return bool(zs and max(zs) >= dz), (max(zs) if zs else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=["100-109"])
    ap.add_argument("--out", default="runs/jepa/grasps")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--p-classical", type=float, default=0.3)
    ap.add_argument("--prefer", default=None, help="variant name substring to try first (e.g. h0.35: more drops)")
    ap.add_argument("--p-prefer", type=float, default=0.7)
    ap.add_argument("--triangles", action="store_true",
                    help="TriangleProbePolicy: random finger direction on triangles only, logs shape + triangle yaw")
    ap.add_argument("--bench", action="store_true",
                    help="triangle bench: 5 triangles per layout, TriangleBenchPolicy (finger-direction mode x height)")
    ap.add_argument("--held-z", type=float, default=0.06, help="object centre this far above the counter = held")
    ap.add_argument("--size", type=int, nargs=2, default=[240, 320])
    a = ap.parse_args(argv)
    cfg = load_config()
    k = int(cfg["macro_ticks"])
    env, sc = make_env("sim", a.host, a.time_limit)
    if a.bench:                                   # only triangles, 5 per layout (the episode region fits 5)
        lk = env.scene.layout_kwargs
        env.scene.layout_kwargs = lambda: {**lk(), "shapes": ["triangle"], "n_min": 5, "n_max": 5}
    space = ActionSpace(cfg, safety=env.filter.cfg)
    out = Path(a.out)
    try:
        for seed in parse_seeds(a.seeds):
            if (out / "picks" / f"sim_seed{seed:04d}.json").exists():
                continue
            env.rng = np.random.default_rng(seed)
            if a.bench:
                pol = TriangleBenchPolicy(safety=env.filter.cfg, threaded=True, time_limit=a.time_limit,
                                          log=lambda m: None, seed=seed)
            elif a.triangles:
                pol = TriangleProbePolicy(safety=env.filter.cfg, threaded=True, time_limit=a.time_limit,
                                          log=lambda m: None, seed=seed)
            else:
                pol = ExploreGraspPolicy(safety=env.filter.cfg, threaded=True, time_limit=a.time_limit,
                                         log=lambda m: None, seed=seed, p_classical=a.p_classical, prefer=a.prefer,
                                         p_prefer=a.p_prefer)
            t0 = time.monotonic()
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            rec = {"head_rgb": [], "arm_q": [], "hand_q": [], "arm_cmd": [], "hand_cmd": [], "t": []}
            queue, tick, info, n_seen, checks, labels = [], 0, {}, 0, {}, {}
            ends, delivered, n_ok_prev = {}, {}, 0
            while True:
                if tick % k == 0:
                    rec["head_rgb"].append(_downscale(np.asarray(obs["head_rgb"]), tuple(a.size)))
                    rec["arm_q"].append(np.asarray(obs["arm_q"], np.float32))
                    rec["hand_q"].append(np.asarray(obs["hand_q"], np.float32))
                    rec["arm_cmd"].append(env.filter.arm_cmd.astype(np.float32))
                    rec["hand_cmd"].append(env.filter.hand_cmd.astype(np.float32))
                    rec["t"].append(float(obs["t"]))
                if tick in checks:
                    i = checks.pop(tick)
                    pk = pol.picks[i]
                    labels[i] = held_now(sc, env.scene.counter_z, a.held_z, lift_xy(pk))
                if not queue:
                    act = pol.act(obs)
                    if len(pol.picks) > n_seen:            # a pick starts executing with this chunk
                        for i in range(n_seen, len(pol.picks)):
                            pk = pol.picks[i]
                            pk["tick0"] = tick
                            if pk["lift_end"] is not None:
                                checks[tick + pk["lift_end"]] = i
                            ends[tick + len(pk["plan"].arm)] = i
                        n_seen = len(pol.picks)
                    ar, hd = np.atleast_2d(act["arm_q_target"]), np.atleast_2d(act["hand_q_target"])
                    n = max(len(ar), len(hd))
                    ar = np.repeat(ar, n, 0) if len(ar) == 1 else ar
                    hd = np.repeat(hd, n, 0) if len(hd) == 1 else hd
                    queue = [(ar[i], hd[i], bool(act.get("done")) and i == n - 1) for i in range(n)]
                arm, hand, done = queue.pop(0)
                obs, info = env.step({"arm_q_target": arm, "hand_q_target": hand, "done": done})
                if tick in ends:   # pick's trajectory done: did the number of correctly sorted objects go up?
                    i = ends.pop(tick)
                    n_now = sum(o.get("outcome") == "correct" for o in sc.evaluate("sort_fixed")["outcomes"])
                    delivered[i] = n_now > n_ok_prev
                    n_ok_prev = n_now
                tick += 1
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            ev = sc.evaluate("sort_fixed")
            n_ok = sum(1 for o in ev["outcomes"] if o.get("outcome") == "correct")
            meta = {"source": "collect_grasps:sim", "seed": seed, "teacher": "classical+explore", "instruction": env.instruction,
                    "task": env.task, "mapping": env.mapping, "layout": env.layout, "end_reason": info.get("end_reason"),
                    "macro_ticks": k, "n_correct": n_ok, "n_objects": len(env.layout),
                    "success": n_ok == len(env.layout), "outcomes": ev["outcomes"]}
            save_episode(out / "frames" / f"sim_seed{seed:04d}", {kk: np.stack(v) for kk, v in rec.items()}, meta)
            arrays, pmeta = {}, []
            for i, pk in enumerate(pol.picks):
                le = pk["lift_end"]
                if le is None or i not in labels:
                    continue
                arrays[f"img_{i}"] = np.asarray(pk["obs"]["head_rgb"])
                arrays[f"arm0_{i}"], arrays[f"hand0_{i}"] = pk["arm0"], pk["hand0"]
                arrays[f"arm_{i}"], arrays[f"hand_{i}"] = pk["plan"].arm[:le], pk["plan"].hand[:le]
                pmeta.append({"i": i, "held": labels[i][0], "max_dz": labels[i][1], "variant": pk["variant"],
                              "delivered": delivered.get(i),
                              "color": pk["plan"].color, "side": pk["plan"].side, "target": pk["plan"].target_bin,
                              "t": pk["t"], "lift_end": le, "tick0": pk.get("tick0"),
                              "grasp_xyz": [float(x) for x in pk["plan"].waypoints["grasp"]],
                              "shape": pk.get("shape")})
            save_episode(out / "picks" / f"sim_seed{seed:04d}", arrays, {"seed": seed, "picks": pmeta, **{
                kk: meta[kk] for kk in ("n_correct", "n_objects", "outcomes")}})
            nh = sum(p["held"] for p in pmeta)
            print(f"[grasps] seed {seed}: {n_ok}/{len(env.layout)} correct, picks held {nh}/{len(pmeta)} "
                  f"delivered {sum(1 for p in pmeta if p.get('delivered'))}/{len(pmeta)} "
                  f"({', '.join(p['variant']['name'] + ('+' if p['held'] else '-') + ('D' if p.get('delivered') else '') for p in pmeta)}) "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
