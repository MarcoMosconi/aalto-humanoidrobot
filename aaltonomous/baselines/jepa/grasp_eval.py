"""Rung 1 in the Isaac sim: stock classical vs classical + JEPA grasp critic, same seeds, same harness.

    python -m aaltonomous.baselines.jepa.grasp_eval --policy jepa --critic runs/jepa/critic.pt --seeds 0-7 --log runs/jepa/eval_jepa.jsonl
    python -m aaltonomous.baselines.jepa.grasp_eval --policy classical --seeds 0-7 --log runs/jepa/eval_classical.jsonl

Like classical/sim_eval.py (AaltoEnv.reset -> policy at 30 Hz through the safety filter -> SceneClient.evaluate),
plus per pick: the chosen variant, the critic's scores and whether an object was held at the lift end (sim state).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .collect import make_env, parse_seeds
from .collect_grasps import held_now


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["classical", "jepa"], default="jepa")
    ap.add_argument("--critic", default="runs/jepa/critic.pt")
    ap.add_argument("--n-variants", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.05, help="variant must beat the stock plan's p(held) by this")
    ap.add_argument("--seeds", nargs="+", default=["0-7"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    env, sc = make_env("sim", a.host, a.time_limit)
    from .grasp import JepaGraspPolicy, _VariantPolicy, make_variants, plan_variant

    if a.policy == "jepa":
        cp = __import__("torch").load(a.critic, map_location="cpu", weights_only=False)
        if "half_full" in cp:                         # crop_critic.py output
            from .crop_critic import CropGraspCritic
            critic = CropGraspCritic(a.critic)
        else:
            from .train_critic import load_critic
            critic = load_critic(a.critic)
        pol = JepaGraspPolicy(safety=env.filter.cfg, threaded=True, time_limit=a.time_limit, critic=critic,
                              n_variants=a.n_variants, margin=a.margin)
    else:
        class Stock(_VariantPolicy):     # stock classical plan, but logged the same way
            def choose(self, det, arm_q, hand_q, obstacles):
                p = plan_variant(self._inner, det, make_variants(det, None, 1)[0], arm_q, hand_q, obstacles)
                p.variant = {"name": "classical"}
                return p
        pol = Stock(safety=env.filter.cfg, threaded=True, time_limit=a.time_limit)
    rows = []
    try:
        for seed in parse_seeds(a.seeds):
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            clips, info, tick, n_seen, checks, held = [], {}, 0, 0, {}, {}
            while True:
                act = pol.act(obs)
                if len(pol.picks) > n_seen:
                    for i in range(n_seen, len(pol.picks)):
                        if pol.picks[i]["lift_end"] is not None:
                            checks[tick + pol.picks[i]["lift_end"]] = i
                    n_seen = len(pol.picks)
                ar, hd = np.atleast_2d(act["arm_q_target"]), np.atleast_2d(act["hand_q_target"])
                n = max(len(ar), len(hd))
                for j in range(n):
                    obs, info = env.step({"arm_q_target": ar[min(j, len(ar) - 1)], "hand_q_target": hd[min(j, len(hd) - 1)],
                                          "done": bool(act.get("done")) and j == n - 1})
                    if tick in checks:
                        held[checks.pop(tick)] = held_now(sc, env.scene.counter_z, 0.06)[0]
                    tick += 1
                    if info["clipped"]:
                        clips.append((round(info["t"], 2), info["clip_reasons"]))
                    if info["episode_over"]:
                        break
                if info["episode_over"]:
                    break
            time.sleep(1.0)
            ev = sc.evaluate("sort_fixed")
            n_ok = sum(1 for o in ev["outcomes"] if o.get("outcome") == "correct")
            picks = [{"variant": p["variant"]["name"], "scores": p["scores"], "held": held.get(i), "color": p["plan"].color}
                     for i, p in enumerate(pol.picks)]
            row = {"policy": a.policy, "seed": seed, "n_objects": len(env.layout), "correct": n_ok,
                   "end": info["end_reason"], "t": round(info["t"], 1), "clip_ticks": len(clips), "picks": picks,
                   "outcomes": ev["outcomes"], "events": pol.events}
            rows.append(row)
            nh = sum(1 for p in picks if p["held"])
            print(f"### {a.policy} seed {seed}: {n_ok}/{len(env.layout)} correct, picks held {nh}/{len(picks)}, "
                  f"end={info['end_reason']} t={info['t']:.0f}s clips={len(clips)} "
                  f"[{', '.join(p['variant'] + ('+' if p['held'] else '-') for p in picks)}]", flush=True)
            if a.log:
                with open(a.log, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    finally:
        env.close()
    tot = sum(r["n_objects"] for r in rows)
    ph = [p["held"] for r in rows for p in r["picks"] if p["held"] is not None]
    print(f"\nTOTAL {a.policy}: {sum(r['correct'] for r in rows)}/{tot} correct, "
          f"{sum(r['correct'] == r['n_objects'] for r in rows)}/{len(rows)} episodes all-correct, "
          f"picks held {sum(ph)}/{len(ph)}, clip ticks {sum(r['clip_ticks'] for r in rows)}")
    return rows


if __name__ == "__main__":
    main()
