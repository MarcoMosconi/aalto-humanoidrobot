"""Run the JEPA policy in-process on the Aalto Isaac sim (or the rendered mock) and score each episode.

    python -m aaltonomous.baselines.jepa.sim_eval --ckpt runs/jepa/ckpt/best.pt --bank runs/jepa/latents --seeds 0 1 2
    python -m aaltonomous.baselines.jepa.sim_eval --backend mock --ckpt ... --bank ... --seeds 0

Same flow as classical/sim_eval.py: AaltoEnv.reset() places a layout -> policy at 30 Hz through the safety filter ->
SceneClient.evaluate("sort_fixed") (mock: the mock world's colour rule). `--log` appends one json line per episode,
incl. the planner trace (goal match, CEM energy, think time per macro step).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/jepa/ckpt/best.pt")
    ap.add_argument("--bank", default="runs/jepa/latents")
    ap.add_argument("--backend", choices=["sim", "mock"], default="sim")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--set", default=None, help='json overrides, e.g. \'{"planner.samples": 512}\'')
    ap.add_argument("--log", default=None)
    a = ap.parse_args(argv)

    from .collect import make_env
    from .policy import JepaPolicy
    env, sc = make_env(a.backend, a.host, a.time_limit)
    pol = JepaPolicy.from_checkpoint(a.ckpt, a.bank, overrides=json.loads(a.set) if a.set else None,
                                     safety=env.filter.cfg, threaded=(a.backend == "sim"), time_limit=a.time_limit)
    rows = []
    try:
        for seed in a.seeds:
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            clips, info = [], {}
            while True:
                obs, info = env.step(pol.act(obs))
                if info["clipped"]:
                    clips.append((round(info["t"], 2), info["clip_reasons"]))
                if info["episode_over"]:
                    break
            if sc is not None:
                time.sleep(1.0)
                outs = sc.evaluate("sort_fixed")["outcomes"]
            else:
                outs = [{"outcome": o} for o in env.backend.outcomes(env.mapping)]
            n_ok = sum(1 for o in outs if o.get("outcome") == "correct")
            ms = [s["ms"] for s in pol.trace]
            row = {"seed": seed, "n_objects": len(env.layout), "correct": n_ok, "end": info["end_reason"],
                   "t": round(info["t"], 1), "clip_ticks": len(clips), "outcomes": outs, "events": pol.events,
                   "think_ms_mean": round(float(np.mean(ms)), 1) if ms else None, "trace": pol.trace}
            rows.append(row)
            print(f"\n### seed {seed}: {n_ok}/{len(env.layout)} correct, end={info['end_reason']} t={info['t']:.0f}s "
                  f"clip_ticks={len(clips)} macro steps={len(pol.trace)} think={row['think_ms_mean']} ms")
            if a.log:
                with open(a.log, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    finally:
        env.close()
    tot = sum(r["n_objects"] for r in rows)
    print(f"\nTOTAL {sum(r['correct'] for r in rows)}/{tot} correct over seeds {a.seeds}, "
          f"clip ticks {sum(r['clip_ticks'] for r in rows)}")
    return rows


if __name__ == "__main__":
    main()
