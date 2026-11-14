"""Live demo: watch the JEPA grasp critic sort objects, in the browser, while it runs.

    # Isaac sim running (TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh):
    python -m aaltonomous.baselines.jepa.live_demo --critic runs/jepa/crop_critic.pt
    # no sim: the rendered mock world (kinematic grasps, no physics), still the real V-JEPA 2 critic
    python -m aaltonomous.baselines.jepa.live_demo --backend mock --critic runs/jepa/crop_critic.pt
    # then open http://localhost:8766   (remote box: ssh -L 8766:localhost:8766 <host>)

The page shows, live:
  * the head camera at ~10 fps with an overlay: target object, chosen grasp point, drop point, both hands (FK),
    the current phase of the pick;
  * every grasp decision: the classical planner's candidate grasps as the object-centred crops the V-JEPA 2 world
    model imagines from, each with its predicted p(object held after the lift), the chosen one highlighted;
  * the event feed, the running score (objects sorted, grasps held) and one row per finished episode.
Episodes run back to back on seeds `--seeds` (looping). `--policy classical` runs the stock plan with the same view
(for side-by-side comparison on a second screen). Ctrl-C stops.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PAGE = HERE / "live_demo.html"
COLORS = {"red": (230, 60, 60), "yellow": (240, 205, 40), "blue": (60, 130, 240)}


def _jpg(rgb, q=80):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


class Board:
    """Everything the page shows; written by the control loop, read by HTTP handlers."""

    def __init__(self, policy_name, backend):
        self.lock = threading.Lock()
        self.frame = b""
        self.frame_id = 0
        self.state = {"policy": policy_name, "backend": backend, "episode": None, "seed": None, "t": 0.0,
                      "phase": "starting", "instruction": "", "decision": None, "decisions_n": 0,
                      "totals": {"objects": 0, "correct": 0, "episodes": 0, "all_correct": 0, "picks": 0, "held": 0},
                      "episodes": [], "picks": []}
        self.events = deque(maxlen=60)

    def set_frame(self, jpg):
        with self.lock:
            self.frame, self.frame_id = jpg, self.frame_id + 1

    def event(self, msg):
        msg = msg.split("] ", 1)[-1] if msg.startswith("[classical") else msg   # drop the policy's log prefix
        with self.lock:
            self.events.appendleft({"wall": time.strftime("%H:%M:%S"), "msg": msg})

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["events"] = list(self.events)
            s["frame_id"] = self.frame_id
            return json.dumps(s, default=float)


def build_policy(kind, critic_path, safety, board, n_variants, margin, time_limit):
    from .grasp import JepaGraspPolicy, _VariantPolicy, make_variants, plan_variant
    log = board.event

    def record(self, det, grasp, sc, pick):
        crops = getattr(self.critic, "last", {}).get("crops") if hasattr(self, "critic") else None
        cands = []
        for j, p in enumerate(grasp):
            c = {"name": p.variant["name"], "p": None if sc is None else float(sc[j]), "chosen": j == pick,
                 "dur": round(p.duration, 1)}
            if crops is not None:
                c["img"] = base64.b64encode(_jpg(crops[j], 85)).decode()
            cands.append(c)
        board.update(decision={"color": det.color, "t": float(self._look_obs.get("t", 0.0)), "cands": cands,
                               "uv": None}, decisions_n=board.state["decisions_n"] + 1)

    if kind == "critic":
        from .crop_critic import CropGraspCritic
        critic = CropGraspCritic(critic_path)

        class Demo(JepaGraspPolicy):
            def choose(self, det, arm_q, hand_q, obstacles):
                self.critic.last = {}
                p = super().choose(det, arm_q, hand_q, obstacles)
                grasp = [q for q in getattr(self, "_cands", [])]
                if p is not None and getattr(p, "scores", None) is not None and grasp:
                    record(self, det, grasp, np.asarray(p.scores), next(j for j, q in enumerate(grasp) if q is p))
                elif p is not None and p.ok:
                    p.variant = getattr(p, "variant", {"name": "classical"})
                    record(self, det, [p], None, 0)
                return p

        pol = Demo(safety=safety, threaded=True, time_limit=time_limit, critic=critic, n_variants=n_variants,
                   margin=margin, log=log)
    else:
        class Demo(_VariantPolicy):
            def choose(self, det, arm_q, hand_q, obstacles):
                p = plan_variant(self._inner, det, make_variants(det, None, 1)[0], arm_q, hand_q, obstacles)
                p.variant = {"name": "classical"}
                if p.ok:
                    record(self, det, [p], None, 0)
                return p

        pol = Demo(safety=safety, threaded=True, time_limit=time_limit, log=log)
    return pol


class Overlay:
    def __init__(self):
        from aaltonomous.baselines.classical.config import load_config as lc
        from aaltonomous.baselines.classical.kinematics import ArmKinematics
        from aaltonomous.interface.calibration import Calibration
        self.kin, self.cal = ArmKinematics(lc(profile="sim")), Calibration.load()

    def _uv(self, p):
        u, v, z = self.cal.project("head", p)
        return (int(u), int(v)) if z > 0 else None

    def draw(self, rgb, arm_cmd, pick, phase, t, title):
        img = np.ascontiguousarray(rgb).copy()
        if pick is not None:
            plan = pick["plan"]
            col = COLORS.get(plan.color, (255, 255, 255))
            g = self._uv(plan.waypoints.get("grasp")) if "grasp" in plan.waypoints else None
            d = self._uv(plan.waypoints.get("drop")) if "drop" in plan.waypoints else None
            if g:
                cv2.circle(img, g, 14, col, 2, cv2.LINE_AA)
                cv2.drawMarker(img, g, (255, 255, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            if g and d:
                cv2.arrowedLine(img, g, d, col, 2, cv2.LINE_AA, tipLength=0.04)
        for side, col in (("left", (80, 255, 160)), ("right", (255, 120, 220))):
            uv = self._uv(self.kin.grasp_pose(side, arm_cmd)[0])
            if uv:
                cv2.circle(img, uv, 6, col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, f"{title}  t={t:5.1f}s  {phase}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1,
                    cv2.LINE_AA)
        return img


def phase_of(pick, tick):
    if pick is None:
        return "look / plan"
    k = tick - pick.get("tick0", tick)
    for name, s, e in pick["plan"].phases:
        if s <= k < e:
            return f"{pick['plan'].color} -> {pick['plan'].target_bin}: {name}"
    return "back home / look"


def run(a, board):
    from .collect import parse_seeds
    if a.backend == "mock":
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        env, sc = make_mock_env(seed=0, realtime=True), None
    else:
        from .collect import make_env
        env, sc = make_env("sim", a.host, a.time_limit)
    pol = build_policy(a.policy, a.critic, env.filter.cfg, board, a.n_variants, a.margin, a.time_limit)
    ov = Overlay()
    seeds = parse_seeds(a.seeds)
    ep_i = 0
    try:
        while True:
            seed = seeds[ep_i % len(seeds)]
            env.rng = np.random.default_rng(seed)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            board.update(episode=ep_i, seed=seed, instruction=obs["instruction"], picks=[], decision=None)
            board.event(f"episode {ep_i} (seed {seed}): {len(env.layout)} objects")
            tick, n_seen, checks, held, info = 0, 0, {}, {}, {}
            while True:
                act = pol.act(obs)
                if len(pol.picks) > n_seen:
                    for i in range(n_seen, len(pol.picks)):
                        pol.picks[i]["tick0"] = tick
                        if pol.picks[i]["lift_end"] is not None and sc is not None:
                            checks[tick + pol.picks[i]["lift_end"]] = i
                    n_seen = len(pol.picks)
                ar, hd = np.atleast_2d(act["arm_q_target"]), np.atleast_2d(act["hand_q_target"])
                n = max(len(ar), len(hd))
                for j in range(n):
                    obs, info = env.step({"arm_q_target": ar[min(j, len(ar) - 1)],
                                          "hand_q_target": hd[min(j, len(hd) - 1)],
                                          "done": bool(act.get("done")) and j == n - 1})
                    if tick in checks:
                        from .collect_grasps import held_now
                        i = checks.pop(tick)
                        held[i] = held_now(sc, env.scene.counter_z, 0.06)[0]
                        pk = pol.picks[i]
                        board.event(f"{pk['plan'].color} grasp ({pk['variant']['name']}): "
                                    f"{'HELD' if held[i] else 'slipped'} after the lift")
                        with board.lock:
                            board.state["totals"]["picks"] += 1
                            board.state["totals"]["held"] += int(held[i])
                            board.state["picks"] = [{"color": p["plan"].color, "variant": p["variant"]["name"],
                                                     "held": held.get(q)} for q, p in enumerate(pol.picks)]
                    if tick % a.frame_every == 0:
                        pick = pol.picks[-1] if pol.picks and tick - pol.picks[-1].get("tick0", 0) < len(pol.picks[-1]["plan"].arm) else None
                        ph = phase_of(pick, tick)
                        img = ov.draw(obs["head_rgb"], env.filter.arm_cmd, pick, ph, float(obs["t"]),
                                      f"ep {ep_i} seed {seed}")
                        board.set_frame(_jpg(img))
                        board.update(t=float(obs["t"]), phase=ph)
                    tick += 1
                    if info["episode_over"]:
                        break
                if info["episode_over"]:
                    break
            time.sleep(1.0 if sc is not None else 0.0)
            if sc is not None:
                outs = [o.get("outcome") for o in sc.evaluate("sort_fixed")["outcomes"]]
            else:
                outs = list(env.backend.outcomes(env.mapping))
            n_ok = sum(o == "correct" for o in outs)
            with board.lock:
                T = board.state["totals"]
                T["objects"] += len(outs)
                T["correct"] += n_ok
                T["episodes"] += 1
                T["all_correct"] += int(n_ok == len(outs))
                board.state["episodes"] = ([{"episode": ep_i, "seed": seed, "correct": n_ok, "n": len(outs),
                                             "t": round(float(info["t"]), 1), "end": info["end_reason"],
                                             "picks": len(pol.picks), "held": sum(bool(v) for v in held.values())}]
                                           + board.state["episodes"])[:30]
            board.event(f"episode {ep_i} done: {n_ok}/{len(outs)} sorted in {info['t']:.0f} s ({info['end_reason']})")
            ep_i += 1
            if a.episodes and ep_i >= a.episodes:
                board.update(phase="finished")
                break
    finally:
        env.close()


def make_handler(board):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path.startswith("/state.json"):
                return self._send(200, board.snapshot().encode(), "application/json")
            if self.path.startswith("/frame.jpg"):
                with board.lock:
                    f = board.frame
                return self._send(200, f, "image/jpeg")
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with board.lock:
                            fid, f = board.frame_id, board.frame
                        if fid != last and f:
                            last = fid
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, b"not found", "text/plain")
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sim", "mock"], default="sim")
    ap.add_argument("--policy", choices=["critic", "classical"], default="critic")
    ap.add_argument("--critic", default="runs/jepa/crop_critic.pt")
    ap.add_argument("--seeds", nargs="+", default=["0-15"])
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = loop forever)")
    ap.add_argument("--n-variants", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--frame-every", type=int, default=3, help="publish every n-th control tick (3 = 10 fps)")
    ap.add_argument("--host", default="127.0.0.1", help="sim host")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8766)))
    a = ap.parse_args(argv)
    if a.policy == "critic" and not Path(a.critic).exists():
        raise SystemExit(f"{a.critic} not found: train it (crop_critic.py) or run --policy classical")
    board = Board(a.policy, a.backend)
    srv = ThreadingHTTPServer((a.bind, a.port), make_handler(board))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[live demo] http://localhost:{a.port}  ({a.policy} on {a.backend})", flush=True)
    try:
        run(a, board)
        print("[live demo] finished; page stays up, Ctrl-C to stop", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
