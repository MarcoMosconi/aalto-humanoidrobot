"""Interactive viewer: the classical baseline's algorithms run live on recorded data, in the browser.

    bash aaltonomous/cloud/aalto.sh py -m aaltonomous.baselines.classical.viz_demo \
        [--frames runs/perception_eval_raw] [--episodes runs/eval] [--port 8765]
    # then open http://localhost:8765  (remote box: ssh -L 8765:localhost:8765 <host>)

No sim or GPU needed. Two views:

Perception  Every saved head RGB-D frame (perception_eval --raw) goes through the real `Perception` code, stage by
            stage: HSV masks -> 3D keep-mask (counter / bins / hands removed) -> blobs + top-face pixels -> pelvis-frame
            detections, matched to sim ground truth with perception_eval's own matcher. Every perception parameter
            (HSV ranges, morphology, top band, xy_offset, ...) is editable; the frame and the whole-set metrics
            re-run on each change, next to the stock config.
Episodes    Evaluator episodes (sim/evaluate .npz + .jsonl): measured arm joints -> Pinocchio FK of shoulder, elbow,
            wrist, palm, grasp point and fingertips, replayed top-down and from the side with hand closure, command
            tracking error and safety clips on a timeline, objects at start and end.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .perception_eval import classify_miss, frame_row, overlay, summarize

HERE = Path(__file__).resolve().parent
PAGE = HERE / "viz_demo.html"
RGB = {"red": (230, 50, 50), "yellow": (240, 210, 30), "blue": (40, 120, 235)}
ARM_CHAIN = ("shoulder_pitch_link", "elbow_link", "wrist_roll_link", "wrist_yaw_link", "hand_palm_link")


def _b64(img_bgr, ext=".jpg", q=88):
    import cv2
    params = [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    return f"data:image/{'jpeg' if ext == '.jpg' else 'png'};base64," + base64.b64encode(buf).decode()


def _finite(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None (browsers reject the NaN token)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _r(a, nd=4):
    return np.round(np.asarray(a, float), nd).tolist()


class Engine:
    def __init__(self, frames_dir, episodes_dir):
        from aaltonomous.interface.calibration import Calibration
        from aaltonomous.scene.scene import Scene
        self.scene, self.cal = Scene.load(), Calibration.load()
        self.frames = {f.stem: f for f in sorted(Path(frames_dir).glob("*.npz"))} if frames_dir else {}
        self.episodes = {}
        if episodes_dir:
            for f in sorted(Path(episodes_dir).glob("**/episode_*.npz")):
                if f.with_suffix(".jsonl").exists():
                    self.episodes[f"{f.parent.name}/{f.stem}"] = f
        self._kin = None
        self._lock = threading.Lock()   # Perception/Pinocchio objects are not shared across threads

    # ------------------------------------------------------------------ shared
    def cfg(self, overrides=None):
        from .config import load_config
        return load_config(None, overrides or None, None)

    def perception(self, overrides=None):
        from .perception import Perception
        return Perception(self.cfg(overrides), self.scene, self.cal)

    @lru_cache(maxsize=64)
    def frame(self, name):
        z = np.load(self.frames[name])
        return z["rgb"], z["depth"], json.loads(str(z["gt"])), json.loads(str(z["layout"]))

    def geometry(self):
        sc, cam = self.scene, self.perception().cam
        c = self.cal.cams[cam]
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2
        bins = {n: _r(sc.mat_to_pelvis(np.array([[cx + sx * half[0], cy + sy * half[1]]
                                                 for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))])), 4)
                for n, (cx, cy) in b["centers"].items()}
        W, D = sc.cfg["counter"]["mat_size"]
        mat = _r(sc.mat_to_pelvis(np.array([[0, 0], [0, D], [W, D], [W, 0]])), 4)
        return {"bins": bins, "mat": mat, "counter_z": sc.counter_z, "size": sc.size,
                "object_height": sc.object_height(), "K": c.K.tolist(), "T": self.cal._T(cam).tolist(),
                "depth_scale": c.depth_scale, "width": c.width, "height": c.height,
                "wall_height": float(b["wall_height"])}

    def index(self):
        cfg = self.cfg()
        return {"frames": list(self.frames), "episodes": list(self.episodes), "geometry": self.geometry(),
                "perception": cfg["perception"], "profile": cfg["profile"]}

    # ------------------------------------------------------------------ perception
    def trace(self, per, rgb, depth):
        """Same steps as Perception.detect, keeping the intermediates for drawing."""
        import cv2
        c, sc = per.cfg, self.scene
        H, W = depth.shape
        dlo, dhi = c["depth_range_mm"]
        raw = per.masks(rgb)
        keep = per._keep_mask(depth)
        # why each pixel was dropped (re-derived from _keep_mask's tests): 1 counter/too low, 2 too high, 3 bin
        why = np.zeros((H, W), np.uint8)
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() >= c["min_valid_frac"]:
            vv, uu = np.nonzero(valid)
            P = self.cal.pixel_to_pelvis(per.cam, uu, vv, d[vv, uu])
            h = (P[:, 2] - sc.counter_z) / per.height
            pm = sc.pelvis_to_mat(P)
            half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
            inbin = np.zeros(len(P), bool)
            for ctr in sc.cfg["bins"]["centers"].values():
                inbin |= np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
            code = np.where(inbin, 3, np.where(h < 0.25, 1, np.where(h > c["height_range"][1], 2, 0)))
            why[vv, uu] = code
        else:
            why[keep == 0] = 3
        blobs = []
        k = int(c["open_px"])
        for color, m0 in raw.items():
            m = m0 & keep
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                blob = (lab == i).astype(np.uint8)
                info = {"color": color, "area": area, "bbox": stats[i, :4].tolist()}
                if area < c["min_area_px"]:
                    info["status"] = f"small ({area} px)"
                    blobs.append((info, blob, None, None))
                    continue
                e = int(c["erode_px"])
                core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                if core.sum() < 0.25 * area:
                    core = blob
                v, u = np.nonzero(core)
                dd = depth[v, u].astype(float)
                ok = (dd >= dlo) & (dd <= dhi)
                top_uv = None
                det = None
                if ok.sum() >= c["min_valid_depth_px"]:
                    det = per._from_depth(color, u[ok], v[ok], dd[ok], area)
                    P = self.cal.pixel_to_pelvis(per.cam, u[ok], v[ok], dd[ok])
                    z_top = np.percentile(P[:, 2], c["top_percentile"])
                    sel = P[:, 2] >= z_top - c["top_band"]
                    top_uv = (u[ok][sel], v[ok][sel], P[sel])
                if det is None:
                    det = per._from_plane(color, cent[i], area, int(ok.sum()))
                rej = per._reject(det)
                info.update(status=rej or "detection", method=det.method, n_core=int(core.sum()),
                            n_depth=int(ok.sum()))
                if np.all(np.isfinite(det.top)):
                    info["top"] = _r(det.top)
                info["uv"] = _r(det.uv, 1)
                blobs.append((info, blob, core, top_uv))
        return raw, keep, why, blobs

    def run_frame(self, name, overrides):
        import cv2
        rgb, depth, gt, lay = self.frame(name)
        scen, i = name.rsplit("_", 1)
        with self._lock:
            per = self.perception(overrides)
            row, pairs, missed, fps, rejected = frame_row(per, self.scene, rgb, depth, gt, scen, int(i), lay, True)
            raw, keep, why, blobs = self.trace(per, rgb, depth)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        imgs = {"rgb": _b64(bgr)}

        # depth: turbo over the valid range; also lossless hi/lo bytes for hover read-out in the browser
        d = depth.astype(float)
        dlo, dhi = per.cfg["depth_range_mm"]
        valid = (d >= dlo) & (d <= dhi)
        lo, hi = (np.percentile(d[valid], [1, 99]) if valid.any() else (dlo, dhi))
        dn = np.clip((d - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        dc = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
        dc[~valid] = 0
        imgs["depth"] = _b64(dc)
        packed = np.dstack([np.zeros_like(depth, np.uint8), (depth & 0xFF).astype(np.uint8),
                            (depth >> 8).astype(np.uint8)])          # BGR order -> R=hi, G=lo
        imgs["depth_raw"] = _b64(packed, ".png")

        # 1 HSV masks
        m_img = (gray * 0.3).astype(np.uint8)
        for color, m in raw.items():
            m_img[m > 0] = RGB[color][::-1]
        imgs["masks"] = _b64(m_img)

        # 2 keep mask: dropped pixels tinted by reason
        k_img = bgr.copy()
        tint = {1: (25, 25, 25), 2: (200, 60, 200), 3: (0, 140, 255)}
        for code, col in tint.items():
            sel = (why == code) & (keep == 0)
            k_img[sel] = (0.35 * k_img[sel] + 0.65 * np.array(col)).astype(np.uint8)
        imgs["keep"] = _b64(k_img)

        # 3 blobs: fill, eroded core outline, top-face pixels, status
        b_img = (gray * 0.45).astype(np.uint8)
        top_xy = []
        for info, blob, core, top in blobs:
            col = np.array(RGB[info["color"]][::-1], float)
            ok = info["status"] == "detection"
            b_img[blob > 0] = (0.35 * b_img[blob > 0] + 0.65 * (col if ok else col * 0.45)).astype(np.uint8)
            cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(b_img, cs, -1, tuple(int(x) for x in col), 1)
            if core is not None:
                cs, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(b_img, cs, -1, (255, 255, 255), 1)
            if top is not None:
                b_img[top[1], top[0]] = (255, 255, 255)
                info["top_px"] = int(len(top[0]))
                step = max(1, len(top[2]) // 300)
                top_xy.append({"color": info["color"], "pts": _r(top[2][::step, :2], 4)})
            x, y, w, h = info["bbox"]
            if not ok and info["area"] >= 30:
                cv2.putText(b_img, info["status"][:16], (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                            (180, 180, 180), 1)
            if ok:
                u, v = int(info["uv"][0]), int(info["uv"][1])
                cv2.circle(b_img, (u, v), 3, (0, 255, 0), -1)
        imgs["blobs"] = _b64(b_img)

        why_s = ",".join(m["why"] for m in row["missed"])
        title = f"{name}: {row['n_matched']}/{row['n_gt']} found, {len(fps)} FP" + (f"  miss: {why_s}" if why_s else "")
        imgs["result"] = _b64(overlay(rgb, per, per.cal, per.cam, pairs, missed, fps, rejected, title))

        # top-down point cloud (every 4th pixel), pelvis frame mm + rgb
        vv, uu = np.nonzero(valid[::4, ::4])
        vv, uu = vv * 4, uu * 4
        P = self.cal.pixel_to_pelvis(per.cam, uu, vv, d[vv, uu])
        near = (P[:, 0] > -0.1) & (P[:, 0] < 1.2) & (np.abs(P[:, 1]) < 0.9)
        cloud = {"xyz": np.round(P[near] * 1000).astype(int).ravel().tolist(),
                 "rgb": rgb[vv[near], uu[near]].ravel().tolist()}

        dets = [{"color": d_.color, "top": _r(d_.top), "method": d_.method, "n_px": d_.n_px,
                 "yaw_deg": None if d_.yaw is None else round(float(np.degrees(d_.yaw)), 1)} for d_ in
                (p[1] for p in pairs)]
        fp = [{"color": d_.color, "top": _r(d_.top), "method": d_.method} for d_ in fps]
        return {"name": name, "row": row, "images": imgs, "cloud": cloud, "top_faces": top_xy,
                "blobs": [b[0] for b in blobs if b[0]["area"] >= 30], "detections": dets, "false_pos": fp,
                "pairs": [{"gt": g["name"], "det": _r(d_.top)} for g, d_ in pairs]}

    def run_all(self, overrides):
        rows = []
        with self._lock:
            per = self.perception(overrides)
            t = time.perf_counter()
            for name in self.frames:
                rgb, depth, gt, lay = self.frame(name)
                scen, i = name.rsplit("_", 1)
                rows.append(frame_row(per, self.scene, rgb, depth, gt, scen, int(i), lay, True)[0])
        scens = list(dict.fromkeys(r["scenario"] for r in rows))
        per_frame = {f"{r['scenario']}_{r['i']:02d}": {"n_gt": r["n_gt"], "n": r["n_matched"],
                                                       "fp": len(r["false_pos"]),
                                                       "xy": [m["err_xy_mm"] for m in r["matches"]]} for r in rows}
        return {"overall": summarize(rows), "by_scenario": {s: summarize([r for r in rows if r["scenario"] == s])
                                                           for s in scens},
                "frames": per_frame, "seconds": round(time.perf_counter() - t, 2)}

    # ------------------------------------------------------------------ episodes
    @property
    def kin(self):
        if self._kin is None:
            from .kinematics import ArmKinematics
            self._kin = ArmKinematics(self.cfg())
        return self._kin

    def _closure(self, q7, sl):
        """Hand posture as a fraction from config `open` (0) to `closed` (1); < 0 = opened wider (release)."""
        from .config import hand_postures
        hp = hand_postures(self.cfg())
        o, c = hp["open"][sl], hp["closed"][sl]
        return (q7 - o) @ (c - o) / ((c - o) @ (c - o))

    @lru_cache(maxsize=8)
    def run_episode(self, name, stride=2):
        f = self.episodes[name]
        z = np.load(f)
        summ = json.loads(f.with_suffix(".jsonl").read_text().splitlines()[0])["summary"]
        kin = self.kin
        fid = {s: [kin.model.getFrameId(f"{s}_{n}") for n in ARM_CHAIN] for s in ("left", "right")}
        idx = np.arange(0, len(z["t"]), stride)
        chain = {s: [] for s in fid}
        hand = {s: [] for s in fid}
        with self._lock:
            for t in idx:
                for s in ("left", "right"):
                    hp = kin.hand_points(s, z["arm_q"][t].astype(float), z["hand_q"][t].astype(float))
                    links = np.array([kin.data.oMf[j].translation for j in fid[s]])
                    chain[s].append(np.vstack([links, hp[1:2]]))      # shoulder..palm, grasp point
                    hand[s].append(hp[5:8])                             # fingertips
        sc = self.scene
        objs0 = []
        for o in summ.get("layout") or []:
            p = sc.mat_to_pelvis(sc.cube_spawn_mat(o))
            objs0.append({"color": o["color"], "shape": o.get("shape", "cube"), "pos": _r(p),
                          "yaw": round(float(np.pi / 2 - o.get("yaw", 0.0)), 3)})
        outc = {o["name"]: o["outcome"] for o in summ.get("outcomes") or []}
        objs1 = [{"name": o["name"], "color": o["name"].split("_")[1], "shape": o["name"].split("_")[2],
                  "pos": _r(o["pos"]), "outcome": outc.get(o["name"])} for o in summ.get("final_objects") or []]
        hq = z["hand_q"][idx].astype(float)
        track = np.abs(z["arm_q_cmd"][idx] - z["arm_q"][idx])
        keep = ("episode", "seed", "instruction", "end_reason", "ticks", "sim_s", "score", "max_score",
                "n_correct", "n_objects", "all_correct", "mapping", "safety", "latency_ms_mean", "command_hz")
        return {"name": name, "summary": {k: summ.get(k) for k in keep}, "t": _r(z["t"][idx], 3),
                "chain": {s: _r(np.array(v), 4) for s, v in chain.items()},
                "tips": {s: _r(np.array(v), 4) for s, v in hand.items()},
                "closure": {s: _r(self._closure(hq[:, sl], sl), 3)
                            for s, sl in (("left", slice(0, 7)), ("right", slice(7, 14)))},
                "track_err": {"left": _r(track[:, :7].max(1), 4), "right": _r(track[:, 7:].max(1), 4)},
                "arm_q": _r(z["arm_q"][idx], 3), "clipped": z["clipped"][idx].astype(int).tolist(),
                "hold": z["hold"][idx].astype(int).tolist(), "objects_start": objs0, "objects_end": objs1}


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, body, ctype="application/json", code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(_finite(body))
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                ov = json.loads(q["set"]) if q.get("set") else None
                if u.path in ("/", "/index.html"):
                    return self._send(PAGE.read_text(), "text/html; charset=utf-8")
                if u.path == "/api/index":
                    return self._send(engine.index())
                if u.path == "/api/frame":
                    return self._send(engine.run_frame(q["name"], ov))
                if u.path == "/api/all":
                    return self._send(engine.run_all(ov))
                if u.path == "/api/episode":
                    return self._send(engine.run_episode(q["name"]))
                return self._send({"error": "not found"}, code=404)
            except Exception as e:  # show config/key errors in the page instead of a dropped connection
                import traceback
                traceback.print_exc()
                return self._send({"error": f"{type(e).__name__}: {e}"}, code=400)
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="runs/perception_eval_raw", help="perception_eval --raw output (*.npz)")
    ap.add_argument("--episodes", default="runs/eval", help="evaluator output dir (searched for episode_*.npz)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    engine = Engine(a.frames, a.episodes)
    print(f"{len(engine.frames)} frames, {len(engine.episodes)} episodes")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine))
    print(f"open http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
