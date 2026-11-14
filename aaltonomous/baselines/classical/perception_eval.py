"""Measure the classical Perception against sim ground truth (no arm motion needed).

    # sim running (bash aaltonomous/cloud/aalto.sh sim)
    bash aaltonomous/cloud/aalto.sh py -m aaltonomous.baselines.classical.perception_eval \
        [--scenarios random crowded near_bins wide] [--n 10] [--out runs/perception_eval]

Per frame: place a layout through the scene service (:55580), wait until every object is at rest, grab a fresh head
RGB-D frame, run Perception.detect(rgb, depth, debug=True) and match its detections to the true top-face centres
(state() pose + h/2, exactly what sim/oracle.py PrivilegedPerception uses) per colour, Hungarian on xy distance.

Scenarios (layouts in mat coords, same format as tasks.sample_layout):
  random     the scene service's own random layout (scene.yaml knobs) = what an episode sees
  crowded    4-5 objects in a tight cluster, 0-8 mm gaps, same-colour pairs touching
  near_bins  3-5 objects in the row next to the bins (2-20 mm from the bin wall)
  wide       3-5 objects spread over the whole mat incl. the corners under the home-pose hands

Writes <out>/summary.json, <out>/frames.jsonl and one overlay JPEG per frame (<out>/img/<scenario>_<i>.jpg):
mask contours (colour), detections (filled dot + label), rejected blobs (x + reason), ground truth (white ring),
match lines (green < 1 cm, orange < 2 cm, red otherwise).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MATCH_RADIUS = 0.04           # m: a detection farther than this from every same-colour GT object is a false positive
DRAW_BGR = {"red": (40, 40, 230), "yellow": (0, 220, 240), "blue": (230, 120, 20)}


# ---------------------------------------------------------------------- layouts
def _layout_crowded(rng, sc, colors, shapes):
    n = int(rng.integers(4, 6))
    cx, cy = rng.uniform(0.30, 0.60), rng.uniform(0.13, 0.19)
    cols = list(rng.permutation(colors))
    cols += [cols[int(rng.integers(3))]] * (n - 3)            # duplicate colours -> same-colour neighbours
    out, size = [], sc.size
    for c in cols:
        for _ in range(2000):
            if not out:
                x, y = cx, cy
            else:
                a = out[int(rng.integers(len(out)))]           # attach to a random placed object
                th = rng.uniform(0, 2 * np.pi)
                r = size * np.sqrt(2) / 2 + size / 2 + rng.uniform(0.0, 0.008) if rng.random() < 0.3 else size + rng.uniform(0.0, 0.008)
                x, y = a["x"] + r * np.cos(th), a["y"] + r * np.sin(th)
            if not (0.20 <= x <= 0.70 and 0.10 <= y <= 0.225):
                continue
            if all(np.hypot(x - o["x"], y - o["y"]) >= size * 1.12 for o in out):
                out.append({"color": str(c), "x": round(float(x), 3), "y": round(float(y), 3)})
                break
    return _finish(out, rng, shapes)


def _layout_near_bins(rng, sc, colors, shapes):
    n = int(rng.integers(3, 6))
    bin_edge = min(c[1] for c in sc.cfg["bins"]["centers"].values()) - sc.cfg["bins"]["outer_size"][1] / 2
    out = []
    cols = list(rng.permutation(colors)) + list(rng.choice(colors, n - 3))
    for c in cols:
        for _ in range(2000):
            x = rng.uniform(0.15, 0.75)
            y = bin_edge - sc.size / np.sqrt(2) - rng.uniform(0.002, 0.020)   # corner 2-20 mm from the wall
            if all(abs(x - o["x"]) >= 0.07 for o in out):
                out.append({"color": str(c), "x": round(float(x), 3), "y": round(float(y), 3)})
                break
    return _finish(out, rng, shapes)


def _layout_wide(rng, sc, colors, shapes):
    n = int(rng.integers(3, 6))
    cols = list(rng.permutation(colors)) + list(rng.choice(colors, n - 3))
    out = []
    for c in cols:
        for _ in range(2000):
            x, y = rng.uniform(0.08, 0.82), rng.uniform(0.04, 0.22)
            if all(np.hypot(x - o["x"], y - o["y"]) >= 0.08 for o in out):
                out.append({"color": str(c), "x": round(float(x), 3), "y": round(float(y), 3)})
                break
    return _finish(out, rng, shapes)


def _finish(objs, rng, shapes):
    for o in objs:
        o["shape"] = str(shapes[int(rng.integers(len(shapes)))])
        o["yaw"] = round(float(rng.uniform(-np.pi / 4, np.pi / 4)), 3)
    return objs


LAYOUTS = {"crowded": _layout_crowded, "near_bins": _layout_near_bins, "wide": _layout_wide}


# ---------------------------------------------------------------------- ground truth
def _quat_yaw_tilt(q):
    x, y, z, w = q
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    up_z = 1 - 2 * (x * x + y * y)                            # body z axis . world z
    return yaw, float(np.degrees(np.arccos(np.clip(up_z, -1, 1))))


def ground_truth(sc, state):
    gt = []
    for o in state["objects"]:
        if not o["active"]:
            continue
        p = np.asarray(o["pos"], float)
        m = sc.pelvis_to_mat(p)
        yaw, tilt = _quat_yaw_tilt(o.get("quat_xyzw") or [0, 0, 0, 1])
        where = "bin" if sc.bin_of(m) is not None else ("counter" if sc.on_counter(m) else "off")
        gt.append({"name": o["name"], "color": o["color"], "shape": o["shape"], "where": where,
                   "top": (p + [0, 0, sc.object_height(o["shape"]) / 2]).tolist(), "yaw": yaw, "tilt_deg": tilt})
    return gt


def _yaw_err_deg(det_yaw, gt_yaw, period):
    d = (det_yaw - gt_yaw + period / 2) % period - period / 2
    return float(abs(np.degrees(d)))


def match(gt, dets):
    """Per colour Hungarian on xy distance; pairs farther than MATCH_RADIUS are unmatched."""
    from scipy.optimize import linear_sum_assignment
    pairs, used_d = [], set()
    targets = [g for g in gt if g["where"] == "counter"]
    for color in {g["color"] for g in targets} | {d.color for d in dets}:
        G = [i for i, g in enumerate(targets) if g["color"] == color]
        D = [j for j, d in enumerate(dets) if d.color == color]
        if not G or not D:
            continue
        C = np.array([[np.linalg.norm(np.asarray(targets[i]["top"][:2]) - dets[j].top[:2]) for j in D] for i in G])
        for r, c in zip(*linear_sum_assignment(C)):
            if C[r, c] <= MATCH_RADIUS:
                pairs.append((targets[G[r]], dets[D[c]]))
                used_d.add(D[c])
    matched_g = {id(g) for g, _ in pairs}
    missed = [g for g in targets if id(g) not in matched_g]
    fps = [d for j, d in enumerate(dets) if j not in used_d]
    return pairs, missed, fps


# ---------------------------------------------------------------------- drawing
def overlay(rgb, per, cal, cam, pairs, missed, fps, rejected, title):
    import cv2
    img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    for color, m in per.masks(rgb).items():
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, DRAW_BGR.get(color, (255, 255, 255)), 1)

    def uv(p):
        u, v, z = cal.project(cam, np.asarray(p, float))
        return (int(round(u)), int(round(v))) if z > 0 and np.isfinite(u) else None

    for g, d in pairs:
        a, b = uv(g["top"]), uv(d.top)
        e = np.linalg.norm(np.asarray(g["top"][:2]) - d.top[:2])
        col = (0, 200, 0) if e < 0.01 else (0, 140, 255) if e < 0.02 else (0, 0, 255)
        if a and b:
            cv2.line(img, a, b, col, 1)
        if a:
            cv2.circle(img, a, 6, (255, 255, 255), 1)
        if b:
            cv2.circle(img, b, 3, col, -1)
            cv2.putText(img, f"{e * 1000:.0f}mm", (b[0] + 5, b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for g in missed:
        a = uv(g["top"])
        if a:
            cv2.circle(img, a, 7, (0, 0, 255), 2)
            cv2.putText(img, "MISS", (a[0] + 6, a[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for d in fps:
        b = uv(d.top) if np.all(np.isfinite(d.top)) else None
        if b:
            cv2.drawMarker(img, b, (255, 0, 255), cv2.MARKER_SQUARE, 10, 2)
            cv2.putText(img, "FP", (b[0] + 6, b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for d, why in rejected:
        b = (int(d.uv[0]), int(d.uv[1]))
        cv2.drawMarker(img, b, (160, 160, 160), cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.putText(img, why[:14], (b[0] + 4, b[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------------- main loop
def settle(client, timeout=6.0, v_max=0.005):
    t0 = time.time()
    time.sleep(0.8)
    while time.time() - t0 < timeout:
        st = client.state()
        if all(o["speed"] < v_max for o in st["objects"] if o["active"]):
            return st, True
        time.sleep(0.2)
    return client.state(), False


def fresh_frame(rgbd, after_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = rgbd.get()
        if f is not None and f[3] > after_id + 1:
            return f
        time.sleep(0.03)
    raise TimeoutError("no fresh head RGB-D frame")


def summarize(rows):
    def stats(a):
        a = np.asarray(a, float)
        if not len(a):
            return None
        return {"n": int(len(a)), "mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
                "p90": round(float(np.percentile(a, 90)), 2), "max": round(float(a.max()), 2)}
    n_gt = sum(r["n_gt"] for r in rows)
    n_hit = sum(r["n_matched"] for r in rows)
    xy = [m["err_xy_mm"] for r in rows for m in r["matches"]]
    z = [m["err_z_mm"] for r in rows for m in r["matches"]]
    out = {
        "frames": len(rows), "objects": n_gt, "recall": round(n_hit / max(n_gt, 1), 3),
        "frames_all_found": round(float(np.mean([r["n_matched"] == r["n_gt"] for r in rows])), 3),
        "false_positives": sum(len(r["false_pos"]) for r in rows),
        "frames_clean": round(float(np.mean([r["n_matched"] == r["n_gt"] and not r["false_pos"] for r in rows])), 3),
        "err_xy_mm": stats(xy), "err_z_mm": stats(z),
        "err_x_mm_mean_signed": round(float(np.mean([m["dx_mm"] for r in rows for m in r["matches"]])), 2) if xy else None,
        "err_y_mm_mean_signed": round(float(np.mean([m["dy_mm"] for r in rows for m in r["matches"]])), 2) if xy else None,
        "method": {k: sum(m["method"] == k for r in rows for m in r["matches"]) for k in ("depth", "plane")},
        "yaw_err_deg_cube": stats([m["yaw_err_deg"] for r in rows for m in r["matches"]
                                   if m["shape"] == "cube" and m["yaw_err_deg"] is not None]),
        "missed_why": {},
        "rejected": {},
        "missed_by_shape": {}, "objects_by_shape": {},
    }
    for r in rows:
        for why in r["rejected"]:
            k = why.split(":")[0].split(" ")[0]
            out["rejected"][k] = out["rejected"].get(k, 0) + 1
        for g in r["gt"]:
            if g["where"] == "counter":
                out["objects_by_shape"][g["shape"]] = out["objects_by_shape"].get(g["shape"], 0) + 1
        for g in r["missed"]:
            out["missed_why"][g.get("why", "?")] = out["missed_why"].get(g.get("why", "?"), 0) + 1
            out["missed_by_shape"][g["shape"]] = out["missed_by_shape"].get(g["shape"], 0) + 1
    vis = n_gt - out["missed_why"].get("out_of_view", 0)
    out["recall_in_view"] = round(n_hit / max(vis, 1), 3)
    if xy:   # what is left after removing the constant offset (an extrinsics bias, fixable with xy_offset)
        dxy = np.array([[m["dx_mm"], m["dy_mm"]] for r in rows for m in r["matches"]])
        out["err_xy_mm_bias_removed"] = stats(np.linalg.norm(dxy - dxy.mean(0), axis=1))
    out["xy_by_shape"] = {s: stats([m["err_xy_mm"] for r in rows for m in r["matches"] if m["shape"] == s])
                          for s in out["objects_by_shape"]}
    return out


def classify_miss(g, rgb, depth, per, pairs, scene):
    """Why was this on-counter object missed?  out_of_view | occluded (something nearer the camera, e.g. a hand,
    covers its top centre) | merged (its pixel lies in the same colour blob as a matched same-colour neighbour) |
    detector (visible, own blob, still not reported)."""
    import cv2
    cal, cam = per.cal, per.cam
    H, W = depth.shape
    u, v, z = cal.project(cam, np.asarray(g["top"], float))
    if not (z > 0 and 2 <= u < W - 2 and 2 <= v < H - 2):
        return "out_of_view"
    u, v = int(round(u)), int(round(v))
    d = depth[v - 2:v + 3, u - 2:u + 3].astype(float)
    d = d[(d > 0)]
    if len(d) and np.median(d) / 1000.0 < z - 0.02:
        return "occluded"
    m = per.masks(rgb)[g["color"]]
    _, lab = cv2.connectedComponents(m, connectivity=8)
    if lab[v, u]:
        for g2, _ in pairs:
            if g2["color"] != g["color"]:
                continue
            u2, v2, _ = cal.project(cam, np.asarray(g2["top"], float))
            if 0 <= u2 < W and 0 <= v2 < H and lab[int(round(v2)), int(round(u2))] == lab[v, u]:
                return "merged"
    return "detector"


def evaluate_frame(per, scene, rgb, depth, gt, scen, i, lay, still, out):
    import cv2
    t = time.perf_counter()
    dets, rejected = per.detect(rgb, depth, debug=True)
    dt_ms = (time.perf_counter() - t) * 1000
    pairs, missed, fps = match(gt, dets)
    ms = []
    for g, d in pairs:
        e = d.top - np.asarray(g["top"])
        period = {"cube": np.pi / 2}.get(g["shape"])
        ye = _yaw_err_deg(d.yaw, g["yaw"], period) if (period and d.yaw is not None and g["tilt_deg"] < 10) else None
        ms.append({"name": g["name"], "shape": g["shape"], "method": d.method,
                   "err_xy_mm": round(float(np.linalg.norm(e[:2]) * 1000), 2),
                   "dx_mm": round(float(e[0] * 1000), 2), "dy_mm": round(float(e[1] * 1000), 2),
                   "err_z_mm": round(float(e[2] * 1000), 2), "yaw_err_deg": None if ye is None else round(ye, 1),
                   "n_px": d.n_px, "det_yaw_deg": None if d.yaw is None else round(float(np.degrees(d.yaw)), 1),
                   "gt_yaw_deg": round(float(np.degrees(g["yaw"])), 1), "tilt_deg": round(g["tilt_deg"], 1)})
    n_gt = sum(g["where"] == "counter" for g in gt)
    row = {"scenario": scen, "i": i, "settled": still, "layout": lay, "gt": gt, "n_gt": n_gt,
           "n_matched": len(pairs), "matches": ms, "detect_ms": round(dt_ms, 1),
           "missed": [{"name": g["name"], "shape": g["shape"], "top": g["top"],
                       "why": classify_miss(g, rgb, depth, per, pairs, scene)} for g in missed],
           "false_pos": [repr(d) for d in fps], "rejected": [why for _, why in rejected]}
    why = ",".join(m["why"] for m in row["missed"])
    title = f"{scen} #{i}: {len(pairs)}/{n_gt} found, {len(fps)} FP" + (f"  miss: {why}" if why else "")
    cv2.imwrite(str(out / "img" / f"{scen}_{i:02d}.jpg"),
                overlay(rgb, per, per.cal, per.cam, pairs, missed, fps, rejected, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
    errs = " ".join(f"{m['err_xy_mm']:.0f}" for m in ms)
    print(f"{scen:9s} #{i:02d} {len(pairs)}/{n_gt} fp={len(fps)} rej={len(rejected)} xy_mm=[{errs}]"
          f"{' miss=' + why if why else ''}{'' if still else ' (not settled)'}", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["random", "crowded", "near_bins", "wide"])
    ap.add_argument("--n", type=int, default=10, help="frames per scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", default=None, help='config overrides as json, e.g. \'{"perception.xy_offset": [0, -0.005]}\'')
    ap.add_argument("--out", default="runs/perception_eval")
    ap.add_argument("--raw", default=None, help="also save rgb/depth/gt per frame as .npz here (offline re-runs)")
    ap.add_argument("--offline", default=None, help="no sim: re-run detection on the .npz frames of an earlier --raw")
    a = ap.parse_args(argv)

    import os
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    from .config import load_config
    from .perception import Perception

    cfg = load_config(a.config, json.loads(a.set) if a.set else None, a.profile)
    scene, cal = Scene.load(), Calibration.load()
    per = Perception(cfg, scene, cal)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = []

    if a.offline:
        for f in sorted(Path(a.offline).glob("*.npz")):
            z = np.load(f)
            scen, i = f.stem.rsplit("_", 1)
            if scen not in a.scenarios:
                continue
            rows.append(evaluate_frame(per, scene, z["rgb"], z["depth"], json.loads(str(z["gt"])), scen, int(i),
                                       json.loads(str(z["layout"])), True, out))
    else:
        from aaltonomous.interface.head_rgbd import HeadRGBDClient
        from aaltonomous.sim.scene_client import SceneClient
        client = SceneClient(a.host)
        rgbd = HeadRGBDClient(a.host, int(os.environ.get("AALTO_RGBD_PORT", "55560")))
        if a.raw:
            Path(a.raw).mkdir(parents=True, exist_ok=True)
        colors, shapes = list(scene.colors), list(scene.cfg["objects"].get("shapes", ["cube"]))
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        while rgbd.get() is None and time.time() - t0 < 10:
            time.sleep(0.1)
        for scen in a.scenarios:
            for i in range(a.n):
                if scen == "random":
                    lay = client.reset_random(seed=a.seed * 1000 + i)["layout"]
                else:
                    lay = LAYOUTS[scen](rng, scene, colors, shapes)
                    client.reset(lay)
                last = rgbd.get()[3]
                _, still = settle(client)
                rgb, depth, _, _ = fresh_frame(rgbd, last)
                gt = ground_truth(scene, client.state())
                if a.raw:
                    np.savez_compressed(Path(a.raw) / f"{scen}_{i:02d}.npz", rgb=rgb, depth=depth,
                                        gt=json.dumps(gt), layout=json.dumps(lay))
                rows.append(evaluate_frame(per, scene, rgb, depth, gt, scen, i, lay, still, out))
        rgbd.close()

    with open(out / "frames.jsonl", "w") as fj:
        for r in rows:
            fj.write(json.dumps(r) + "\n")
    scens = [s for s in a.scenarios if any(r["scenario"] == s for r in rows)]
    summary = {"config": a.config or "config.yaml", "profile": a.profile, "overrides": a.set, "seed": a.seed,
               "offline": a.offline, "overall": summarize(rows),
               "by_scenario": {s: summarize([r for r in rows if r["scenario"] == s]) for s in scens}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
