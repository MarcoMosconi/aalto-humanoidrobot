"""Shape-aware detection on top of the classical Perception (classical code untouched).

The classical detector finds every object's top-face centre well but treats all objects as squares: its yaw is a
min-area-rect angle folded to 90 deg, wrong for triangles, and nothing downstream knows the shape. Triangles are the
worst-sorted shape (55 % vs ~80 %): the scene's grasp width (45 mm) is the triangle's ALTITUDE (flat face to apex),
so a hand closing across another direction meets up to 52 mm of edge or a corner.

ShapePerception keeps every classical detection and adds, from the same top-face 3D points:
  det.shape       "cube" | "cylinder" | "triangle" (convex-hull fill of the min-area rectangle / enclosing triangle)
  det.shape_score the per-shape scores used for the decision
  det.tri_yaw     triangle only: direction (rad, pelvis frame) from the triangle's centre to an apex, modulo 120 deg
                  (minEnclosingTriangle on the top-face hull); the altitude axis to close along

    python -m aaltonomous.baselines.jepa.shapes runs/perc_raw/a runs/perc_raw/b    # offline check vs sim ground truth
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from aaltonomous.baselines.classical import perception as _cp
from aaltonomous.baselines.classical.perception import Detection, Perception

TRI_PERIOD = 2 * np.pi / 3
TRI_MAX_DEV = 15.0          # deg: max corner deviation from 60 for a triangle


def shape_features(xy):
    """Top-face points (N,2) metres -> dict of fill ratios (hull area / enclosing shape area)."""
    import cv2
    p = np.ascontiguousarray((np.asarray(xy, float) - np.mean(xy, 0)) * 1000.0, np.float32)   # mm
    hull = cv2.convexHull(p)
    a = float(cv2.contourArea(hull))
    if a <= 1.0:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    (_, _), r = cv2.minEnclosingCircle(hull)
    ta, tri = cv2.minEnclosingTriangle(hull)
    t = None if tri is None else tri.reshape(3, 2).astype(float)
    dev = 90.0
    if t is not None:                       # equilateral? (a square cut diagonally by occlusion gives a 90 deg corner)
        ang = []
        for i in range(3):
            e1, e2 = t[(i + 1) % 3] - t[i], t[(i + 2) % 3] - t[i]
            ang.append(np.degrees(np.arccos(np.clip(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9), -1, 1))))
        dev = float(np.max(np.abs(np.asarray(ang) - 60.0)))
    return {"rect": a / max(w * h, 1e-6), "circle": a / max(np.pi * r * r, 1e-6), "tri": a / max(float(ta), 1e-6),
            "tri_dev": dev, "area_mm2": a, "tri_pts": None if t is None else t / 1000.0 + np.mean(xy, 0)}


def classify(f):
    """Nearest prototype on (rect fill, circle fill, triangle fill). Ideal shapes: square (1, .64, .5..),
    circle (.785, 1, .6), triangle (.5, .41..., 1); eroded / sampled masks pull everything a bit down."""
    if f is None:
        return None, {}
    s = {"cube": f["rect"] - max(f["circle"], f["tri"]) * 0.5,
         "cylinder": f["circle"] - max(f["rect"], f["tri"]) * 0.5,
         "triangle": f["tri"] - max(f["rect"], f["circle"]) * 0.5}
    if f["tri_dev"] > TRI_MAX_DEV:          # not equilateral: never a triangle
        s["triangle"] = -1.0
    return max(s, key=s.get), {k: round(float(v), 3) for k, v in s.items()}


def tri_yaw(f):
    """Centre -> apex direction modulo 120 deg (circular mean of 3 * vertex angles)."""
    if f is None or f["tri_pts"] is None:
        return None
    v = f["tri_pts"]
    c = v.mean(0)
    ang = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])
    m = np.angle(np.exp(3j * ang).mean()) / 3.0
    return float(m)


class ShapePerception(Perception):
    """Classical Perception + det.shape / det.tri_yaw (same detections, same order)."""

    def __init__(self, *a, min_points=25, **kw):
        super().__init__(*a, **kw)
        self.min_points = min_points

    def _from_depth(self, color, u, v, d, area):
        seen = []
        orig = _cp._yaw

        def grab(xy):                       # the classical code calls _yaw(top-face xy) once per detection
            seen.append(np.array(xy))
            return orig(xy)
        _cp._yaw = grab
        try:
            dets = super()._from_depth(color, u, v, d, area)
        finally:
            _cp._yaw = orig
        for det, xy in zip(dets, seen):
            annotate(det, xy, self.min_points)
        return dets

    def detect(self, rgb, depth, debug=False):
        out = super().detect(rgb, depth, debug)
        for det in (out[0] if debug else out):
            if not hasattr(det, "shape"):   # plane fallback: no 3D top face
                det.shape, det.shape_score, det.tri_yaw = None, {}, None
        return out


def annotate(det: Detection, xy, min_points=25):
    if len(xy) < min_points:
        det.shape, det.shape_score, det.tri_yaw = None, {}, None
        return det
    f = shape_features(xy)
    det.shape, det.shape_score = classify(f)
    det.tri_yaw = tri_yaw(f) if det.shape == "triangle" else None
    det.shape_feat = None if f is None else {k: round(float(f[k]), 3) for k in ("rect", "circle", "tri", "tri_dev", "area_mm2")}
    return det


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


def offline_check(dirs, offset=None):
    """Re-detect saved perception_eval --raw frames; shape confusion + triangle yaw error vs ground truth."""
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception_eval import match
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    per = ShapePerception(load_config(None), Scene.load(), Calibration.load())
    conf, rows = {}, []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            gt = json.loads(str(z["gt"]))
            pairs, missed, fps = match(gt, per.detect(z["rgb"], z["depth"]))
            for g, det in pairs:
                conf.setdefault(g["shape"], {}).setdefault(str(det.shape), 0)
                conf[g["shape"]][str(det.shape)] += 1
                rows.append({"gt": g["shape"], "pred": det.shape, "gt_yaw": g["yaw"], "tilt": g["tilt_deg"],
                             "tri_yaw": det.tri_yaw, "feat": getattr(det, "shape_feat", None), "frame": f.stem})
    print("shape confusion (rows = truth):")
    for k, v in sorted(conf.items()):
        n = sum(v.values())
        print(f"  {k:9s} n={n:3d}  " + "  ".join(f"{p}={c}" for p, c in sorted(v.items())) +
              f"   correct {v.get(k, 0) / n:.2f}")
    tr = [r for r in rows if r["gt"] == "triangle" and r["tri_yaw"] is not None and r["tilt"] < 10]
    if tr:
        diff = np.array([r["tri_yaw"] - r["gt_yaw"] for r in tr])
        off = float(np.angle(np.exp(3j * diff).mean()) / 3.0) if offset is None else offset
        err = np.degrees(np.abs(_wrap(diff - off, TRI_PERIOD)))
        print(f"triangle yaw: n={len(tr)}  convention offset {np.degrees(off):+.1f} deg  |err| median "
              f"{np.median(err):.1f}  p90 {np.percentile(err, 90):.1f}  max {err.max():.1f} deg")
    return rows


def closing_yaw(kin, side, arm_q, hand_q):
    """In-plane direction (rad, pelvis) from the thumb tip to the index/middle tips: the axis the hand closes along."""
    pts = kin.hand_points(side, arm_q, hand_q)          # palm, grasp, 3 finger links, tips (index, middle, thumb)
    d = pts[5:7].mean(0) - pts[7]
    return float(np.arctan2(d[1], d[0]))


def grasp_rows(picks_dirs):
    """Triangle picks from collect_grasps.py --triangles -> rows with the hand's closing axis vs the triangle."""
    from aaltonomous.baselines.classical.kinematics import ArmKinematics
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.jepa.data import load_episode
    kin = ArmKinematics(load_config(None))
    rows = []
    for d in picks_dirs:
        for pj in sorted(Path(d).glob("*.json")):
            arr, meta = load_episode(pj.with_suffix(""))
            for p in meta["picks"]:
                sh = p.get("shape") or {}
                if sh.get("shape") != "triangle" or sh.get("tri_yaw") is None:
                    continue
                arm, hand = arr[f"arm_{p['i']}"], arr[f"hand_{p['i']}"]
                z = [kin.grasp_pose(p["side"], q)[0][2] for q in arm]
                j = int(np.argmin(z))
                cy = closing_yaw(kin, p["side"], arm[j], hand[j])
                rows.append({"held": bool(p["held"]), "delivered": p.get("delivered"), "variant": p["variant"]["name"],
                             "side": p["side"], "close_yaw": cy, "tri_yaw": sh["tri_yaw"],
                             "rel60": float(np.degrees(_wrap(cy - sh["tri_yaw"], np.pi / 3))),
                             "rel120": float(np.degrees(_wrap(cy - sh["tri_yaw"], TRI_PERIOD))), "ep": pj.stem})
    return rows


def grasp_report(picks_dirs):
    rows = grasp_rows(picks_dirs)
    print(f"triangle picks: {len(rows)}, held {sum(r['held'] for r in rows)}")
    if not rows:
        return rows
    r60 = np.abs([r["rel60"] for r in rows])
    held = np.array([r["held"] for r in rows])
    dl = np.array([bool(r["delivered"]) for r in rows])
    print("|closing axis - nearest face-to-apex axis| (deg):  n  held  delivered")
    for lo, hi in ((0, 7.5), (7.5, 15), (15, 22.5), (22.5, 30.01)):
        m = (r60 >= lo) & (r60 < hi)
        if m.any():
            print(f"  {lo:4.1f}-{hi:4.1f}  {m.sum():3d}  {held[m].mean():.2f}  {dl[m].mean():.2f}")
    r120 = np.array([r["rel120"] for r in rows])
    m = r60 < 15
    if m.any():
        apex = m & (np.abs(r120) < 30)               # thumb on the flat face, fingers on the apex
        face = m & (np.abs(r120) >= 30)              # fingers on the flat face, thumb on the apex
        for nm, mm in (("fingers->apex", apex), ("fingers->face", face)):
            if mm.any():
                print(f"  aligned, {nm:14s} n={mm.sum():3d} held {held[mm].mean():.2f} delivered {dl[mm].mean():.2f}")
    return rows


def bench_report(picks_dirs):
    """collect_grasps.py --bench: held / delivered per (finger-direction mode, grasp height) on triangles."""
    import json
    cell = {}
    for d in picks_dirs:
        for pj in sorted(Path(d).glob("*.json")):
            for p in json.load(open(pj))["picks"]:
                n = p["variant"]["name"]
                key = tuple(n.split("_")[1:3]) if n.startswith("tri_") else ("stock", "")
                cell.setdefault(key, []).append((bool(p["held"]), bool(p.get("delivered"))))
    print(f"{'mode':8s} {'height':6s}   n  held  delivered")
    for k in sorted(cell):
        v = np.array(cell[k])
        se = np.sqrt(v[:, 0].mean() * (1 - v[:, 0].mean()) / len(v))
        print(f"{k[0]:8s} {k[1]:6s} {len(v):3d}  {v[:, 0].mean():.2f}  {v[:, 1].mean():.2f}   (held +-{se:.2f})")
    for i, nm in ((0, "mode"), (1, "height")):
        agg = {}
        for k, v in cell.items():
            if k[0] != "stock":
                agg.setdefault(k[i], []).extend(v)
        print(f"by {nm}: " + ", ".join(f"{a}={np.mean([x[0] for x in v]):.2f} (n={len(v)})" for a, v in sorted(agg.items())))
    return cell


if __name__ == "__main__":
    if sys.argv[1:2] == ["bench"]:
        bench_report(sys.argv[2:] or ["runs/jepa/grasps_bench/picks"])
    elif sys.argv[1:2] == ["grasps"]:
        grasp_report(sys.argv[2:] or ["runs/jepa/grasps_tri/picks"])
    else:
        offline_check(sys.argv[1:] or ["runs/perc_raw/a", "runs/perc_raw/b"])
