"""Head RGB-D -> coloured objects in the pelvis frame. Shape-agnostic: an object is a coloured blob, its grasp
target is the centre of its top face.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw), ...]

Per colour: HSV threshold (config.yaml perception.hsv) -> opening -> connected components. Per blob: erode, take
the pixels with valid depth, deproject them (calibration.yaml intrinsics + T_pelvis_optical), keep the top face
(points within `top_band` of a robust max height) and use the median of those as the top centre. A top face whose
convex hull is larger than `max_top_area` (in object size^2; one object is <= 1) is several touching same-colour
objects merged into one blob: its points are split with k-means (smallest k whose clusters all fit), one detection
per cluster (method "split"). The opposite case: a hand in front of an object cuts its blob in pieces. A piece whose
top face is smaller than `min_top_area` (size^2) with a same-colour blob within `merge_gap` is part of that object: the
pieces' pixels are pooled and the top face is taken from all of them (method "joined"), as long as the pooled top face
still fits one object (`max_top_area`). With too few valid
depth pixels (holes, or no depth stream) the blob centroid ray is intersected with the plane at the known object
height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong height are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "split" (one of several objects in one blob) | "joined" (pieces of one
                               # object cut apart by a hand) | "plane"
    yaw: float | None = None   # in-plane orientation of the top face (rad), info only
    uv: tuple = field(default=(0.0, 0.0))

    def __repr__(self):
        x, y, z = self.top
        return (f"{self.color:<6} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]")


class Perception:
    def __init__(self, cfg, scene, calibration):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # pelvis-frame xy correction added to every detection (head extrinsics error; tuned on the robot)
        self.xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        out = {}
        for color, ranges in self.cfg["hsv"].items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def detect(self, rgb, depth, debug=False):
        dets, rejected = [], []
        comps = self.components(rgb, depth)
        for color in self.cfg["hsv"]:
            for _, found in self.group(color, [b for b in comps if b["color"] == color]):
                for det in found:
                    why = self._reject(det)
                    if why:
                        rejected.append((det, why))
                    else:
                        dets.append(det)
        return (dets, rejected) if debug else dets

    def components(self, rgb, depth, keep_small=False):
        """Colour blobs -> list of dicts: color, blob (uint8 mask), core (eroded mask), area (px), cent (uv),
        u/v/d (core pixels with valid depth), small (below min_area_px; only returned with keep_small)."""
        import cv2
        c = self.cfg
        dlo, dhi = c["depth_range_mm"]
        keep = self._keep_mask(depth)
        out = []
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                small = area < c["min_area_px"]
                if small and not keep_small:
                    continue
                blob = (lab == i).astype(np.uint8)
                rec = {"color": color, "blob": blob, "area": area, "cent": cent[i], "bbox": stats[i, :4].tolist(),
                       "small": small, "core": None}
                if not small:
                    e = int(c["erode_px"])
                    core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                    if core.sum() < 0.25 * area:
                        core = blob
                    v, u = np.nonzero(core)
                    d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
                    ok = (d >= dlo) & (d <= dhi)
                    rec.update(core=core, u=u[ok], v=v[ok], d=d[ok])
                out.append(rec)
        return out

    def group(self, color, blobs):
        """Blobs of one colour (from components(), not small) -> [(blob indices, [Detection, ...]), ...]. Blobs with
        enough depth are grouped (pieces of one object cut by a hand are joined) and turned into detections from
        depth (split when too big); the rest fall back to the plane, one detection per blob."""
        c = self.cfg
        deep = [k for k, b in enumerate(blobs) if len(b["u"]) >= c["min_valid_depth_px"]]
        out = []
        for g in self._join(blobs, deep):
            u, v, d = (np.concatenate([blobs[k][x] for k in g]) for x in "uvd")
            found = self._from_depth(color, u, v, d, sum(blobs[k]["area"] for k in g))
            if found and len(g) > 1:
                for det in found:
                    if det.method == "depth":
                        det.method = "joined"
            if not found:  # top face too thin: plane from the largest piece
                k = max(g, key=lambda k: blobs[k]["area"])
                found = [self._from_plane(color, blobs[k]["cent"], blobs[k]["area"], len(blobs[k]["u"]))]
            out.append((g, found))
        for k, b in enumerate(blobs):
            if k not in deep:
                out.append(([k], [self._from_plane(color, b["cent"], b["area"], len(b["u"]))]))
        return out

    def _join(self, blobs, idx):
        """Union of blob pieces cut apart by an occluder (hand): a piece whose top face is below min_top_area * size^2
        is joined to the nearest same-colour group within merge_gap (3D, xy) if the pooled top face still fits one
        object (<= max_top_area * size^2). -> list of index lists."""
        from scipy.spatial import cKDTree
        c = self.cfg
        groups = [[k] for k in idx]
        lim_small = float(c.get("min_top_area") or 0) * self.size ** 2
        if lim_small <= 0 or len(groups) < 2:
            return groups
        lim_big = float(c.get("max_top_area") or 0) * self.size ** 2
        gap = float(c.get("merge_gap", 0.03))
        P = {k: self.cal.pixel_to_pelvis(self.cam, blobs[k]["u"], blobs[k]["v"], blobs[k]["d"]) for k in idx}
        memo = {}  # per group (sorted index tuple): xy points, xy bbox, top-face hull area, k-d tree

        def info(g):
            key = tuple(sorted(g))
            if key not in memo:
                Q = np.vstack([P[k] for k in g])
                memo[key] = {"xy": Q[:, :2], "lo": Q[:, :2].min(0), "hi": Q[:, :2].max(0),
                             "area": _hull_area(Q[self._top_sel(Q)][:, :2]), "tree": None}
            return memo[key]

        def dist(a, b):
            ia, ib = info(a), info(b)
            if np.any(ia["lo"] - ib["hi"] > gap) or np.any(ib["lo"] - ia["hi"] > gap):
                return np.inf  # bounding boxes already farther apart than the gap
            if ib["tree"] is None:
                ib["tree"] = cKDTree(ib["xy"])
            return float(ib["tree"].query(ia["xy"], k=1, distance_upper_bound=gap)[0].min())

        changed = True
        while changed:
            changed = False
            for gi in sorted(range(len(groups)), key=lambda i: info(groups[i])["area"]):
                g = groups[gi]
                if info(g)["area"] >= lim_small:
                    break  # sorted by area: every remaining group is a whole object
                best = None
                for gj, h in enumerate(groups):
                    if gj == gi:
                        continue
                    dd = dist(g, h)
                    if dd <= gap and (best is None or dd < best[0]):
                        best = (dd, gj)
                if best is None:
                    continue
                merged = g + groups[best[1]]
                if lim_big > 0 and info(merged)["area"] > lim_big:
                    continue
                groups = [x for k, x in enumerate(groups) if k not in (gi, best[1])] + [merged]
                changed = True
                break
        return groups

    def _keep_mask(self, depth):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        keep = self._bin_hulls_mask(depth.shape if depth is not None else (480, 640))
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        ok = (h >= 0.25) & (h <= c["height_range"][1])
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
        for ctr in sc.cfg["bins"]["centers"].values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _from_depth(self, color, u, v, d, area):
        """-> list of Detections: one, or several when the top face is too big for one object (merged blob)."""
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        sel = self._top_sel(P)
        top = P[sel]
        if len(top) < 5:
            return []
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[sel] / 1000.0) ** 2
        us, vs = u[sel], v[sel]
        parts = self._split(top[:, :2])
        method = "split" if len(parts) > 1 else "depth"
        out = []
        for idx in parts:
            if len(idx) < 5:
                continue
            t, wi = top[idx], w[idx]
            xy = (t[:, :2] * wi[:, None]).sum(0) / wi.sum() + self.xy_offset
            z = float(np.median(t[:, 2]))
            n_px = int(round(area * len(idx) / len(top)))
            out.append(Detection(color, np.array([xy[0], xy[1], z]), n_px, len(idx), method, _yaw(t[:, :2]),
                                 (float(np.median(us[idx])), float(np.median(vs[idx])))))
        return out

    def _top_sel(self, P):
        """Mask of the top-face points: within top_band of a robust (percentile) max height."""
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        return P[:, 2] >= z_top - self.cfg["top_band"]

    def _split(self, xy):
        """Top-face points (n, 2) -> list of index arrays, one per object. One object's top face has a convex hull
        of <= size^2 (cube; cylinder/triangle less); above `max_top_area` * size^2 the blob holds several touching
        same-colour objects: k-means with the smallest k (2..max_split) whose every cluster hull fits."""
        import cv2
        lim = float(self.cfg.get("max_top_area") or 0) * self.size ** 2
        everything = [np.arange(len(xy))]
        if lim <= 0 or _hull_area(xy) <= lim:
            return everything
        pts = np.ascontiguousarray(xy, np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
        best = everything
        for k in range(2, int(self.cfg.get("max_split", 4)) + 1):
            if len(pts) < 5 * k:
                break
            cv2.setRNGSeed(k)  # deterministic
            _, lab, _ = cv2.kmeans(pts, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
            best = [np.flatnonzero(lab.ravel() == j) for j in range(k)]
            if all(_hull_area(xy[idx]) <= lim for idx in best):
                break
        return best

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.cal._T(self.cam)
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det):
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        return None


def _hull_area(xy):
    """Area (m^2) of the convex hull of points (n, 2)."""
    import cv2
    if len(xy) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.ascontiguousarray(xy, np.float32))))


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
