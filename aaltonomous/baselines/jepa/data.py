"""Episode files for the JEPA world model, and an importer for the LeRobot v3.0 demo drops.

One episode = one `.npz` (+ a `.json` sidecar) sampled every `macro_ticks` control ticks:

    head_rgb (N,H,W,3) uint8 [+ left_wrist_rgb / right_wrist_rgb]   images at the sample ticks
    arm_q, hand_q (N,14)          measured joints at those ticks
    arm_cmd, hand_cmd (N,14)      commanded (filtered) targets at those ticks; actions are their differences
    t (N,)                        seconds since episode start
    sidecar: instruction, task, mapping, success, n_correct, n_objects, source, ...

Writers: collect.py (sim / rendered mock rollouts of a teacher policy) and `import-lerobot` below (real demos).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
# Our camera key -> names tried in a LeRobot dataset (Aalto drop format first, then Unitree's xr_teleoperate names
# as in the zupermiez/g1_junction_drop_2 and g1_pick_cube Hugging Face datasets)
CAMERA_ALIASES = {"head_rgb": ("head_rgb", "cam_high"), "left_wrist_rgb": ("left_wrist_rgb", "cam_left_wrist"),
                  "right_wrist_rgb": ("right_wrist_rgb", "cam_right_wrist")}


def save_episode(path, arrays: dict, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))


def load_episode(path, images=True):
    path = Path(path)
    with np.load(path.with_suffix(".npz")) as d:
        arrays = {k: d[k] for k in d.files if images or k not in IMAGE_KEYS}
    sc = path.with_suffix(".json")
    meta = json.loads(sc.read_text()) if sc.exists() else {}
    return arrays, meta


def list_episodes(root):
    return sorted(p.with_suffix("") for p in Path(root).rglob("*.npz"))


# ----------------------------------------------------------------------------------- LeRobot v3.0 import
def _read_parquets(paths):
    import pyarrow.parquet as pq
    import pyarrow as pa
    return pa.concat_tables([pq.read_table(p) for p in paths]).to_pydict()


def _decode_frames(video, t_wanted, size=None):
    """Decode `video` and return the frame nearest to each timestamp in t_wanted (sorted, s). Optional resize."""
    import av
    out, i = [], 0
    with av.open(str(video)) as c:
        st = c.streams.video[0]
        last = None
        for fr in c.decode(st):
            ts = float(fr.pts * st.time_base)
            while i < len(t_wanted) and ts >= t_wanted[i] - 0.5 / 30.0:
                img = fr.to_ndarray(format="rgb24") if size is None else \
                    fr.reformat(width=size[1], height=size[0]).to_ndarray(format="rgb24")
                out.append(img)
                i += 1
            last = fr
            if i >= len(t_wanted):
                break
        while i < len(t_wanted) and last is not None:  # past the end: repeat the last frame
            out.append(out[-1] if out else last.to_ndarray(format="rgb24"))
            i += 1
    return np.stack(out)


def import_lerobot(root, out_dir, macro_ticks=8, cameras=("head_rgb",), size=(240, 320), only_success=False):
    """LeRobot v3.0 dataset dir (meta/info.json, meta/episodes/, data/, videos/) -> episode npz files.

    Written against the v3.0 layout of the Aalto drops (DATA_CARD.md); `observation.state` = arm_q ++ hand_q,
    `action` = arm target ++ hand target (joint_spec.yaml order). Camera names: CAMERA_ALIASES. Per-episode extras come
    from meta/aalto_episodes.jsonl when present (else success is unknown = None, kept in the goal bank)."""
    root, out_dir = Path(root), Path(out_dir)
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info.get("fps", 30))
    vkeys = {}
    for cam in cameras:
        names = [f"observation.images.{n}" for n in CAMERA_ALIASES.get(cam, (cam,))]
        found = [n for n in names if n in info["features"]]
        if not found:
            raise ValueError(f"{root}: no camera for {cam!r} (tried {names})")
        vkeys[cam] = found[0]
    if round(fps) != 30:
        print(f"[import] warning: {root} is {fps} fps, the interface runs at 30 Hz (macro steps will be off)")
    eps = _read_parquets(sorted((root / "meta/episodes").rglob("*.parquet")))
    extras = {}
    side = root / "meta/aalto_episodes.jsonl"
    if side.exists():
        for line in side.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                extras[int(r["episode_index"])] = r
    data_cache = {}
    n_written = 0
    for j, ep in enumerate(eps["episode_index"]):
        ep = int(ep)
        ex = extras.get(ep, {})
        if only_success and ex.get("success") is False:
            continue
        dkey = (int(eps["data/chunk_index"][j]), int(eps["data/file_index"][j]))
        if dkey not in data_cache:
            p = root / info["data_path"].format(chunk_index=dkey[0], file_index=dkey[1])
            data_cache = {dkey: _read_parquets([p])}
        d = data_cache[dkey]
        idx = np.asarray(d["episode_index"]) == ep
        state = np.asarray(d["observation.state"], np.float32)[idx]
        action = np.asarray(d["action"], np.float32)[idx]
        ts = np.asarray(d["timestamp"], float)[idx]
        sel = np.arange(0, len(ts), macro_ticks)
        arrays = {"arm_q": state[sel, :14], "hand_q": state[sel, 14:], "arm_cmd": action[sel, :14],
                  "hand_cmd": action[sel, 14:], "t": ts[sel] - ts[0]}
        for cam in cameras:
            vk = vkeys[cam]
            vc, vf = int(eps[f"videos/{vk}/chunk_index"][j]), int(eps[f"videos/{vk}/file_index"][j])
            t0 = float(eps[f"videos/{vk}/from_timestamp"][j])
            video = root / info["video_path"].format(video_key=vk, chunk_index=vc, file_index=vf)
            arrays[cam] = _decode_frames(video, t0 + ts[sel], size)
        tasks = eps.get("tasks", [[None]] * len(eps["episode_index"]))[j]
        meta = {"source": f"lerobot:{root.name}", "episode_index": ep, "fps": fps, "macro_ticks": macro_ticks,
                "instruction": ex.get("instruction") or (tasks[0] if tasks else None), "task": ex.get("task"),
                "mapping": ex.get("mapping"), "success": ex.get("success")}
        save_episode(out_dir / f"{root.name}_ep{ep:04d}", arrays, meta)
        n_written += 1
        print(f"[import] episode {ep}: {len(sel)} samples -> {out_dir}/{root.name}_ep{ep:04d}.npz")
    return n_written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import a LeRobot v3.0 Aalto demo drop into JEPA episode files.")
    ap.add_argument("root", help="LeRobot v3.0 dataset directory")
    ap.add_argument("--out", default="runs/jepa/episodes")
    ap.add_argument("--macro-ticks", type=int, default=8)
    ap.add_argument("--cameras", nargs="+", default=["head_rgb"])
    ap.add_argument("--only-success", action="store_true")
    a = ap.parse_args(argv)
    n = import_lerobot(a.root, a.out, a.macro_ticks, tuple(a.cameras), only_success=a.only_success)
    print(f"{n} episode(s) written to {a.out}")


if __name__ == "__main__":
    main()
