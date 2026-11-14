# Classical perception vs. sim ground truth (2026-11-14)

Question: is perception or the grasp the weak point of the classical rung-1 baseline?
Short answer: **for episode-like layouts, perception is not the bottleneck.** It finds every object, with about 6 mm
of xy error, and most of that error is a constant offset that can be corrected. The remaining failures are layout
problems that a plain HSV-blob detector cannot solve (touching same-colour objects, objects behind the hands). They
are not threshold problems.

**Update (same day): merged blobs are now split.** A blob whose top face is bigger than one object
(`perception.max_top_area`) is split with k-means. The crowded scenario goes from 0.82 to 0.98 recall, and no object
is missed as `merged` any more (see finding 2).

**Update 2: pieces cut off by a hand are joined.** A blob piece whose top face is smaller than one object
(`perception.min_top_area`) with a same-colour blob within `merge_gap` is pooled with it into one detection. This
removes all 9 false positives (see finding 3). The tables below are for the current code. The numbers from before both
changes are kept in the "before" rows.

## How

`python -m aaltonomous.baselines.classical.perception_eval` (see its docstring) runs the steps below, with no arm motion:

1. Place a layout through the scene service and wait until all objects are at rest.
2. Grab a fresh head RGB-D frame and run `Perception.detect(rgb, depth)` with the stock `config.yaml` (sim profile).
3. Match the detections per colour (Hungarian, 4 cm gate) to the true top-face centres, taken as state() pose + h/2,
   the same source `sim/oracle.py` uses.

The robot stays in the sim home pose (`layout.ARM_HOME`). In this pose the hands cover the left and right edges of
the mat.

| scenario | what | frames | objects |
|---|---|---|---|
| `random` | the scene service's own `reset_random` (scene.yaml knobs), i.e. what an episode sees | 12 | 55 |
| `crowded` | 4-5 objects in one cluster, 0-8 mm gaps, same-colour neighbours | 12 | 55 |
| `near_bins` | a row 2-20 mm from the bin walls, x over the whole mat | 12 | 52 |
| `wide` | spread over the whole mat incl. the corners | 12 | 49 |

Each missed object gets one of these causes:
- `out_of_view`: its top centre projects outside the image.
- `occluded`: depth at that pixel is more than 2 cm nearer than the object, i.e. a hand is in front.
- `merged`: it lies in the same colour blob as a matched same-colour neighbour.
- `detector`: anything else.

## Results

| scenario | recall | recall (in view) | frames all found | false pos. | xy err median / p90 / max (mm) | xy err, bias removed, median / p90 | miss causes |
|---|---|---|---|---|---|---|---|
| random | **1.00** | 1.00 | 12/12 | 0 | 5.6 / 11.6 / 16.7 | 1.8 / 8.7 | none |
| crowded | **0.98** | 0.98 | 11/12 | 0 | 5.9 / 10.4 / 19.7 | 1.0 / 6.9 | occluded 1 |
| near_bins | 0.83 | 0.90 | 3/12 | 0 | 6.9 / 15.5 / 28.7 | 3.5 / 10.3 | occluded 5, out_of_view 4 |
| wide | 0.63 | 0.94 | 1/12 | 0 | 6.4 / 19.8 / 23.1 | 4.4 / 17.6 | out_of_view 16, occluded 2 |
| **all** | 0.87 | 0.96 | 27/48 | **0** | 6.1 / 14.0 / 28.7 | 1.8 / 10.7 | out_of_view 20, occluded 8, **merged 0, detector 0** |
| *crowded, before both* | *0.82* | *0.82* | *4/12* | *2* | *6.5 / 23.8 / 36.0* | *2.0 / 20.4* | *merged 9, occluded 1* |
| *all, before both* | *0.83* | *0.91* | *20/48* | *9* | *6.3 / 17.2 / 36.0* | *2.0 / 16.7* | *out_of_view 20, merged 9, occluded 8* |

Other numbers:
- Every detection came from depth; the plane fallback never triggered. Of the 183 matched detections, 17 come from a
  split blob (method `split`, all in `crowded`) and 9 from joined pieces (method `joined`, the 9 frames that had a
  false positive before). No single object was split or joined by mistake. Frames with no false positive and every
  object found: 27/48 (was 16/48 before both changes).
- detect() takes 50 ms median per frame on the CPU, 65 ms max. The first frame of a process takes ~250 ms once
  (scipy import for the join).
- z error is +5.9 mm median, and very consistent (p90 6.5 mm).
- Cube yaw (from the `minAreaRect` the code already computes, compared mod 90°) has a median error of 0.2°. It is
  wrong (20-42°) only when the blob is merged or clipped, and those are the same frames that have large xy errors.

The full per-frame data is in `summary.json` and `frames.jsonl`. `img/` holds overlays:
- coloured outlines: HSV masks
- white ring: ground truth
- dot: detection (green < 1 cm, orange < 2 cm, red worse)
- magenta square: false positive
- grey ×: rejected blob

## Findings

1. **A constant offset makes up most of the error.** Detections average +3.6 mm in x and +4.4 mm in y (pelvis
   frame), and z is +5.9 mm. Removing the mean offset brings the median xy error down from 6.1 mm to 1.8 mm.
   - The y error depends on position: +8.3 mm at pelvis y in [-0.2, -0.1) and +1.1 mm at y in [+0.1, +0.2]. That
     pattern fits a small yaw or roll error in the head camera extrinsics (calibration.yaml vs. the sim camera
     mount), rather than a pure translation.
   - At y≈-0.15, where the grasps failed, perception is about 8 mm off (y-component +8.3 mm). That is well under the
     45 mm object size, so it can at most add to a tight grasp margin; on its own it does not explain a miss. This
     agrees with the reading that **the grasp, not perception, is the weak point**. A quick fix is
     `perception.xy_offset ≈ [-0.004, -0.004]`. Checked offline on the same `random` frames, it lowers the median xy
     error from 5.6 mm to 2.0 mm and the p90 from 11.6 mm to 9.6 mm, with recall unchanged at 1.00. Better is to
     re-derive `T_pelvis_optical` for the sim.
2. **Touching same-colour objects merged; fixed by splitting on top-face area.** Before the fix, two or three
   same-colour objects became one blob (crowded: 9 of 10 misses), and the one detection landed between them (20-36 mm
   off), so the grasp hit an edge.
   - The limit is the area of the top face's convex hull in the pelvis frame, in object size². Pixel area does not
     work: a single near object reached 5453 px, while a merged pair farther away was only 6038 px. In metric units the
     two separate cleanly: one object ≤ 0.99 size², merged blobs ≥ 1.92 size². `max_top_area: 1.5` sits between them.
   - Above the limit, the top-face points are split with k-means, using the smallest k (2..`max_split`) for which every
     cluster's hull fits under the limit. Each cluster becomes one detection.
   - Result: all 9 merged objects are found, crowded xy p90 drops from 23.8 to 11.0 mm, and there are no new false
     positives. Limit: there is no split without depth (plane fallback).
3. **The home-pose hands cover the mat edges** (near_bins and wide: 7 occluded). The classical policy's own
   `home_arm_q` (arms out to the sides) avoids most of this. Any policy that looks from `ARM_HOME` must move the hands
   out of view first, or ignore blobs that touch hand pixels.
   - **Pieces cut off by a finger: fixed by joining.** Before, a finger across an object cut its blob in two, and the
     small piece became a second detection: all 9 false positives were such pieces, of an object that was also found.
     The pieces had top faces of 0.00-0.43 size² (8 of 9 under 0.08) and lay 0-25 mm from the rest of the object.
     Some showed only a side face, up to 25 mm below the top.
   - Now a piece with a top face under `min_top_area: 0.5` size² is joined to the nearest same-colour blob within
     `merge_gap: 0.03` m, as long as the pooled top face still fits one object (`max_top_area`). The pixels are pooled
     and the top face is taken from all of them, so a side-face-only piece does not pull the result down. A small piece
     with no same-colour neighbour (an object mostly behind a hand) is still detected on its own.
   - Result: false positives 9 -> 0, recall unchanged, xy p90 14.7 -> 14.0 mm.
4. **Out of view** (20 misses) happens only for layouts beyond `scene.yaml region_x/region_y`, mostly the near
   corners. Episode layouts never go there.
5. **The bins fall inside the yellow HSV range.** They are removed only by the 3D bin mask and the in-bin check, and
   that works: 0 false positives on bins. But a yellow object 2-5 mm from a bin wall can be clipped by
   `bin_pixel_margin`. Watch this on real data.
6. **Yaw from the top face already works** when the blob is clean. The next step is to use it for cube grasps, as the
   oracle's `FACE_ALIGN` does, and to add an orientation estimate for triangles, which `minAreaRect` cannot give.

## Re-running

```bash
# with the sim up; raw frames (rgb, depth, gt per frame) go to runs/ (git-ignored)
bash aaltonomous/cloud/aalto.sh py -m aaltonomous.baselines.classical.perception_eval --n 12 \
    --out aaltonomous/baselines/classical/results/perception_eval --raw runs/perception_eval_raw
# no sim: re-run detection on saved frames, e.g. to test a threshold or offset change
bash aaltonomous/cloud/aalto.sh py -m aaltonomous.baselines.classical.perception_eval \
    --offline runs/perception_eval_raw --out runs/pe_try --set '{"perception.xy_offset": [-0.004, -0.004]}'
```
