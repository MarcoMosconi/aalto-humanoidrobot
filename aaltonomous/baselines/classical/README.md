# Classical rung-1 baseline (`sort_fixed`)

Head RGB-D → HSV colour blobs + depth → object top centre in the pelvis frame → top-down "claw" grasp planned with
Pinocchio IK → carry to the bin given by the mapping (rung 1: red → left, yellow → center, blue → right) → release → home → look again, until nothing sortable is left
(`done`). Plans are pre-checked against the safety filter (0 clips). Cross-body objects are put down where the other
arm can finish. Rung 1 only: ignores shape and the instruction text. Meant as a floor and a code reference.

Perception splits a blob whose top face is too big for one object (touching same-colour objects) and joins pieces of
one object cut apart by a hand (`results/perception_eval/README.md`). **Clear look:** the head camera is fixed (no
neck, waist locked), but the hands can be in its view (sim home = `ARM_HOME`). If the object about to be picked touches
a hand in the image, or nothing is left while a hand hides part of the object band, the arms move to
`policy.look_arm_q` (out of view), the policy looks again and plans from there; at most once per pick.
`policy.look_arm_q: null` turns it off. Objects outside the camera view are not searched for (the episode band is fully
in view).

| File | What |
|---|---|
| `perception.py` | HSV masks + head depth → detections in the pelvis frame (`interface/calibration.yaml`) |
| `kinematics.py` | Pinocchio FK/IK on the G1 URDF (`third_party/xr_teleoperate/.../g1_body29_hand14.urdf`) |
| `planner.py` | Cartesian waypoints → 30 Hz joint trajectories, safety pre-check |
| `policy.py` | `ClassicalPolicy`: state machine (home → look → plan → execute; clear look when a hand hides objects); plans in a thread and returns holds meanwhile |
| `policy_server.py` | The same policy behind the submission protocol (port 8000) = **reference submission** |
| `config.yaml` | Everything tunable (HSV ranges, grasp, home pose, speeds, hand postures); `profiles.sim` = sim-tuned overrides |
| `run.py`, `sim_eval.py`, `mock_world.py` | In-process runs: mock world / real robot, Isaac sim scoring, rendered mock |
| `viz_demo.py` + `viz_demo.html` | Browser viewer: perception stages live on saved RGB-D frames (editable params, metrics vs stock), FK replay of evaluator episodes |

## Numbers (nominal sim = DR off, rung 1)

**2026-11-14, clear look on vs off** (same code incl. blob split/join, same 8 layouts, `sim_eval` in-process, seeds 0–7,
one Isaac instance on twinflow):

| | clear look off (`policy.look_arm_q: null`) | clear look on (default) |
|---|---|---|
| objects sorted | 25/38 (66 %) | **29/38 (76 %)** |
| episodes all-correct | 0/8 | 1/8 |
| safety clips / time-limit hits | 0 / 0 | 0 / 0 |
| episode time, total | 615 s | 665 s (+6 s per episode; seed 5: 119.5 s vs 60.5 s) |
| clear looks | – | 20 (12 before a pick next to a hand, 8 before `done`) |

Better on 4 seeds (+1, +1, +1, +2), equal on 3, worse on 1 (−1). +4 objects is about the +-10 % run-to-run spread,
so promising, not proven. Misses are the same kind in both: objects left on the counter, mostly triangles (grasp).

**2026-11-11** (before split/join/clear look):

| Where | Result |
|---|---|
| Aalto Isaac sim, `sim_eval` in-process, seeds 0–7 | **31/38 objects (82 %)**, 2/8 episodes all-correct, 0 clips; misses: 6 triangles (pinch slips/spins), 1 cube; 2 episodes hit the 120 s limit |
| Aalto Isaac sim, image `aalto/baseline-classical:v1` through `sim.evaluate --container --gpus none`, seeds 0–7 | **29/38 (76 %)**, 0 clips, command rate 30.0 Hz, sim gate PASS |
| Before (same day, old config) | 0/24 through the runner (seeds 0–4): perception 15–20 mm short, grasps slipped |
| Rendered mock world, seed 1 | 4/4, 0 clips |
| Real robot | not run yet (M8) |

Sim runs had 2–3 Isaac instances on twinflow at once (the sim then runs at ~0.65x real time, which makes the
30 Hz motions look ~1.5x faster to the physics). Expect +-10 % between runs.

What fixed it (2026-11-11): (1) HSV saturation floor 120 -> 70: lit top faces are washed out, so only the front face
was segmented and the "top" landed on its upper edge, 15-20 mm short; now ~4 mm. (2) Grasp point shifted 15 mm
sideways in the palm (thumb vs one finger; centred, the object wedged between index and middle). (3) `open` = the
preshape. (4) Profile `sim` (`config.yaml` bottom): sim home pose = the task's `ARM_HOME`, faster motion (0.97 rad/s),
looser IK/clearances, shorter holds. (5) 6-tick hold chunks while planning + a low-priority planner thread, so the
command rate stays at 30 Hz under CPU load.

**Sim-tuned (profile `sim`, the default; `AALTO_PROFILE=none` or `run.py --backend real` = robot base values):**
home pose, speeds/accelerations, holds, counter clearance 2 cm, IK tilt 35 deg / joint step 0.6. The HSV floor,
grasp point and `open` posture are in the base config but were also only checked in the sim.

## Run it

```bash
# in-process (conda env with pinocchio, opencv, scipy)
python -m aaltonomous.baselines.classical.run --backend mock --seed 3          # rendered mock world, scored
python -m aaltonomous.baselines.classical.run --backend mock --dry-run         # detections + plans, no motion
python -m aaltonomous.baselines.classical.sim_eval --seeds 0 1 2 3 4 5 6 7    # Isaac sim running (2nd instance: AALTO_* env vars)
python -m unittest aaltonomous/baselines/classical/test_classical.py

# browser viewer on saved data (no sim): perception_eval --raw frames + evaluator episodes -> http://localhost:8765
python -m aaltonomous.baselines.classical.viz_demo --frames runs/perception_eval_raw --episodes runs/eval

# as a policy server / submission image (CPU only, ~1.5 GB, up in < 2 s)
python -m aaltonomous.baselines.classical.policy_server                        # :8000 (PORT env to change)
docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .
python -m aaltonomous.deploy.replay --synthetic 100 --container aalto/baseline-classical:v1 --gpus none
python -m aaltonomous.deploy.runner --backend sim --container aalto/baseline-classical:v1 --team baseline --episodes 5
```

Real robot (organisers only, robot on the gantry, operator confirms every object): `python -m
aaltonomous.baselines.classical.run --backend real --network-interface <iface>` (see `run.py --help`).

Server env vars: `AALTO_TIME_LIMIT` (s, default 120; the policy skips an object it can't finish in time),
`AALTO_ARM_VEL` (overrides `motion.arm_max_vel`), `AALTO_PROFILE` (config profile, default `sim`). Calibration and `config.yaml` are baked into the image:
rebuild after changing them.
