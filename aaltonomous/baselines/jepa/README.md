# JEPA world-model policy (experimental)

A V-JEPA 2-AC style policy: plan in the latent space of a frozen self-supervised video encoder with a learned
action-conditioned predictor. It sits next to the classical baseline and does not change it (it only reads the
classical `config.yaml` for the Dex3 `open` / `closed` postures and the home pose, and uses the classical policy as
the teacher that collects training data).

```
head RGB ──► V-JEPA 2 ViT-L (frozen, HF facebook/vjepa2-vitl-fpc64-256) ──► 8x8x1024 tokens z_t
                                                                               │
demo latent bank ──► nearest demo frame ──► sub-goal z_goal (k steps later)    │
                                                                               ▼
CEM over (H x 16) actions ──► predictor P(z_t, s_t, a_t..) ──► ẑ_{t+H}   energy = |ẑ_{t+H} − z_goal|₁
        │
        └─► first macro step (8 ticks: Δ arm 14 + grip L/R) ──► shadow safety check ──► (8,14) chunk
```

| File | What |
|---|---|
| `config.yaml` | Encoder, action space, predictor, training, CEM, goal bank, policy timing |
| `common.py` | Config + `ActionSpace`: 16-d planning space (Δ arm joints, one grip per hand on the open→closed line) |
| `encoder.py` | Frozen V-JEPA 2 (single frame as one tubelet, pooled tokens) and a `tiny` random CNN for tests |
| `predictor.py` | `ACPredictor`: block-causal transformer over [action, state, frame tokens] per step |
| `data.py` | Episode `.npz` format (sampled every macro step) + `python -m ...jepa.data <lerobot_dir>` importer |
| `collect.py` | Roll out the classical teacher in sim / rendered mock and record episodes |
| `encode.py` | Cache encoder latents per episode (training never runs the ViT) |
| `train.py` | Teacher-forcing + rollout L1 loss in latent space |
| `goals.py` | `GoalBank`: sub-goal retrieval from demo latents, rung-2 instruction → mapping parser, `done` detection |
| `planner.py` | `CEMPlanner`: latent MPC with joint-limit-aware action bounds |
| `policy.py` | `JepaPolicy`: home → settle → think (thread) → one macro step → … ; holds while thinking |
| `policy_server.py` | Submission protocol server (port 8000); Docker: `aaltonomous/docker/jepa.Dockerfile` |
| `sim_eval.py` | In-process scoring on the Isaac sim / mock, with the per-step planner trace |
| `test_jepa.py` | Offline tests (tiny encoder, mock world): pipeline, causality, safety (0 clips) |

## Run it

Everything in the `isaac` env (`~/miniforge3/envs/isaac/bin/python`, has torch + transformers). Sim running for the
`--backend sim` steps (`TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh`, `AALTO_DR=train` for
variety).

```bash
# 1. data: classical teacher in the sim (DR on), + optional random detours so the model sees off-teacher actions
python -m aaltonomous.baselines.jepa.collect --backend sim --seeds 0-99 --explore 0.2 --out runs/jepa/episodes
#    and/or the real demos (LeRobot v3.0 drop):  python -m aaltonomous.baselines.jepa.data <drop_dir> --out runs/jepa/episodes
# 2. latents (V-JEPA 2 ViT-L, ~45 ms/frame on one GPU; first run downloads ~1.3 GB from Hugging Face)
python -m aaltonomous.baselines.jepa.encode --episodes runs/jepa/episodes --out runs/jepa/latents
# 3. predictor
python -m aaltonomous.baselines.jepa.train --latents runs/jepa/latents --out runs/jepa/ckpt
# 4. score it
python -m aaltonomous.baselines.jepa.sim_eval --ckpt runs/jepa/ckpt/best.pt --bank runs/jepa/latents --seeds 0 1 2 --log runs/jepa/eval.jsonl
# 5. submission
python -m aaltonomous.baselines.jepa.policy_server --ckpt runs/jepa/ckpt/best.pt --bank runs/jepa/latents
mkdir -p jepa_weights && cp runs/jepa/ckpt/best.pt jepa_weights/ && cp -r runs/jepa/latents jepa_weights/bank
docker build -f aaltonomous/docker/jepa.Dockerfile -t myteam/jepa:v1 .

python -m unittest aaltonomous/baselines/jepa/test_jepa.py -v      # offline, ~1-2 min on CPU
```

Use the same latent dir for training and as the goal bank, or a separate one (e.g. only real demos) for the bank;
both must come from the same encoder config (`manifest.json`; the checkpoint stores it and the policy uses it).

## Live demo (watch it in the browser)

```bash
python -m aaltonomous.baselines.jepa.live_demo --critic runs/jepa/crop_critic.pt                 # Isaac sim running
python -m aaltonomous.baselines.jepa.live_demo --backend mock --critic runs/jepa/crop_critic.pt  # no sim needed
# open http://localhost:8766   (remote: ssh -L 8766:localhost:8766 <host>)
```

Episodes run back to back. The page streams the head camera (~10 fps) with the target, chosen grasp point, drop
arrow and both hands overlaid; each grasp decision as the object-centred crops the world model imagines from, with
p(held) per candidate and the chosen one highlighted; whether the object was actually held after the lift (sim);
an event feed and the running score. `--policy classical` shows the stock planner in the same view.

## Rung-1 grasp critic (classical + JEPA)

`grasp.py`, `collect_grasps.py`, `crop_critic.py`, `grasp_eval.py`: the classical planner proposes grasp variants
per object; V-JEPA 2 encodes a crop around the grasp, the action-conditioned predictor imagines approach + close +
lift in latent space, a probe reads p(held) and the best candidate runs. On the full head frame "held" is barely
readable even from the real lift-end latent (AUC 0.68); on the crop it is (0.93), so the critic works on crops.

## Design notes and known limits

- **Goals are the hard part.** V-JEPA 2-AC reaches goal images with short plans; sorting is long-horizon. Here the
  sub-goal is "where the nearest demo was `goals.lookahead` macro steps later", so the policy follows demos it can
  match. Quality depends on bank coverage (many layouts, DR) and on the match (`goals.proprio_weight`). Alternatives
  worth trying: phase-specific goal images per object, or the classical perception choosing the object and the
  JEPA planner doing the grasp.
- **Action space** is joint-space Δ (14) + 2 grips, not end-effector deltas like V-JEPA 2-AC (no IK in the loop). Arm
  steps are bounded at 0.9 × the filter's 1 rad/s and every executed step is pre-checked on a shadow `SafetyFilter`
  (halved up to 3 times, else hold), so clips should stay at 0. Self-collision and fingertips vs table are **not**
  checked (same as the filter).
- **Speed:** think ≈ encode (~45 ms) + CEM (4 iters × 256 samples × 3-step rollout) per 8-tick macro step, run in a
  thread with 3-tick holds, so motion is about 60-70 % duty cycle. Lower `planner.samples`/`iters` or plan the next
  step while the current one executes if this is too slow.
- **Head camera only** by default. Add the wrist cameras in `encoder.cameras` (all of collect / encode / train must
  use them) for finer grasping; the sim wrist cameras use placeholder poses.
- Rung 1 and 2 via the goal bank (demos with the same mapping); rung 3 needs demos of the lid task.
- Status: pipeline tested offline (mock world, tiny encoder) and the V-JEPA 2 encoder checked on the GPU. No sim
  score yet: it needs a collected dataset and a trained predictor first.
