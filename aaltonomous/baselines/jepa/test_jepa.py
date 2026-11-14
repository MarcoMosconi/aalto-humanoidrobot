"""Offline tests of the JEPA policy pipeline (no sim, no V-JEPA 2 download: tiny encoder + rendered mock world).

    ~/miniforge3/envs/isaac/bin/python -m unittest aaltonomous/baselines/jepa/test_jepa.py -v

Checks plumbing, shapes and safety, not task success (the tiny encoder is random and the training is a few steps).
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aaltonomous.interface.tasks import FIXED_INSTRUCTION, FIXED_MAPPING, TEMPLATES

from .common import A_DIM, ActionSpace, load_config
from .goals import GoalBank, parse_mapping
from .predictor import ACPredictor, step_state

TINY = {"encoder.kind": "tiny", "encoder.pool_grid": 4, "predictor.dim": 64, "predictor.depth": 2,
        "predictor.heads": 2, "train.steps": 30, "train.batch": 4, "train.log_every": 10, "train.num_workers": 0,
        "train.warmup": 5, "planner.samples": 32, "planner.iters": 2, "planner.elites": 4}


class TestPieces(unittest.TestCase):
    def test_parse_mapping_all_templates(self):
        self.assertEqual(parse_mapping(FIXED_INSTRUCTION), FIXED_MAPPING)
        cs, bs = ["blue", "red", "yellow"], ["center", "right", "left"]
        w = {f"c{i}": c for i, c in enumerate(cs)} | {f"C{i}": c.capitalize() for i, c in enumerate(cs)} | \
            {f"b{i}": b for i, b in enumerate(bs)}
        for t in TEMPLATES:
            self.assertEqual(parse_mapping(t.format(**w)), dict(zip(cs, bs)), t)

    def test_action_space_roundtrip(self):
        cfg = load_config()
        sp = ActionSpace(cfg)
        g = np.array([0.3, 0.8])
        np.testing.assert_allclose(sp.grip_of(sp.hand_of(g)), g, atol=1e-9)
        a = sp.clip_action(np.r_[np.full(14, 9.0), 2.0, -1.0], sp.arm_lo + 0.01)
        self.assertLessEqual(np.abs(a[:14]).max(), sp.max_step + 1e-12)
        self.assertEqual(list(a[14:]), [1.0, 0.0])
        arm, hand = sp.chunk(sp.arm_lo + 0.1, sp.hand_of(np.zeros(2)), a)
        self.assertEqual(arm.shape, (cfg["macro_ticks"], 14))
        self.assertEqual(hand.shape, (cfg["macro_ticks"], 14))

    def test_predictor_causal(self):
        m = ACPredictor(8, 5, dim=32, depth=2, heads=2, max_steps=4).eval()
        z, s, a = torch.randn(2, 3, 5, 8), torch.randn(2, 3, A_DIM), torch.randn(2, 3, A_DIM)
        p = m(z, s, a)
        self.assertEqual(p.shape, z.shape)
        z2 = z.clone()
        z2[:, 2] += 1.0                                    # the future must not change earlier predictions
        torch.testing.assert_close(m(z2, s, a)[:, :2], p[:, :2])
        r = m.rollout(z[:, 0], s[:, 0], a, step_state)
        self.assertEqual(r.shape, (2, 3, 5, 8))


class TestGraspCritic(unittest.TestCase):
    """Variants plan with the classical planner, actions match the world model's space, critic scores each."""

    def test_variants_and_scoring(self):
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        from .encoder import make_encoder
        from .grasp import GraspCritic, HeldProbe, lift_end, make_variants, plan_actions, plan_variant
        env = make_mock_env(seed=1)
        obs = env.reset()
        pol = ClassicalPolicy(log=None)
        dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
        self.assertTrue(dets)
        d = dets[0]
        vs = make_variants(d, np.random.default_rng(0), n=6)
        self.assertEqual(vs[0].name, "classical")
        cfg = load_config(None, TINY)
        space = ActionSpace(cfg)
        acts, n_ok = [], 0
        hint0, w0, g0 = pol.planner.kin.hint.copy(), pol.planner.kin.w_hint, dict(pol.planner.cfg["grasp"])
        for v in vs:
            p = plan_variant(pol.planner, d, v, pol.home, pol.post["open"])
            if p.ok and lift_end(p):
                n_ok += 1
                a = plan_actions(space, p, pol.home, pol.post["open"], lift_end(p))
                self.assertEqual(a.shape[1], A_DIM)
                np.testing.assert_allclose(pol.home + a[:, :14].sum(0), p.arm[lift_end(p) - 1], atol=1e-6)
                acts.append(a)
        np.testing.assert_array_equal(pol.planner.kin.hint, hint0)      # planner state restored
        self.assertEqual(pol.planner.kin.w_hint, w0)
        self.assertEqual(pol.planner.cfg["grasp"], g0)
        self.assertGreater(n_ok, 1)
        enc = make_encoder(cfg["encoder"], "cpu")
        model = ACPredictor(enc.dim, enc.n_tokens, 32, 1, 2, max_steps=5).eval()
        critic = GraspCritic(model, HeldProbe(enc.dim).eval(), enc, space, "cpu")
        sc = critic.score(critic.encode(obs), space.state(pol.home, pol.post["open"]), acts)
        self.assertEqual(sc.shape, (len(acts),))
        self.assertTrue(((sc >= 0) & (sc <= 1)).all())


class TestPipelineMock(unittest.TestCase):
    """collect (classical teacher, mock) -> encode (tiny) -> train -> JepaPolicy on the mock env, 0 clips."""

    def test_end_to_end(self):
        from . import collect, encode, train
        from .policy import JepaPolicy
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            collect.main(["--backend", "mock", "--seeds", "0", "--max-ticks", "400", "--out", str(d / "ep")])
            self.assertTrue((d / "ep/mock_seed0000.npz").exists())
            ov = json.dumps(TINY)
            encode.main(["--episodes", str(d / "ep"), "--out", str(d / "lat"), "--set", ov])
            train.main(["--latents", str(d / "lat"), "--out", str(d / "ck"), "--set", ov, "--device", "cpu"])
            self.assertTrue((d / "ck/best.pt").exists())

            from .collect import make_env
            env, _ = make_env("mock", None, 120.0)
            pol = JepaPolicy.from_checkpoint(d / "ck/best.pt", d / "lat", device="cpu", overrides=TINY,
                                             safety=env.filter.cfg, log=None)
            self.assertIsInstance(pol.bank, GoalBank)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            clips = 0
            for _ in range(60):
                obs, info = env.step(pol.act(obs))
                clips += int(info["clipped"])
                if info["episode_over"]:
                    break
            self.assertEqual(clips, 0)
            self.assertGreater(len(pol.trace), 0, "the policy never planned a macro step")
            env.close()


if __name__ == "__main__":
    unittest.main()
