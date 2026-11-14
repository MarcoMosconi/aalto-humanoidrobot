"""Frozen image encoders -> grids of latent tokens.

    enc = make_encoder(cfg["encoder"], device="cuda")
    z = enc.encode({"head_rgb": (B,H,W,3) uint8, ...})      # (B, n_tokens, D) float32, layer-normed

`vjepa2`: Meta's V-JEPA 2 video encoder from Hugging Face (frozen). A single frame is fed as a 2-frame clip (the
tubelet is 2 frames), giving 16x16 patch tokens at 256 px; they are average-pooled to `pool_grid`^2 tokens per
camera and layer-normed (as in V-JEPA 2-AC, which plans on normalised encoder tokens). Cameras are concatenated
along the token axis.

`tiny`: a fixed, randomly initialised conv net (seeded). No pretrained weights, no download: only for unit tests and
for checking the pipeline end to end on a machine without the V-JEPA 2 weights. It is NOT a useful representation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(imgs, size, device):
    """(B,H,W,3) uint8 -> (B,3,size,size) float normalised. Aspect ratio is not kept (same squash in train and test)."""
    x = torch.as_tensor(np.ascontiguousarray(imgs), device=device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - m) / s


class Encoder(nn.Module):
    dim: int
    tokens_per_cam: int

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.cameras = list(cfg["cameras"])
        self.size = int(cfg["image_size"])
        self.grid = int(cfg["pool_grid"])
        self.device = torch.device(device)

    @property
    def n_tokens(self):
        return self.tokens_per_cam * len(self.cameras)

    def _encode_cam(self, x):  # (B,3,S,S) -> (B, g*g, D)
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, images: dict, batch=32):
        outs = []
        n = len(images[self.cameras[0]])
        for i in range(0, n, batch):
            toks = []
            for cam in self.cameras:
                x = _to_tensor(images[cam][i:i + batch], self.size, self.device)
                toks.append(self._encode_cam(x))
            z = torch.cat(toks, 1).float()
            outs.append(F.layer_norm(z, (z.shape[-1],)))
        return torch.cat(outs, 0)


class VJEPA2Encoder(Encoder):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        from transformers import AutoModel
        dt = torch.float16 if (cfg.get("dtype") == "float16" and self.device.type == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(cfg["hf_id"], dtype=dt).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dtype = dt
        self.dim = int(self.model.config.hidden_size)
        self.patch = int(self.model.config.patch_size)
        self.tokens_per_cam = self.grid * self.grid

    def _encode_cam(self, x):
        b = x.shape[0]
        clip = x.to(self.dtype)[:, None].repeat(1, 2, 1, 1, 1)               # (B, T=2, 3, S, S): one tubelet
        h = self.model(pixel_values_videos=clip, skip_predictor=True).last_hidden_state  # (B, (S/p)^2, D)
        n = self.size // self.patch
        h = h[:, -n * n:].reshape(b, n, n, -1).permute(0, 3, 1, 2).float()
        h = F.adaptive_avg_pool2d(h, self.grid)
        return h.flatten(2).transpose(1, 2)


class TinyEncoder(Encoder):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        d = int(cfg.get("tiny_dim", 64))
        g = torch.Generator().manual_seed(0)
        self.net = nn.Sequential(nn.Conv2d(3, 32, 5, 4, 2), nn.GELU(), nn.Conv2d(32, d, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(d, d, 3, 2, 1))
        with torch.no_grad():
            for p in self.net.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / np.sqrt(max(p[0].numel(), 1))))
        self.net.to(self.device).eval()
        self.dim = d
        self.tokens_per_cam = self.grid * self.grid

    def _encode_cam(self, x):
        h = F.adaptive_avg_pool2d(self.net(x), self.grid)
        return h.flatten(2).transpose(1, 2)


def make_encoder(cfg, device="cuda" if torch.cuda.is_available() else "cpu") -> Encoder:
    kind = cfg["kind"]
    if kind == "vjepa2":
        return VJEPA2Encoder(cfg, device)
    if kind == "tiny":
        return TinyEncoder(cfg, device)
    raise ValueError(f"unknown encoder kind {kind!r}")
