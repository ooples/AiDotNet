"""Stable Video Diffusion VAE reference: diffusers' AutoencoderKLTemporalDecoder (Blattmann et al., 2023) at a tiny size.

SVD's VAE runs the latent-diffusion KL encoder and its 1x1 quant conv on every frame. Its decoder (TemporalDecoder) is
a 3x3 conv in, a middle block of spatio-temporal residual blocks around single-head self-attention, up blocks of
spatio-temporal residual blocks with nearest upsamplers, GroupNorm, SiLU and conv_out, then a (3, 1, 1) Conv3d over
time. Each spatio-temporal block is a 2D residual block per frame, a residual block of (3, 1, 1) Conv3ds across frames
and a learned blend of the two.

Every parameter, the norms and the blend factors included, is set by a closed formula of its state-dict name and its
flat index (weight_value below), which the C# test evaluates too, so the fixture carries each tensor's shape rather
than its 340k values. The fixture records the configuration, each tensor's shape, a clip in [batch, channels, frames, height, width] (this
repository's layout; diffusers takes frames stacked on the batch axis), the encoder's moments after quant_conv in the
same layout, and the decoder's output for the posterior mean. Everything runs in float64.
Default: verify; --write: regenerate.
"""
import argparse
import json
import math
import os

import torch
from diffusers import AutoencoderKLTemporalDecoder

FIXTURE = "tests/AiDotNet.Tests/Diffusion/ReferenceData/svd_vae_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# SVD's configuration (in 3, latent 4, layers_per_block 2; diffusers fixes 32 groups) at the smallest widths 32 groups
# allow.
CONFIG = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=[32, 32], layers_per_block=2)
INPUT_SHAPE = [1, 3, 3, 8, 8]  # batch, channels, frames, height, width


def build():
    torch.manual_seed(5)
    blocks = len(CONFIG["block_out_channels"])
    model = AutoencoderKLTemporalDecoder(in_channels=CONFIG["in_channels"], out_channels=CONFIG["out_channels"],
                                         latent_channels=CONFIG["latent_channels"],
                                         down_block_types=("DownEncoderBlock2D",) * blocks,
                                         block_out_channels=tuple(CONFIG["block_out_channels"]),
                                         layers_per_block=CONFIG["layers_per_block"])
    model = model.double().eval()
    with torch.no_grad():
        for name, p in model.state_dict().items():
            p.copy_(torch.tensor([weight_value(name, i) for i in range(p.numel())], dtype=torch.float64).reshape(p.shape))
    return model


def name_offset(name):
    """An integer of the name, so every tensor gets its own phase."""
    return sum(ord(c) * (k + 1) for k, c in enumerate(name)) % 1000


def weight_value(name, index):
    """A bounded, irregular value for element index of the named tensor (AiDotNet's test computes the same)."""
    c = name_offset(name) / 100.0
    return 0.2 * math.sin(0.37 * index + c) * math.cos(0.11 * index + 0.5 * c)


def to_frames(x):
    b, c, f, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)


def to_clip(x, frames):
    bf, c, h, w = x.shape
    return x.reshape(bf // frames, frames, c, h, w).permute(0, 2, 1, 3, 4)


def compute():
    model = build()
    shapes = {name: list(t.shape) for name, t in model.state_dict().items()}
    generator = torch.Generator().manual_seed(23)
    x = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float64) * 2 - 1
    frames = INPUT_SHAPE[2]
    with torch.no_grad():
        moments = model.quant_conv(model.encoder(to_frames(x)))
        mean = moments[:, :CONFIG["latent_channels"]]
        decoded = model.decode(mean, num_frames=frames).sample
    moments, decoded = to_clip(moments, frames), to_clip(decoded, frames)
    return dict(config=CONFIG, shapes=shapes, input=x.reshape(-1).tolist(), input_shape=list(x.shape),
                moments=moments.reshape(-1).tolist(), moments_shape=list(moments.shape),
                decoded=decoded.reshape(-1).tolist(), decoded_shape=list(decoded.shape))


def max_deviation(expected, actual, what):
    """The largest elementwise difference; a length mismatch is a stale fixture, not a prefix to compare."""
    if len(expected) != len(actual):
        raise SystemExit(f"{what}: the fixture has {len(actual)} values, the reference {len(expected)}")
    return max((abs(a - b) for a, b in zip(expected, actual)), default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = compute()
    path = os.path.join(REPO, FIXTURE)
    if args.write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote {path}")
        return
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    if data["config"] != stored["config"]:
        raise SystemExit("the fixture was written for another configuration")
    for key in ("shapes", "input_shape", "moments_shape", "decoded_shape"):
        if data[key] != stored[key]:
            raise SystemExit(f"{key}: the fixture has {stored[key]}, the reference {data[key]}")
    worst = 0.0
    for key in ("input", "moments", "decoded"):
        worst = max(worst, max_deviation(data[key], stored[key], key))
    print(f"max deviation from the fixture: {worst:.3e}")
    raise SystemExit(0 if worst <= 1e-12 else 1)


if __name__ == "__main__":
    main()
