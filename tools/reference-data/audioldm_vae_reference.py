"""AudioLDM VAE reference: diffusers' AutoencoderKL as AudioLDM configures it (Liu et al., 2023), at a tiny size.

AudioLDM's VAE is the latent-diffusion AutoencoderKL applied to a log-mel spectrogram as a one-channel image
[batch, 1, time frames, mel bins]: conv_in, down blocks of residual blocks with stride-2 downsamplers, a middle block
of residual block -> single-head self-attention -> residual block, GroupNorm, SiLU and conv_out to the posterior's
mean and log-variance, then quant_conv; the decoder mirrors it with post_quant_conv and nearest upsamplers.

Every parameter is drawn at random (seeded), the norms' included. The fixture records the configuration, the state
dict with each tensor's shape, a batch of spectrograms, the encoder's moments (after quant_conv, before diffusers
clamps the log-variance) and the decoder's output for the posterior mean. Everything runs in float64.
Default: verify; --write: regenerate.
"""
import argparse
import json
import os

import torch
from diffusers import AutoencoderKL

FIXTURE = "tests/AiDotNet.Tests/Diffusion/ReferenceData/audioldm_vae_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# AudioLDM's configuration (in_channels 1, latent_channels 8, layers_per_block 2) at small widths.
CONFIG = dict(in_channels=1, out_channels=1, latent_channels=8, block_out_channels=[8, 16, 16], layers_per_block=2,
              norm_num_groups=4)
INPUT_SHAPE = [2, 1, 16, 8]


def build():
    torch.manual_seed(5)
    blocks = len(CONFIG["block_out_channels"])
    model = AutoencoderKL(in_channels=CONFIG["in_channels"], out_channels=CONFIG["out_channels"],
                          latent_channels=CONFIG["latent_channels"],
                          down_block_types=("DownEncoderBlock2D",) * blocks,
                          up_block_types=("UpDecoderBlock2D",) * blocks,
                          block_out_channels=tuple(CONFIG["block_out_channels"]),
                          layers_per_block=CONFIG["layers_per_block"], norm_num_groups=CONFIG["norm_num_groups"])
    model = model.double().eval()
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float64) * 0.2)
    return model


def compute():
    model = build()
    state = {name: t.reshape(-1).tolist() for name, t in model.state_dict().items()}
    shapes = {name: list(t.shape) for name, t in model.state_dict().items()}
    generator = torch.Generator().manual_seed(23)
    x = torch.rand(INPUT_SHAPE, generator=generator, dtype=torch.float64) * 2 - 1
    with torch.no_grad():
        moments = model.quant_conv(model.encoder(x))
        mean = moments[:, :CONFIG["latent_channels"]]
        decoded = model.decoder(model.post_quant_conv(mean))
    return dict(config=CONFIG, state_dict=state, shapes=shapes, input=x.reshape(-1).tolist(), input_shape=list(x.shape),
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
    for key in ("input_shape", "moments_shape", "decoded_shape"):
        if data[key] != stored[key]:
            raise SystemExit(f"{key}: the fixture has {stored[key]}, the reference {data[key]}")
    worst = max((max_deviation(v, stored["state_dict"].get(k, []), k) for k, v in data["state_dict"].items()), default=0.0)
    for key in ("input", "moments", "decoded"):
        worst = max(worst, max_deviation(data[key], stored[key], key))
    print(f"max deviation from the fixture: {worst:.3e}")
    raise SystemExit(0 if worst <= 1e-12 else 1)


if __name__ == "__main__":
    main()
