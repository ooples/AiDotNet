"""CogVideoX causal 3D VAE reference: diffusers' AutoencoderKLCogVideoX (Yang et al. 2024) at a tiny size.

The network is diffusers' own CogVideoXEncoder3D and CogVideoXDecoder3D, run on whole clips: AutoencoderKLCogVideoX.encode
and decode only split long clips into frame batches that carry a convolution cache to bound memory, which the clips here
are too short for. Every parameter is drawn at random (seeded), the norms' scales and shifts included, so no term
hides behind an identity initialization.

The fixture records:
- the configuration and the state dict (float64 values) with each tensor's shape;
- for a 9-frame clip, an 8-frame clip and a single frame: the input, the encoder's moments (the posterior's mean and
  log-variance before diffusers clamps the log-variance), and the decoder's output for the posterior mean.

Everything runs in float64. Default: verify; --write: regenerate the fixture.
"""
import argparse
import json
import os

import torch
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDecoder3D, CogVideoXEncoder3D

FIXTURE = "tests/AiDotNet.Tests/Diffusion/ReferenceData/cogvideox_vae_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CONFIG = dict(in_channels=3, latent_channels=4, block_out_channels=[8, 8, 16], layers_per_block=1, norm_num_groups=4,
              temporal_compression_ratio=4)
CLIPS = {"clip9": (1, 3, 9, 16, 16), "clip8": (1, 3, 8, 16, 16), "frame1": (1, 3, 1, 16, 16)}


def build():
    torch.manual_seed(7)
    common = dict(block_out_channels=tuple(CONFIG["block_out_channels"]), layers_per_block=CONFIG["layers_per_block"],
                  norm_num_groups=CONFIG["norm_num_groups"], temporal_compression_ratio=CONFIG["temporal_compression_ratio"])
    blocks = len(CONFIG["block_out_channels"])
    encoder = CogVideoXEncoder3D(in_channels=CONFIG["in_channels"], out_channels=CONFIG["latent_channels"],
                                 down_block_types=("CogVideoXDownBlock3D",) * blocks, **common).double().eval()
    decoder = CogVideoXDecoder3D(in_channels=CONFIG["latent_channels"], out_channels=CONFIG["in_channels"],
                                 up_block_types=("CogVideoXUpBlock3D",) * blocks, **common).double().eval()
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for module in (encoder, decoder):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=generator, dtype=torch.float64) * 0.2)
    return encoder, decoder


def compute():
    encoder, decoder = build()
    state = {}
    shapes = {}
    for prefix, module in (("encoder", encoder), ("decoder", decoder)):
        for name, tensor in module.state_dict().items():
            key = f"{prefix}.{name}"
            state[key] = tensor.reshape(-1).tolist()
            shapes[key] = list(tensor.shape)
    generator = torch.Generator().manual_seed(13)
    cases = {}
    with torch.no_grad():
        for name, shape in CLIPS.items():
            x = torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1
            moments, _ = encoder(x)
            mean = moments[:, :CONFIG["latent_channels"]]
            decoded, _ = decoder(mean)
            cases[name] = dict(input=x.reshape(-1).tolist(), input_shape=list(x.shape),
                               moments=moments.reshape(-1).tolist(), moments_shape=list(moments.shape),
                               decoded=decoded.reshape(-1).tolist(), decoded_shape=list(decoded.shape))
    return dict(config=CONFIG, state_dict=state, shapes=shapes, cases=cases)


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
    worst = 0.0
    for name, case in data["cases"].items():
        for key in ("moments", "decoded"):
            worst = max(worst, max(abs(a - b) for a, b in zip(case[key], stored["cases"][name][key])))
    print(f"max deviation from the fixture: {worst:.3e}")
    raise SystemExit(0 if worst <= 1e-12 else 1)


if __name__ == "__main__":
    main()
