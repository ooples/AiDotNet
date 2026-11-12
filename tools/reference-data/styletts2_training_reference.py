"""StyleTTS 2 training reference: yl4579/StyleTTS2 (losses.py, Modules/slmadv.py, Modules/discriminators.py), the
WavLM of transformers' WavLMModel and torchaudio's Resample, the code StyleTTS 2's training objectives follow.

Pass a checkout of yl4579/StyleTTS2 with --styletts2 (it is not vendored here). Records, in float64:
- a tiny randomly initialized WavLM's hidden states (WavLMModel, output_hidden_states, evaluation mode);
- torchaudio Resample(24000, 16000) of a waveform (its kernel built in float64; the transform caches it in float32);
- the multi-resolution mel spectral-convergence loss (MultiResolutionSTFTLoss, built in float64);
- the least-squares generator / discriminator losses with feature matching and TPR on synthetic scores and maps;
- WavLMLoss's feature matching, SLM generator and discriminator losses and discriminator_forward, with a tiny
  WavLMDiscriminator, computed as the class computes them but with the batch axis kept (its squeeze() drops it at
  batch size 1);
- slmadv's differentiable duration upsampling of duration logits.
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts2_training_reference.json. Default: verify; --write: regenerate.

Needs torch, torchaudio, transformers, einops, einops-exts, munch, pyyaml and scipy.
"""
import argparse
import base64
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts2_training_reference.json"
WAVLM = dict(hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16, conv_dim=(8,) * 7,
             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=320, max_bucket_distance=800)
WD = dict(slm_hidden=8, slm_layers=3, initial_channel=4)
SAMPLES = 7200
LOGITS = (7, 6)
SIGMA = 1.5


def load(root):
    sys.path.insert(0, root)
    import losses
    from Modules.discriminators import WavLMDiscriminator
    from transformers import WavLMConfig, WavLMModel
    return losses, WavLMDiscriminator, WavLMConfig, WavLMModel


def build(loaded, state=None):
    _, WavLMDiscriminator, WavLMConfig, WavLMModel = loaded
    torch.manual_seed(2294)
    wavlm = WavLMModel(WavLMConfig(**WAVLM))
    wd = WavLMDiscriminator(**WD)
    module = torch.nn.ModuleDict({"wavlm": wavlm, "wd": wd})
    if state is None:
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            for layer in wavlm.encoder.layers:
                layer.attention.gru_rel_pos_const.copy_(torch.rand(layer.attention.gru_rel_pos_const.shape, generator=g) + 0.5)
            # Group-norm and layer-norm affine parameters away from 1 and 0, so loading them is tested.
            for m in module.modules():
                if isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    else:
        module.load_state_dict(state, strict=True)
    return module.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    t = torch.arange(SAMPLES, dtype=torch.float64) / 24000
    real = (0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.randn(SAMPLES, generator=g, dtype=torch.float64)).float()
    fake = (0.25 * torch.sin(2 * torch.pi * 230 * t + 0.3) + 0.05 * torch.randn(SAMPLES, generator=g, dtype=torch.float64)).float()
    scores_real = [torch.randn(1, n, generator=g, dtype=torch.float32) for n in (5, 9, 4)]
    scores_fake = [torch.randn(1, n, generator=g, dtype=torch.float32) for n in (5, 9, 4)]
    maps_real = [[torch.randn(1, 3, k, generator=g, dtype=torch.float32) for k in (6, 4)] for _ in range(3)]
    maps_fake = [[torch.randn(1, 3, k, generator=g, dtype=torch.float32) for k in (6, 4)] for _ in range(3)]
    logits = torch.randn(*LOGITS, generator=g, dtype=torch.float32) * 1.5
    return real, fake, scores_real, scores_fake, maps_real, maps_fake, logits


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def run(loaded, module):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return run_double(loaded, module)
    finally:
        torch.set_default_dtype(previous)


def run_double(loaded, module):
    import torchaudio
    losses = loaded[0]
    real, fake, scores_real, scores_fake, maps_real, maps_fake, logits = inputs()
    real, fake = real.double(), fake.double()
    out = {}
    with torch.no_grad():
        resample = torchaudio.transforms.Resample(24000, 16000, dtype=torch.float64)
        real16, fake16 = resample(real[None]), resample(fake[None])
        out["resampled"] = flat(real16)
        wavlm, wd = module["wavlm"], module["wd"]
        states_real = wavlm(input_values=real16, output_hidden_states=True).hidden_states
        states_fake = wavlm(input_values=fake16, output_hidden_states=True).hidden_states
        out["wavlm_states"] = [flat(s) for s in states_real]

        def stacked(states):
            return torch.stack(states, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        out["slm_feature"] = [float(sum(torch.mean(torch.abs(r - f)) for r, f in zip(states_real, states_fake)))]
        out["slm_generator"] = [float(torch.mean((1 - wd(stacked(states_fake))) ** 2))]
        out["slm_discriminator"] = [float(torch.mean((1 - wd(stacked(states_real))) ** 2) + torch.mean(wd(stacked(states_fake)) ** 2))]
        out["slm_scores"] = flat(wd(stacked(states_real)))

        stft = losses.MultiResolutionSTFTLoss()
        out["mel_loss"] = [float(stft(fake[None], real[None]))]

        sr = [s.double() for s in scores_real]
        sf = [s.double() for s in scores_fake]
        mr = [[m.double() for m in maps] for maps in maps_real]
        mf = [[m.double() for m in maps] for maps in maps_fake]
        generator = (losses.generator_loss(sf)[0] + losses.feature_loss(mr, mf) + losses.generator_TPRLS_loss(sr, sf))
        discriminator = losses.discriminator_loss(sr, sf)[0] + losses.discriminator_TPRLS_loss(sr, sf)
        out["generator_loss"] = [float(generator)]
        out["discriminator_loss"] = [float(discriminator)]

        # slmadv's differentiable duration modeling for one utterance.
        org = logits.double()
        q = torch.sigmoid(org)
        dur = q.sum(axis=-1)
        length = int(torch.round(q.sum()).item())
        t = torch.arange(0, length).unsqueeze(0).expand((len(q), length))
        loc = torch.cumsum(dur, dim=0) - dur / 2
        h = torch.exp(-0.5 * torch.square(t - (length - loc.unsqueeze(-1))) / SIGMA ** 2)
        conv = F.conv1d(org.unsqueeze(0), h.unsqueeze(1), padding=h.shape[-1] - 1, groups=len(q))[..., :length]
        out["alignment"] = flat(F.softmax(conv.squeeze(), dim=0))
        out["alignment_frames"] = length
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styletts2", required=True, help="a checkout of yl4579/StyleTTS2")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    os.chdir(args.styletts2)
    loaded = load(args.styletts2)
    real, fake, scores_real, scores_fake, maps_real, maps_fake, logits = inputs()
    shared = {"wavlm": {k: list(v) if isinstance(v, tuple) else v for k, v in WAVLM.items()}, "wd": WD, "sigma": SIGMA,
              "real": flat(real), "fake": flat(fake), "scores_real": [flat(s) for s in scores_real],
              "scores_fake": [flat(s) for s in scores_fake], "maps_real": [[flat(m) for m in maps] for maps in maps_real],
              "maps_fake": [[flat(m) for m in maps] for maps in maps_fake], "map_shapes": [[list(m.shape) for m in maps] for maps in maps_real],
              "logits": flat(logits), "logits_shape": list(LOGITS)}
    if args.write:
        state = {k: v.float().clone().contiguous() for k, v in build(loaded).state_dict().items()}
        data = {"shared": shared, "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                **run(loaded, build(loaded, state).double())}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    again = run(loaded, build(loaded, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double())
    check = _fixture.Comparison("StyleTTS 2 training")
    for key, value in again.items():
        if key == "wavlm_states":
            for i, (a, b) in enumerate(zip(data[key], value)):
                check.floats(f"{key} {i}", a, b, 1e-12, float)
        elif key == "alignment_frames":
            check.exact(key, data[key], value)
        else:
            check.floats(key, data[key], value, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
