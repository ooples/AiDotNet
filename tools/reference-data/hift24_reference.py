"""HiFT at 24 kHz reference, CosyVoice 2's vocoder: FunAudioLLM/CosyVoice's HiFTGenerator with a three-stage upsampler
(the release's 8·5·3, narrowed here to 2·3·2) and three source branches, whose sine source at any rate but 22.05 kHz is
SineGen2. Pass a checkout of FunAudioLLM/CosyVoice with --cosyvoice.

Reuses hift_reference.py's build and run with this configuration, recording in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hift24_reference.json the F0 of a mel, the merged excitation of given
harmonics, decode()'s waveform and the waveform after one SGD step; and SineGen2's harmonics for an F0 contour with
voiced and unvoiced frames, its initial phases (torch.rand) and noise (torch.randn_like) pinned (the voicing mask in
F0's dtype, where the reference rounds it to float32). Default: verify;
--write: regenerate.
"""
import argparse
import base64
from unittest import mock

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture
import hift_reference as base

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hift24_reference.json"
base.CONFIG = dict(in_channels=8, base_channels=16, nb_harmonics=2, sampling_rate=24000, nsf_alpha=0.1, nsf_sigma=0.003,
                   nsf_voiced_threshold=10, upsample_rates=[2, 3, 2], upsample_kernel_sizes=[4, 5, 4],
                   istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 5],
                   resblock_dilation_sizes=[[1, 3], [1, 3]], source_resblock_kernel_sizes=[3, 5, 3],
                   source_resblock_dilation_sizes=[[1, 3], [1, 3], [1, 3]], lrelu_slope=0.1, audio_limit=0.99)
CONFIG = base.CONFIG
SCALE = 2 * 3 * 2 * 4
F0 = [0.0, 120.0, 135.5, 150.25, 0.0, 0.0, 210.0, 205.0, 8.0, 180.0]


def sinegen(model, phases, noise):
    gen = model.m_source.l_sin_gen
    # _f02uv casts the voicing to float32, which rounds the noise amplitudes (0.003, 0.1 / 3); keep F0's dtype instead.
    gen._f02uv = lambda f0: (f0 > gen.voiced_threshold).to(f0.dtype)
    f0 = torch.tensor(F0, dtype=torch.float64).repeat_interleave(SCALE).reshape(1, -1, 1)
    with mock.patch.object(torch, "rand", return_value=phases.clone()), \
            mock.patch.object(torch, "randn_like", return_value=noise.clone()):
        sines, _, _ = gen(f0)
    return sines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    harmonics_count = CONFIG["nb_harmonics"] + 1
    frames = base.FRAMES
    if args.write:
        model = base.build(args.cosyvoice)
        assert type(model.m_source.l_sin_gen).__name__ == "SineGen2"
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        torch.manual_seed(7)
        mel = torch.randn(1, CONFIG["in_channels"], frames)
        harmonics = 0.1 * torch.randn(1, frames * SCALE, harmonics_count)
        phases = torch.rand(1, harmonics_count).double()
        phases[:, 0] = 0
        noise = torch.randn(1, len(F0) * SCALE, harmonics_count).double()
        model = model.double()
        f0, source, audio, after = base.run(model, mel.double(), harmonics.double())
        data = {"config": CONFIG, "f0_channels": base.F0_CHANNELS, "learning_rate": base.LEARNING_RATE, "frames": frames,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "mel": [float(v) for v in mel.double().reshape(-1).tolist()],
                "harmonics": [float(v) for v in harmonics.double().reshape(-1).tolist()],
                "f0_contour": F0, "phases": phases.reshape(-1).tolist(), "noise": noise.reshape(-1).tolist(),
                "sines": sinegen(base.build(args.cosyvoice, state).double(), phases, noise).reshape(-1).tolist()}
        for key, value in [("f0", f0), ("source", source), ("audio", audio), ("audio_after_sgd", after)]:
            data[key] = [float(v) for v in value.detach().reshape(-1).tolist()]
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    state = load_safetensors(base64.b64decode(data["safetensors_base64"]))
    model = base.build(args.cosyvoice, state).double()
    mel = torch.tensor(data["mel"], dtype=torch.float64).reshape(1, CONFIG["in_channels"], frames)
    harmonics = torch.tensor(data["harmonics"], dtype=torch.float64).reshape(1, frames * SCALE, harmonics_count)
    f0, source, audio, after = base.run(model, mel, harmonics)
    phases = torch.tensor(data["phases"], dtype=torch.float64).reshape(1, harmonics_count)
    noise = torch.tensor(data["noise"], dtype=torch.float64).reshape(1, len(F0) * SCALE, harmonics_count)
    sines = sinegen(base.build(args.cosyvoice, state).double(), phases, noise)
    check = _fixture.Comparison("HiFT 24 kHz")
    for key, value in [("f0", f0), ("source", source), ("audio", audio), ("audio_after_sgd", after), ("sines", sines)]:
        check.floats(key, data[key], value.detach().reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
