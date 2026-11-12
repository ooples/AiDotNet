"""HiFT generator reference: FunAudioLLM/CosyVoice's HiFTGenerator and ConvRNNF0Predictor (cosyvoice/hifigan), the
vocoder of CosyVoice 1, 2 and 3. Pass a checkout of FunAudioLLM/CosyVoice with --cosyvoice.

Builds a tiny randomly initialized generator (seeded; 22.05 kHz, so the original SineGen), runs it in float64 and
records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hift_reference.json: the F0 of a mel, the merged excitation
of fixed harmonics (SineGen draws its phases and noise from torch's generator, so the harmonics are given), the waveform
decode() makes from the mel and that excitation, and the waveform after one SGD step (lr 0.1) on the mean square of the
first waveform. The Hann window is float64 (the reference rounds it to float32). Default: verify; --write: regenerate.
"""
import argparse
import base64
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hift_reference.json"
CONFIG = dict(in_channels=8, base_channels=16, nb_harmonics=2, sampling_rate=22050, nsf_alpha=0.1, nsf_sigma=0.003,
              nsf_voiced_threshold=10, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
              istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 5],
              resblock_dilation_sizes=[[1, 3], [1, 3]], source_resblock_kernel_sizes=[3, 5],
              source_resblock_dilation_sizes=[[1, 3], [1, 3]], lrelu_slope=0.1, audio_limit=0.99)
F0_CHANNELS = 16
FRAMES = 6
LEARNING_RATE = 0.1


def build(root, state=None):
    sys.path.insert(0, root)
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.hifigan.generator import HiFTGenerator
    torch.manual_seed(2294)
    model = HiFTGenerator(**CONFIG, f0_predictor=ConvRNNF0Predictor(1, CONFIG["in_channels"], F0_CHANNELS)).eval()
    with torch.no_grad():
        # Snake's alpha starts at 1; spread it so the activations are exercised per channel.
        for name, p in model.named_parameters():
            if name.endswith(".alpha"):
                p.copy_(0.5 + torch.rand_like(p))
    if state is not None:
        model.load_state_dict(state, strict=True)
    # The reference stores its Hann window in float32 (get_window(...).astype(np.float32)), a rounding a float64 port does
    # not make; the window is recomputed in float64 here, as the other fixtures remove their references' float32 steps.
    from scipy.signal import get_window
    model.stft_window = torch.from_numpy(get_window("hann", CONFIG["istft_params"]["n_fft"], fftbins=True))
    return model


def run(model, mel, harmonics):
    with torch.no_grad():
        f0 = model.f0_predictor(mel)
        source = model.m_source.l_tanh(model.m_source.l_linear(harmonics)).transpose(1, 2)    # [1, 1, samples]
    audio = model.decode(x=mel, s=source)
    loss = (audio ** 2).mean()
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
        after = model.decode(x=mel, s=source)
    return f0, source, audio, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True, help="a checkout of FunAudioLLM/CosyVoice")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture")
    args = parser.parse_args()
    scale = CONFIG["upsample_rates"][0] * CONFIG["upsample_rates"][1] * CONFIG["istft_params"]["hop_len"]
    if args.write:
        model = build(args.cosyvoice)
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        torch.manual_seed(7)
        mel = torch.randn(1, CONFIG["in_channels"], FRAMES)
        harmonics = 0.1 * torch.randn(1, FRAMES * scale, CONFIG["nb_harmonics"] + 1)
        f0, source, audio, after = run(model.double(), mel.double(), harmonics.double())
        data = {"config": CONFIG, "f0_channels": F0_CHANNELS, "learning_rate": LEARNING_RATE,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "mel": [float(v) for v in mel.double().reshape(-1).tolist()],
                "harmonics": [float(v) for v in harmonics.double().reshape(-1).tolist()]}
        for key, value in [("f0", f0), ("source", source), ("audio", audio), ("audio_after_sgd", after)]:
            data[key] = [float(v) for v in value.detach().reshape(-1).tolist()]
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(args.cosyvoice, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    mel = torch.tensor(data["mel"], dtype=torch.float64).reshape(1, CONFIG["in_channels"], FRAMES)
    harmonics = torch.tensor(data["harmonics"], dtype=torch.float64).reshape(1, FRAMES * scale, CONFIG["nb_harmonics"] + 1)
    f0, source, audio, after = run(model.double(), mel, harmonics)
    check = _fixture.Comparison("HiFT")
    for key, value in [("f0", f0), ("source", source), ("audio", audio), ("audio_after_sgd", after)]:
        check.floats(key, data[key], value.detach().reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
