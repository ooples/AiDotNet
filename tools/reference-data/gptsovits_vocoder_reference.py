"""Reference fixture for GPT-SoVITS's v4/V5 vocoder and the v3/v4 mel front end (inference_webui.py init_hifigan,
mel_fn, mel_fn_v4, norm_spec).

Runs the release's own modules (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image in float64: a small
module/models.py Generator built as init_hifigan builds the v4 vocoder (resblock 1, is_bias=True, no speaker condition)
but narrower and with two upsamplings, weight norm removed; and mel_spectrogram_torch with the v3 (24 kHz, 1024/256)
and v4 (32 kHz, 1280/320) settings over a short waveform, normalized by norm_spec.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_vocoder_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_vocoder_reference.json.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ.setdefault("version", "v2")

from module.mel_processing import mel_spectrogram_torch  # noqa: E402
from module.models import Generator  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def norm_spec(x):
    return (x - -12) / (2 - -12) * 2 - 1


def main():
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)
    generator = Generator(initial_channel=100, resblock="1", resblock_kernel_sizes=[3, 7], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
                          upsample_rates=[10, 6], upsample_initial_channel=32, upsample_kernel_sizes=[20, 12], gin_channels=0, is_bias=True)
    generator = generator.double().eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        generator.conv_post.bias.copy_(torch.randn_like(generator.conv_post.bias) * 0.3)
        mel = torch.randn(1, 100, 7) * 0.5 - 4
        audio = generator(mel)
    out = {"state": {k: tolist(v) for k, v in generator.state_dict().items()}, "mel": tolist(mel), "audio": tolist(audio)}
    t = torch.arange(6000) / 24000.0
    wav = (0.4 * torch.sin(2 * math.pi * 330 * t) + 0.05 * torch.randn(6000)).unsqueeze(0)
    out["wav"] = tolist(wav)
    for name, kw in {"v3": dict(n_fft=1024, win_size=1024, hop_size=256, num_mels=100, sampling_rate=24000),
                     "v4": dict(n_fft=1280, win_size=1280, hop_size=320, num_mels=100, sampling_rate=32000)}.items():
        spec = norm_spec(mel_spectrogram_torch(wav, fmin=0, fmax=None, center=False, **kw))
        out[name] = {"shape": list(spec.shape), "mel": tolist(spec)}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "gptsovits_vocoder_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
