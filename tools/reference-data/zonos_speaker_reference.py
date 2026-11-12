"""Reference fixture for Zonos's speaker model: Zyphra/Zonos at bc40d98 (zonos/speaker_cloning.py's SpeakerEmbeddingLDA,
ResNet293_based with logFbankCal and torchaudio 2.6's Resample) in float64.

Two cases:
- tiny: ResNet293_based(in_planes=4) with its front replaced by ResNet(4, SimAMBasicBlock, [1, 1, 2, 1]) and an LDA of
  256 -> 128, every parameter and buffer a closed form of its name and flat index i (s = sin(0.37 i + 0.11 len(name) +
  0.0013 (i^2 mod 9973))): batch-norm weights 1 + 0.2 s, running variances 1 + 0.5 |s|, running means and biases 0.1 s,
  convolution and linear weights 0.5 s / sqrt(fan-in); on 0.5 s of a closed-form 24 kHz signal.
- released (when its directory is given): Zyphra/Zonos-v0.1-speaker-embedding's ResNet293_SimAM_ASP_base.pt and LDA-128
  (the LDA rounded to float32, as its nn.Linear holds it) on 3 s of speech (MaskGCT's prompt.wav from Amphion, mono, at its 24 kHz... resampled with librosa's soxr_hq).

Records the 16 kHz waveform's length, the log mel features, the 256-dimensional embedding and its LDA projection.

Usage: python zonos_speaker_reference.py [SPEAKER_DIR] (with ~/source/repos/.venv-zonos and refsrc/Zonos). Writes
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/zonos_speaker_reference.json.
"""
import json
import math
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.join(HERE, "..", "..", "..", "refsrc")
sys.path.insert(0, os.path.join(REFSRC, "Zonos"))
torch.set_default_dtype(torch.float64)

import zonos.speaker_cloning as sc  # noqa: E402


def closed_form(name, shape):
    n = math.prod(shape) if shape else 1
    i = torch.arange(n, dtype=torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((torch.arange(n) ** 2) % 9973).double())
    if name.endswith("running_var"):
        return 1 + 0.5 * s.abs()
    if name.endswith("running_mean") or name.endswith(".bias"):
        return 0.1 * s
    if len(shape) == 1:
        return 1 + 0.2 * s
    fan_in = math.prod(shape[1:])
    return 0.5 * s / math.sqrt(fan_in)


def signal(samples, rate):
    t = torch.arange(samples, dtype=torch.float64)
    return 0.3 * torch.sin(2 * math.pi * 220 * t / rate) + 0.2 * torch.sin(0.37 * t + 0.0013 * ((t ** 2) % 9973)) \
        + 0.1 * torch.sin(2 * math.pi * 1375 * t / rate)


def run(embedder, lda, wav, rate):
    with torch.no_grad():
        prepared = embedder.prepare_input(wav.view(1, -1), rate)
        features = embedder.model.featCal(prepared)
        embedding = embedder.model(prepared)
        projected = lda(embedding)
    return {"samples16k": prepared.shape[-1], "features": features[0].reshape(-1).tolist(), "frames": features.shape[-1],
            "embedding": embedding[0].tolist(), "lda": projected[0].tolist()}


def tiny():
    model = sc.ResNet293_based(in_planes=4)
    model.front = sc.ResNet(4, sc.SimAMBasicBlock, [1, 1, 2, 1])
    model.featCal = sc.logFbankCal()
    lda = torch.nn.Linear(256, 128)
    model, lda = model.double().eval(), lda.double().eval()
    shapes = {}
    with torch.no_grad():
        for prefix, module in (("", model), ("lda.", lda)):
            for name, tensor in module.state_dict().items():
                if not tensor.is_floating_point() or name.startswith("featCal."):
                    continue
                shapes[prefix + name] = list(tensor.shape)
                tensor.copy_(closed_form(prefix + name, list(tensor.shape)).reshape(tensor.shape))
    embedder = types.SimpleNamespace(model=model, prepare_input=sc.SpeakerEmbedding.prepare_input.__get__(_Resampling()))
    wav = signal(12000, 24000)
    return {"shapes": shapes, "rate": 24000, "audio": wav.tolist(), **run(embedder, lda, wav, 24000)}


class _Resampling:
    """The parts of SpeakerEmbedding that prepare_input uses."""

    def __init__(self):
        self.device = "cpu"
        self._resamplers = {}

    def _get_resampler(self, rate):
        import torchaudio
        if rate not in self._resamplers:
            self._resamplers[rate] = torchaudio.transforms.Resample(rate, 16_000, dtype=torch.float64)
        return self._resamplers[rate]


def released(directory):
    import librosa
    import soundfile
    model = sc.ResNet293_based()
    model.load_state_dict(torch.load(os.path.join(directory, "ResNet293_SimAM_ASP_base.pt"), weights_only=True, map_location="cpu"))
    model.featCal = sc.logFbankCal()
    lda_sd = torch.load(os.path.join(directory, "ResNet293_SimAM_ASP_base_LDA-128.pt"), weights_only=True)
    lda = torch.nn.Linear(256, 128)
    # The release builds the LDA as a float32 nn.Linear: its float64 file is rounded to float32 on load.
    lda.load_state_dict({k: v.float().double() for k, v in lda_sd.items()})
    model, lda = model.double().eval(), lda.double().eval()
    data, rate = soundfile.read(os.path.join(REFSRC, "Amphion", "models", "tts", "maskgct", "wav", "prompt.wav"), always_2d=True)
    mono = librosa.to_mono(data.T.astype("float32"))[: rate * 3]
    audio = librosa.resample(mono, orig_sr=rate, target_sr=24000, res_type="soxr_hq").astype("float64")
    embedder = types.SimpleNamespace(model=model, prepare_input=sc.SpeakerEmbedding.prepare_input.__get__(_Resampling()))
    wav = torch.from_numpy(audio)
    return {"rate": 24000, "audio": audio.tolist(), **run(embedder, lda, wav, 24000)}


def main():
    result = {"tiny": tiny()}
    if len(sys.argv) > 1:
        result["released"] = released(sys.argv[1])
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "zonos_speaker_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    for k, v in result.items():
        print(k, v["samples16k"], v["frames"], v["lda"][:3])


if __name__ == "__main__":
    main()
