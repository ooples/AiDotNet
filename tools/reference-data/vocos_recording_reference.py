"""Execute pinned, unmodified Vocos dataset/features/backbone/head code with real torchaudio/SoX.

Run with --source-dir pointing to the pinned upstream checkout, or a cache of the
seven vocos-author-*.py files. Git blob IDs are checked before extracting the AST definitions;
unused EnCodec/Lightning imports are not needed for these mel-vocoder components.
The reference uses actual torchaudio WAV I/O and sox_effects, not a replacement
normalizer. Tiny float64 generator states exercise the released architecture and
its derivatives; they are numerical fixtures, not published trained checkpoints.
"""
import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset

REVISION = "eb39abfc42c1dee4854d9b10d44dd7d4fd3b0e53"
SOURCES = {
    "dataset": ("25b3bc169531d8984056d4074dc4519d79343d24", ["DataConfig", "VocosDataset"]),
    "modules": ("af1d6db16e2f10cc9af7bcc64434e2c983d756b7", ["safe_log", "ConvNeXtBlock", "AdaLayerNorm"]),
    "feature_extractors": ("799f1b48978cdb723ac810bc5e53620ca859d480", ["FeatureExtractor", "MelSpectrogramFeatures"]),
    "models": ("09ed55ddb111f47bb71f220834083a7b0643a4df", ["Backbone", "VocosBackbone"]),
    "spectral_ops": ("e96b4541cd47fbd5e969b43a30fd26b39b34bab6", ["ISTFT"]),
    "heads": ("6e587d627abb3fd762b0c3539368fb603f3c3cbe", ["FourierHead", "ISTFTHead"]),
    "loss": ("029f6ac0407d767b94abafa6fdf013755daabf44", ["FeatureMatchingLoss"]),
}


def load_author(source_dir):
    for name, (expected, names) in SOURCES.items():
        path = source_dir / f"vocos-author-{name}.py"
        if not path.exists():
            path = source_dir / "vocos" / f"{name}.py"
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            raise ValueError(f"{path}: Git blob {actual}, expected {expected}")
        module = ast.parse(content.decode(), filename=str(path))
        definitions = [n for n in module.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        if len(definitions) != len(names):
            raise ValueError(f"Missing author definitions in {path}")
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), globals())


def values(tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    load_author(args.source_dir)
    asset_dir = args.output.parent / "vocos-reference-wavs"
    asset_dir.mkdir(exist_ok=True)
    features = MelSpectrogramFeatures(sample_rate=4000, n_fft=64, hop_length=16, n_mels=8, padding="center")
    # Match the native numeric type when checking spectral arithmetic. Retain the
    # release's float32 trace too: log magnifies float32 FFT roundoff near silence.
    torch.set_default_dtype(torch.float64)
    features64 = MelSpectrogramFeatures(sample_rate=4000, n_fft=64, hop_length=16, n_mels=8, padding="center")
    torch.set_default_dtype(torch.float32)
    cases = []
    specifications = [("short", 37, 128), ("equal", 128, 128), ("long", 2048, 128),
                      ("silence", 37, 128), ("almost", 127, 128), ("odd", 257, 128), ("larger_crop", 2048, 256),
                      ("negative_peak", 257, 128)]
    for index, (name, length, segment) in enumerate(specifications):
        pcm = (9000 * np.sin(np.arange(length) * .17) + 3000 * np.cos(np.arange(length) * .31)).astype(np.int16)
        if name == "silence":
            pcm[:] = 0
        else:
            pcm[0] = 30000  # whole-recording peak, outside the selected long crop
        if name == "negative_peak": pcm[0] = -32768
        path = asset_dir / f"{name}.wav"
        torchaudio.save(str(path), torch.from_numpy(pcm.astype(np.float32) / 32768).unsqueeze(0), 4000,
                       encoding="PCM_S", bits_per_sample=16)
        filelist = asset_dir / f"{name}.txt"
        filelist.write_text(str(path) + "\n")
        cfg = DataConfig(str(filelist), 4000, segment, 1, 0)
        seed = 20261110 + index
        draws = np.random.RandomState(seed)
        gain = float(draws.uniform(-1, -6))
        start = int(draws.randint(0, length - segment + 1)) if length >= segment else None
        np.random.seed(seed)
        waveform = VocosDataset(cfg, train=True)[0]
        mel = features(waveform.unsqueeze(0))
        mel64 = features64(waveform.double().unsqueeze(0))
        cases.append(dict(name=name, raw_pcm=pcm.tolist(), segment=segment, gain_draw=(-1 - gain) / 5,
                          gain_db=f"{gain:.2f}", start_sample=start, audio=values(waveform),
                          mel_shape=list(mel64.shape), mel=values(mel64), mel_float32=values(mel)))

    # Instantiate unmodified author modules directly in float64 for a complete
    # ConvNeXt -> complex spectrum -> centered ISTFT derivative comparison.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20261110)
    model = nn.ModuleDict({"backbone": VocosBackbone(input_channels=8, dim=8, intermediate_dim=16, num_layers=2),
                           "head": ISTFTHead(dim=8, n_fft=64, hop_length=16, padding="center")}).eval()
    mel = (.3 * torch.sin(torch.arange(72) * .19)).reshape(1, 8, 9).requires_grad_()
    hidden = model["backbone"](mel)
    waveform = model["head"](hidden)
    target = (.1 * torch.sin(torch.arange(waveform.numel()) * .13)).reshape_as(waveform)
    loss = torch.nn.functional.mse_loss(waveform, target)
    loss.backward()
    generator = dict(weights={name: values(p) for name, p in model.state_dict().items()},
                     shapes={name: list(p.shape) for name, p in model.state_dict().items()},
                     input_mel=values(mel), hidden=values(hidden), audio=values(waveform), target=values(target),
                     loss=loss.item(), mel_gradient=values(mel.grad),
                     parameter_gradients={name: None if p.grad is None else values(p.grad) for name, p in model.named_parameters()},
                     parameter_scalars=sum(p.numel() for p in model.parameters()))

    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    rates = []
    for _ in range(4):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    real_features = [torch.tensor([.2, -.1, .4]), torch.tensor([.7, -.3, .1, .8]), torch.tensor([-.2, .6])]
    fake_features = [torch.tensor([-.1, .3, .2], requires_grad=True), torch.tensor([.1, -.5, .4, .2], requires_grad=True),
                     torch.tensor([.3, -.1], requires_grad=True)]
    author_sum = FeatureMatchingLoss()([real_features], [fake_features])
    # The released trainer sums over layers. Section 3.3 explicitly divides by L;
    # record both conventions so the native implementation can follow the paper.
    paper_mean = author_sum / len(fake_features)
    paper_mean.backward()
    matching = dict(real=[values(x) for x in real_features], fake=[values(x) for x in fake_features],
                    author_sum=author_sum.item(), paper_mean=paper_mean.item(), gradients=[values(x.grad) for x in fake_features])

    data = dict(revision=REVISION, source_blobs={name: sha for name, (sha, _) in SOURCES.items()},
                torch_version=torch.__version__, torchaudio_version=torchaudio.__version__,
                description=__doc__, cases=cases, generator=generator, cosine_rates=rates, feature_matching=matching,
                input_mel_bank=values(features.mel_spec.mel_scale.fb), input_mel_bank_float64=values(features64.mel_spec.mel_scale.fb))
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(cases)} actual SoX/dataset cases; generator {generator['parameter_scalars']} scalars, loss {loss.item():.15g}")


if __name__ == "__main__":
    main()
