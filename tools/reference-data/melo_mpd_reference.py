"""Reference fixture for MeloTTS's (and VITS's) multi-period discriminator: models.MultiPeriodDiscriminator(False), the
scale discriminator DiscriminatorS (MelGAN's widths) followed by the period discriminators 2, 3, 5, 7 and 11, in float64.

Every parameter is weight(name, i) = 0.02 * sin(0.37 * i + 0.11 * len(name)) (i the flat index; the weight-norm
weight_g, weight_v and biases alike), so the C# test rebuilds the full-size discriminator without a stored checkpoint.
Records the scores for a real and a generated signal of 300 samples (not a multiple of every period), and
losses.discriminator_loss, generator_loss and feature_loss on them (the last covers every feature map).

Needs a clone of myshell-ai/MeloTTS at 2091453 next to this repository (../refsrc/MeloTTS). Run with torch.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_mpd_reference.json.
"""
import json
import os
import sys
from unittest import mock

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "MeloTTS"))
torch.set_default_dtype(torch.float64)

from melo import losses, models  # noqa: E402


def in_float64(fn):
    def run(*args, **kwargs):
        with mock.patch.object(torch.Tensor, "float", lambda t: t.double()):
            return fn(*args, **kwargs)
    return run


def main():
    net = models.MultiPeriodDiscriminator(False)
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            i = torch.arange(parameter.numel(), dtype=torch.float64)
            parameter.copy_((0.02 * torch.sin(0.37 * i + 0.11 * len(name))).reshape(parameter.shape))
    n = 300
    t = torch.arange(n, dtype=torch.float64)
    real = (0.4 * torch.sin(0.07 * t) + 0.1 * torch.cos(0.31 * t)).reshape(1, 1, n)
    fake = (0.3 * torch.sin(0.05 * t + 0.4) + 0.2 * torch.sin(0.23 * t)).reshape(1, 1, n)
    with torch.no_grad():
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = net(real, fake)
        d_loss, _, _ = in_float64(losses.discriminator_loss)(y_d_rs, y_d_gs)
        g_loss, _ = in_float64(losses.generator_loss)(y_d_gs)
        fm_loss = in_float64(losses.feature_loss)(fmap_rs, fmap_gs)
    result = {
        "real": real.flatten().tolist(), "fake": fake.flatten().tolist(),
        "real_scores": [s.flatten().tolist() for s in y_d_rs], "fake_scores": [s.flatten().tolist() for s in y_d_gs],
        "discriminator_loss": d_loss.item(), "generator_loss": g_loss.item(), "feature_loss": fm_loss.item(),
        "names": [name for name, _ in net.named_parameters()],
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "melo_mpd_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print("discriminators:", len(y_d_rs), "losses:", d_loss.item(), g_loss.item(), fm_loss.item())


if __name__ == "__main__":
    main()
