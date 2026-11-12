"""MegaTTS 3's WaveVAE through the release's own code (bytedance/MegaTTS3, tts/modules/wavvae: WavVAE_V3, its SEANet
encoder with LSTM, the diagonal Gaussian, the latent Upsample and the HiFi-GAN-style Generator), on a tiny model.

WavVAE_V3.__init__ hardcodes its sizes (encoder downsampling 6·5·4·4·2 to 512 channels, a 32-wide latent, a 320-channel
latent upsampler by 4, the Generator at ngf 128 with ratios 5·4·4·3); the test builds the same modules in the same order
smaller (encoder 4·6 to 16 channels, an 8-wide posterior for a 4-wide latent, a 20-channel upsampler by 4, ngf 2 with
ratios 3·2: 24 samples a latent frame both ways), a test-size patch. Random weights (weight-norm gains and directions,
LSTM, biases), float64. Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_wavvae_reference.json the
posterior mean and log-variance of 96 samples, the latent sampled with a given draw, and the decoded waveform.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-kani environment.
"""
import argparse
import os
import sys
import types

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_wavvae_reference.json"
SAMPLES = 96


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    from torch import nn
    from tts.modules.wavvae.decoder import wavvae_v3
    from tts.modules.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
    from tts.modules.wavvae.decoder.hifigan_modules import Generator, Upsample
    from tts.modules.wavvae.encoder.common_modules.seanet import SEANetEncoder

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_rate = 25
            self.encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2, dimension=16,
                                         channels=1, n_filters=4, ratios=[4, 6], activation='ELU', kernel_size=7, residual_kernel_size=3,
                                         last_kernel_size=7, dilation_base=2, true_skip=False, compress=2)

        def forward(self, audio):
            return self.encoder(audio.unsqueeze(1))

    class TinyWavVAE(wavvae_v3.WavVAE_V3):
        def __init__(self):
            nn.Module.__init__(self)
            self.encoder = TinyEncoder()
            self.proj_to_z = nn.Linear(16, 8)
            self.proj_to_decoder = nn.Linear(4, 20)
            self.latent_upsampler = Upsample(20, 4)
            args = types.SimpleNamespace(frame_shift=24, use_tanh=True)
            self.decoder = Generator(input_size_=10, ngf=2, n_residual_layers=4, num_band=1, args=args, ratios=[3, 2])

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(131)
    model = TinyWavVAE()
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("weight_g") or name.endswith("original0"):
                p.copy_((0.5 + torch.rand_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        state32 = {k: v.float().clone() for k, v in model.state_dict().items()}
    model = model.double().eval()
    g = torch.Generator().manual_seed(137)
    audio = (0.3 * torch.randn(1, SAMPLES, generator=g)).float().double() if args.write else torch.tensor([stored["audio"]], dtype=torch.float64)
    with torch.no_grad():
        posterior = model.encode(audio)
        eps = torch.randn(posterior.mean.shape, generator=g).float().double() if args.write else \
            torch.tensor(stored["eps"], dtype=torch.float64).reshape(posterior.mean.shape)
        latent = (posterior.mean + posterior.std * eps).permute(0, 2, 1)
        wav = model.decode(latent)
    torch.set_default_dtype(torch.float32)
    data = {"audio": audio[0].tolist(), "mean": posterior.mean.reshape(-1).tolist(), "logvar": posterior.logvar.reshape(-1).tolist(),
            "eps": eps.reshape(-1).tolist(), "latent": latent.reshape(-1).tolist(), "wav": wav.reshape(-1).tolist()}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "latent", tuple(latent.shape), "wav", tuple(wav.shape), "keys", len(state32))
        return
    check = _fixture.Comparison("MegaTTS 3 WaveVAE")
    for key in ("mean", "logvar", "latent", "wav"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
