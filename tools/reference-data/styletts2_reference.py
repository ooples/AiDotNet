"""StyleTTS 2 reference: yl4579/StyleTTS2 (models.py, Modules/, Utils/PLBERT), the code the paper's (Li et al. 2023)
gaps follow.

Builds two tiny randomly initialized StyleTTS 2 models with the reference's build_model (pass a checkout of
yl4579/StyleTTS2 with --styletts2; it is not vendored here): the multispeaker LibriTTS configuration (HiFi-GAN decoder,
StyleTransformer1d) and the single-speaker LJSpeech one (iSTFTNet decoder, Transformer1d). Stores their weights in
float32, runs them in float64 in evaluation mode and records:
- the text encoder's features, PL-BERT's hidden states and their projection, the reference style (both style encoders);
- the style denoiser's output at one noise level (with classifier-free guidance), the sampled style;
- inference as the reference notebooks run it: the duration logits, rounded durations, pitch, energy and the waveform;
- the SLM discriminator's scores.
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts2_reference.json. Default: verify; --write: regenerate.

Changes to the reference, none of which alter its arithmetic: the decoder's hard-coded widths (1024, 64, 512) replaced by
the tiny model's; the harmonic source's random initial phases and noise set to zero (torch.rand / torch.randn_like inside
the decoder), and the sampler's per-step noise taken from recorded draws, so both sides see the same values; the Karras
schedule computed in float64 (the reference computes it in float32 whatever the default dtype). The text aligner and
pitch extractor are StyleTTS's tiny ones (tools/reference-data/styletts_reference.py).

Needs torch, transformers, torchaudio, munch, pyyaml, einops, einops-exts, scipy and monotonic_align.
"""
import argparse
import base64
import contextlib
import inspect
import os
import sys
import textwrap

import torch
from munch import munchify
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import nn

import _fixture
import styletts_reference as v1

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts2_reference.json"
BOTTLENECK, RESIDUAL = 24, 6
COMMON = dict(dim_in=4, hidden_dim=16, max_conv_dim=16, n_layer=2, n_mels=80, n_token=178, max_dur=6, style_dim=8, dropout=0.2,
              slm=dict(hidden=8, nlayers=3, initial_channel=4),
              diffusion=dict(embedding_mask_proba=0.1, transformer=dict(num_layers=2, num_heads=2, head_features=4, multiplier=2),
                             dist=dict(sigma_data=0.2, estimate_sigma_data=True, mean=-3.0, std=1.0)))
CONFIGS = {
    "libritts": dict(COMMON, multispeaker=True, decoder=dict(
        type="hifigan", resblock_kernel_sizes=[3, 7, 11], upsample_rates=[2, 2], upsample_initial_channel=16,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_kernel_sizes=[4, 4])),
    "ljspeech": dict(COMMON, multispeaker=False, decoder=dict(
        type="istftnet", resblock_kernel_sizes=[3, 7, 11], upsample_rates=[2, 2], upsample_initial_channel=16,
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], upsample_kernel_sizes=[4, 4], gen_istft_n_fft=8,
        gen_istft_hop_size=2)),
}
BERT = dict(vocab_size=178, embedding_size=8, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32)
TOKENS = [0, 51, 157, 102, 16, 83, 56, 135, 47, 102, 4]
REFERENCE_FRAMES = 96
STEPS = 5
SIGMA = 0.7
EDM_SIGMA = 0.35
GUIDANCE = 1.5
# The multi-period and multi-resolution discriminators are CosyVoice HiFT's (full width: 30M parameters); their port is
# tested with CosyVoice, so their weights stay out of this fixture.
WAVE_DISCRIMINATORS = ("mpd.", "msd.")
INFERENCE = dict(alpha=0.3, beta=0.7, trim=50, final_extra=0)


def load(root):
    models, ASRCNN, TinyJDC = v1.load(root)
    from Utils.PLBERT.util import CustomAlbert
    from transformers import AlbertConfig
    import Modules.istftnet as istftnet
    import Modules.hifigan as hifigan
    import Modules.discriminators as discriminators
    from Modules.diffusion import sampler

    for module in (istftnet, hifigan):
        source = textwrap.dedent(inspect.getsource(module.Decoder))
        source = (source.replace("1024", str(BOTTLENECK)).replace("+ 64", f"+ {RESIDUAL}")
                  .replace("nn.Conv1d(512, 64", f"nn.Conv1d({COMMON['hidden_dim']}, {RESIDUAL}")
                  .replace(f", 512, style_dim, upsample=True", ", 16, style_dim, upsample=True"))
        namespace = {}
        exec(source, module.__dict__, namespace)
        module.Decoder = namespace["Decoder"]

    # The spectrogram discriminators' Hann window in the working precision (it is made in the default dtype at
    # construction, and moved with y.get_device(), which fails on the CPU).
    source = textwrap.dedent(inspect.getsource(discriminators.SpecDiscriminator.forward))
    old = "self.window.to(y.get_device())"
    assert old in source
    namespace = {}
    exec(source.replace(old, "torch.hann_window(self.win_length, dtype=y.dtype)"), discriminators.__dict__, namespace)
    discriminators.SpecDiscriminator.forward = namespace["forward"]

    def schedule(self, num_steps, device):
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float64)
        sigmas = (self.sigma_max ** rho_inv + (steps / (num_steps - 1)) * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho
        return torch.nn.functional.pad(sigmas, pad=(0, 1), value=0.0)

    sampler.KarrasSchedule.forward = schedule
    return models, ASRCNN, TinyJDC, CustomAlbert, AlbertConfig, sampler


def build(loaded, name, state=None):
    models, ASRCNN, TinyJDC, CustomAlbert, AlbertConfig, _ = loaded
    torch.manual_seed(2294 + len(name))
    bert = CustomAlbert(AlbertConfig(**BERT))
    nets = models.build_model(munchify(CONFIGS[name]), ASRCNN(**v1.ASR), TinyJDC(v1.JDC_WIDTHS, v1.JDC_LSTM), bert)
    module = nn.ModuleDict(dict(nets))
    if state is None:
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                    m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
            # Snake's alphas away from 1, so loading them is tested.
            for key, p in module["decoder"].named_parameters():
                if "alpha" in key:
                    p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            # Pitch that exercises the harmonic source. HiFi-GAN: tens of hertz, voiced and unvoiced. iSTFTNet: kilohertz,
            # so every bin of the 8-point source spectrum carries signal; where a bin is (nearly) empty, its angle is
            # whatever the FFT's rounding leaves, which no other implementation reproduces.
            projection = module["predictor"].F0_proj
            if CONFIGS[name]["decoder"]["type"] == "hifigan":
                projection.weight.mul_(60.0)
                projection.bias.fill_(15.0)
            else:
                projection.weight.mul_(600.0)
                projection.bias.fill_(3000.0)
            # Durations of one to five frames.
            module["predictor"].duration_proj.linear_layer.weight.mul_(4.0)
            module.train()
            for _ in range(3):
                module["style_encoder"](torch.randn(1, 1, 80, REFERENCE_FRAMES, generator=g))
                module["predictor_encoder"](torch.randn(1, 1, 80, REFERENCE_FRAMES, generator=g))
    else:
        module.load_state_dict(state, strict=True)
    # iSTFTNet's Hann window in the working precision (the reference casts it to float32).
    from scipy.signal import get_window
    for m in module.modules():
        if type(m).__name__ == "TorchSTFT":
            m.window = torch.from_numpy(get_window("hann", m.win_length, fftbins=True))
    return module.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    reference = torch.randn(1, 80, REFERENCE_FRAMES, generator=g, dtype=torch.float32) * 0.6 - 1.5
    noise = torch.randn(1, 1, 2 * COMMON["style_dim"], generator=g, dtype=torch.float32)
    steps = [torch.randn(1, 1, 2 * COMMON["style_dim"], generator=g, dtype=torch.float32) for _ in range(STEPS - 1)]
    probe = torch.randn(1, 1, 2 * COMMON["style_dim"], generator=g, dtype=torch.float32)
    hidden = torch.randn(1, 8 * 3, 12, generator=g, dtype=torch.float32)
    edm_target = torch.randn(1, 1, 2 * COMMON["style_dim"], generator=g, dtype=torch.float32) * 0.3
    edm_noise = torch.randn(1, 1, 2 * COMMON["style_dim"], generator=g, dtype=torch.float32)
    return reference, noise, steps, probe, hidden, edm_target, edm_noise


@contextlib.contextmanager
def silent_source():
    """The harmonic source without its random initial phases and noise."""
    rand, randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **{k: v for k, v in kwargs.items() if k != "generator"})
    torch.randn_like = lambda x, **kwargs: torch.zeros_like(x)
    try:
        yield
    finally:
        torch.rand, torch.randn_like = rand, randn_like


@contextlib.contextmanager
def recorded_noise(draws):
    """The sampler's torch.randn_like returning the recorded draws in order."""
    queue = list(draws)
    randn_like = torch.randn_like
    torch.randn_like = lambda x, **kwargs: queue.pop(0).to(x.dtype).reshape(x.shape)
    try:
        yield
    finally:
        torch.randn_like = randn_like
        assert not queue, "unused sampler draws"


def flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def run(loaded, module, name):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return run_double(loaded, module, name)
    finally:
        torch.set_default_dtype(previous)


def run_double(loaded, module, name):
    sampler = loaded[5]
    config = CONFIGS[name]
    multispeaker = config["multispeaker"]
    reference, noise, steps, probe, hidden, edm_target, edm_noise = inputs()
    tokens = torch.tensor([TOKENS])
    lengths = torch.tensor([len(TOKENS)])
    text_mask = torch.zeros(1, len(TOKENS), dtype=torch.bool)
    s_dim = config["style_dim"]
    out = {}
    with torch.no_grad():
        t_en = module["text_encoder"](tokens, lengths, text_mask)
        out["text"] = flat(t_en)
        bert_dur = module["bert"](tokens, attention_mask=(~text_mask).int())
        out["bert"] = flat(bert_dur)
        d_en = module["bert_encoder"](bert_dur).transpose(-1, -2)
        out["bert_encoder"] = flat(d_en)
        ref_s = None
        if multispeaker:
            mel = reference.unsqueeze(1).double()
            ref_s = torch.cat([module["style_encoder"](mel), module["predictor_encoder"](mel)], dim=1)
            out["reference_style"] = flat(ref_s)
        diffusion = module["diffusion"].diffusion
        # KDiffusion.forward (the EDM loss) at a fixed noise level and noise.
        diffusion.sigma_distribution = lambda num_samples, device=None: torch.full((num_samples,), EDM_SIGMA)
        out["edm_loss"] = [float(diffusion(edm_target.double(), noise=edm_noise.double(), embedding=bert_dur, features=ref_s))]
        out["denoised"] = flat(diffusion.denoise_fn(probe.double(), sigma=SIGMA, embedding=bert_dur, features=ref_s,
                                                    embedding_scale=GUIDANCE))
        draw = sampler.DiffusionSampler(diffusion, sampler=sampler.ADPM2Sampler(),
                                        sigma_schedule=sampler.KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
        with recorded_noise(steps):
            s_pred = draw(noise=noise.double(), embedding=bert_dur, embedding_scale=1, features=ref_s, num_steps=STEPS).squeeze(1)
        out["sampled_style"] = flat(s_pred)

        s, ref = s_pred[:, s_dim:], s_pred[:, :s_dim]
        if multispeaker:
            ref = INFERENCE["alpha"] * ref + (1 - INFERENCE["alpha"]) * ref_s[:, :s_dim]
            s = INFERENCE["beta"] * s + (1 - INFERENCE["beta"]) * ref_s[:, s_dim:]
        d = module["predictor"].text_encoder(d_en, s, lengths, text_mask)
        x, _ = module["predictor"].lstm(d)
        logits = module["predictor"].duration_proj(x)
        out["duration_logits"] = flat(logits)
        duration = torch.sigmoid(logits).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_dur[-1] += INFERENCE["final_extra"]
        out["rounded_durations"] = [int(v) for v in pred_dur.tolist()]
        aln = torch.zeros(len(TOKENS), int(pred_dur.sum().item()))
        frame = 0
        for i in range(aln.size(0)):
            aln[i, frame:frame + int(pred_dur[i].item())] = 1
            frame += int(pred_dur[i].item())
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        asr = t_en @ aln.unsqueeze(0)
        if config["decoder"]["type"] == "hifigan":
            for value in ("en", "asr"):
                v = en if value == "en" else asr
                shifted = torch.zeros_like(v)
                shifted[:, :, 0] = v[:, :, 0]
                shifted[:, :, 1:] = v[:, :, 0:-1]
                if value == "en":
                    en = shifted
                else:
                    asr = shifted
        f0, n = module["predictor"].F0Ntrain(en, s)
        out["f0"] = flat(f0)
        out["energy"] = flat(n)
        with silent_source():
            audio = module["decoder"](asr, f0, n, ref.squeeze().unsqueeze(0))
        trim = INFERENCE["trim"] if multispeaker else 0
        audio = audio.squeeze()
        out["wave"] = flat(audio[: audio.shape[-1] - trim] if trim else audio)

        out["wd"] = flat(module["wd"](hidden.double()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styletts2", required=True, help="a checkout of yl4579/StyleTTS2")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    os.chdir(args.styletts2)
    loaded = load(args.styletts2)
    reference, noise, steps, probe, hidden, edm_target, edm_noise = inputs()
    shared = {"tokens": TOKENS, "reference": flat(reference[0]), "reference_frames": REFERENCE_FRAMES, "noise": flat(noise),
              "step_noise": [flat(s) for s in steps], "probe": flat(probe), "sigma": SIGMA, "guidance": GUIDANCE, "steps": STEPS,
              "bert": BERT, "asr": v1.ASR, "jdc_widths": v1.JDC_WIDTHS, "jdc_lstm": v1.JDC_LSTM, "bottleneck": BOTTLENECK,
              "residual": RESIDUAL, "inference": INFERENCE, "hidden": flat(hidden),
              "edm_sigma": EDM_SIGMA, "edm_target": flat(edm_target), "edm_noise": flat(edm_noise)}
    if args.write:
        data = {"shared": shared}
        for name in CONFIGS:
            full = {k: v.float().clone().contiguous() for k, v in build(loaded, name).state_dict().items()}
            state = {k: v for k, v in full.items() if not k.endswith("position_ids") and not k.startswith(WAVE_DISCRIMINATORS)}
            module = build(loaded, name, full).double()
            data[name] = {"config": CONFIGS[name], "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                          **run(loaded, module, name)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    check = _fixture.Comparison("StyleTTS 2")
    for name in CONFIGS:
        committed = data[name]
        state = load_safetensors(base64.b64decode(committed["safetensors_base64"]))
        state.update({k: v.float() for k, v in build(loaded, name).state_dict().items()
                      if k.startswith(WAVE_DISCRIMINATORS) or k.endswith("position_ids")})
        again = run(loaded, build(loaded, name, state).double(), name)
        for key, value in again.items():
            if key == "rounded_durations":
                check.exact(f"{name} {key}", committed[key], value)
            else:
                check.floats(f"{name} {key}", committed[key], value, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
