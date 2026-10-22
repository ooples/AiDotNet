"""Reference fixture for CosyVoice (Du et al. 2024): the official FunAudioLLM/CosyVoice modules in float64.

Runs the official TransformerLM (text encoder + language model) on one utterance at small sizes: the loss of eq. 8 and
one SGD step. Needs clones of FunAudioLLM/CosyVoice at COSYVOICE_COMMIT and of its Matcha-TTS submodule at
MATCHA_COMMIT next to this repository (../refsrc/CosyVoice, ../refsrc/Matcha-TTS), plus diffusers==0.25.0 and
conformer==0.3.2 installed without dependencies. Packages the modules import but these classes never use (onnxruntime,
whisper, ...) are stubbed.

Numerics: ESPnet's relative positional table is computed in float32 by the reference; here it is computed in float64,
as the C# model computes it, so the comparison measures the network rather than float32 sines.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_reference.json. Run with the reference venv.
"""
import importlib.machinery
import json
import math
import os
import sys
from unittest import mock

import huggingface_hub

huggingface_hub.cached_download = huggingface_hub.hf_hub_download  # diffusers 0.25 imports the removed name
for _name in ["onnxruntime", "gdown", "whisper", "tiktoken", "inflect", "pyworld", "wget", "hyperpyyaml", "tensorrt"]:
    try:
        __import__(_name)
    except Exception:
        _stub = mock.MagicMock()
        _stub.__spec__ = importlib.machinery.ModuleSpec(_name, None)
        sys.modules[_name] = _stub

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc"))
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
MATCHA_COMMIT = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
sys.path.insert(0, os.path.join(REFSRC, "Matcha-TTS"))
sys.path.insert(0, os.path.join(REFSRC, "CosyVoice"))

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from cosyvoice.transformer import embedding as cv_embedding  # noqa: E402
from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder  # noqa: E402
from cosyvoice.llm.llm import TransformerLM  # noqa: E402
from cosyvoice.utils.common import ras_sampling  # noqa: E402


def _extend_pe_float64(self, x):
    # EspnetRelPositionalEncoding.extend_pe with the table in float64 instead of float32.
    if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
        return
    pe_positive = torch.zeros(x.size(1), self.d_model)
    pe_negative = torch.zeros(x.size(1), self.d_model)
    position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
    pe_positive[:, 0::2] = torch.sin(position * div_term)
    pe_positive[:, 1::2] = torch.cos(position * div_term)
    pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
    pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
    pe_negative = pe_negative[1:].unsqueeze(0)
    self.pe = torch.cat([pe_positive, pe_negative], dim=1)


cv_embedding.EspnetRelPositionalEncoding.extend_pe = _extend_pe_float64


def flat(module):
    return {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}


def randomize(module, gen, scale):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * scale)


LM = dict(text_vocab=20, speech_tokens=12, text_in=8, text_out=16, width=16, heads=2, units=24, blocks=2, spk=6)


def language_model():
    c = LM
    text_encoder = ConformerEncoder(input_size=c["text_in"], output_size=c["text_out"], attention_heads=c["heads"],
                                    linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                                    positional_dropout_rate=0.0, attention_dropout_rate=0.0, normalize_before=True,
                                    input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                                    selfattention_layer_type="rel_selfattn", use_cnn_module=False,
                                    macaron_style=False, static_chunk_size=1)
    llm = TransformerEncoder(input_size=c["width"], output_size=c["width"], attention_heads=c["heads"],
                             linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                             positional_dropout_rate=0.0, attention_dropout_rate=0.0, input_layer="linear_legacy",
                             pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                             static_chunk_size=1)
    return TransformerLM(text_encoder_input_size=c["text_in"], llm_input_size=c["width"], llm_output_size=c["width"],
                         text_token_size=c["text_vocab"], speech_token_size=c["speech_tokens"], text_encoder=text_encoder,
                         llm=llm, sampling=ras_sampling, length_normalized_loss=True, lsm_weight=0.0,
                         spk_embed_dim=c["spk"])


def run_language_model():
    gen = torch.Generator().manual_seed(11)
    model = language_model()
    randomize(model, gen, 0.3)
    initial = flat(model)
    text = torch.randint(0, LM["text_vocab"], (1, 5), generator=gen)
    speech = torch.randint(0, LM["speech_tokens"], (1, 7), generator=gen)
    embedding = torch.randn(1, LM["spk"], generator=gen)
    batch = {"text_token": text, "text_token_len": torch.tensor([5]), "speech_token": speech,
             "speech_token_len": torch.tensor([7]), "embedding": embedding}
    model.train()
    loss = model(batch, torch.device("cpu"))["loss"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"config": LM, "initial": initial, "text": text.flatten().tolist(), "speech": speech.flatten().tolist(),
            "embedding": embedding.flatten().tolist(), "loss": loss.item(), "lr": 0.1, "final": flat(model)}


FLOW = dict(vocab=10, input=8, mel=80, spk=6, heads=2, units=12, blocks=2, channels=[16, 16], head_dim=4,
            est_heads=2, est_blocks=2, est_mid=2)


def _sinusoidal_float64(self, x, scale=1000):
    # matcha's SinusoidalPosEmb with its frequencies in float64 instead of float32.
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def flow_model():
    import types
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.flow.length_regulator import InterpolateRegulator
    from matcha.models.components import decoder as matcha_decoder
    matcha_decoder.SinusoidalPosEmb.forward = _sinusoidal_float64
    # mask_to_bias asserts a float32/16 dtype; the same bias in float64.
    from cosyvoice.flow import decoder as cv_decoder
    cv_decoder.mask_to_bias = lambda mask, dtype: (1.0 - mask.to(dtype)) * -1.0e+10
    c = FLOW
    encoder = ConformerEncoder(input_size=c["input"], output_size=c["input"], attention_heads=c["heads"],
                               linear_units=c["units"], num_blocks=c["blocks"], dropout_rate=0.0,
                               positional_dropout_rate=0.0, attention_dropout_rate=0.0, normalize_before=True,
                               input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                               selfattention_layer_type="rel_selfattn", use_cnn_module=False, macaron_style=False)
    regulator = InterpolateRegulator(channels=c["mel"], sampling_ratios=[1, 1, 1, 1])
    estimator = ConditionalDecoder(in_channels=4 * c["mel"], out_channels=c["mel"], channels=c["channels"], dropout=0.0,
                                   attention_head_dim=c["head_dim"], n_blocks=c["est_blocks"],
                                   num_mid_blocks=c["est_mid"], num_heads=c["est_heads"], act_fn="gelu")
    cfm_params = types.SimpleNamespace(sigma_min=1e-6, solver="euler", t_scheduler="cosine", training_cfg_rate=0.2,
                                       inference_cfg_rate=0.7, reg_loss_type="l1")
    decoder = ConditionalCFM(in_channels=3 * c["mel"], cfm_params=cfm_params, n_spks=1, spk_emb_dim=c["mel"],
                             estimator=estimator)
    return MaskedDiffWithXvec(input_size=c["input"], output_size=c["mel"], spk_embed_dim=c["spk"], output_type="mel",
                              vocab_size=c["vocab"], input_frame_rate=50, only_mask_loss=True, encoder=encoder,
                              length_regulator=regulator, decoder=decoder)


def run_flow():
    import copy
    import random as pyrandom
    from cosyvoice.flow import flow as cv_flow
    gen = torch.Generator().manual_seed(21)
    model = flow_model()
    randomize(model, gen, 0.2)
    initial = flat(model)
    c = FLOW
    tokens = torch.randint(0, c["vocab"], (1, 6), generator=gen)
    frames = 13
    feat = torch.randn(1, frames, c["mel"], generator=gen)
    embedding = torch.randn(1, c["spk"], generator=gen)
    prompt = 3
    t = torch.full((1, 1, 1), 0.37)
    z = torch.randn(1, c["mel"], frames, generator=gen)
    keep = torch.tensor([0.9])                          # > training_cfg_rate: the conditions are kept
    batch = {"speech_token": tokens, "speech_token_len": torch.tensor([6]), "speech_feat": feat,
             "speech_feat_len": torch.tensor([frames]), "embedding": embedding}

    captured = {}

    def capture(module, args, output):
        captured["pred"] = output

    def loss_and_step(net, absolute):
        rand_values = iter([t, keep])
        hook = net.decoder.estimator.register_forward_hook(capture)
        with mock.patch.object(cv_flow.random, "random", return_value=0.9),                 mock.patch.object(cv_flow.random, "randint", return_value=prompt),                 mock.patch("torch.rand", side_effect=lambda *a, **k: next(rand_values)),                 mock.patch("torch.randn_like", return_value=z.clone()):
            net.train()
            squared = net(batch, torch.device("cpu"))["loss"]
        hook.remove()
        if absolute:
            u = feat.transpose(1, 2) - (1 - 1e-6) * z
            loss = (captured["pred"] - u).abs().sum() / (frames * c["mel"])
        else:
            loss = squared
        optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), flat(net)

    squared_loss, squared_final = loss_and_step(copy.deepcopy(model), absolute=False)
    absolute_loss, absolute_final = loss_and_step(copy.deepcopy(model), absolute=True)

    def generate(token_count, prompt_tokens, prompt_frames, seed):
        g = torch.Generator().manual_seed(seed)
        text = torch.randint(0, c["vocab"], (1, token_count), generator=g)
        ptok = torch.randint(0, c["vocab"], (1, prompt_tokens), generator=g)
        pfeat = torch.randn(1, prompt_frames, c["mel"], generator=g)
        frames_out = prompt_frames + int(token_count / 50 * 22050 / 256)
        noise = torch.randn(1, c["mel"], frames_out, generator=g)
        net = copy.deepcopy(model)
        net.eval()
        with mock.patch("torch.randn_like", return_value=noise.clone()):
            mel, _ = net.inference(token=text, token_len=torch.tensor([token_count]), prompt_token=ptok,
                                   prompt_token_len=torch.tensor([prompt_tokens]), prompt_feat=pfeat,
                                   prompt_feat_len=torch.tensor([prompt_frames]), embedding=embedding,
                                   flow_cache=torch.zeros(1, c["mel"], 0, 2))
        return {"tokens": text.flatten().tolist(), "prompt_tokens": ptok.flatten().tolist(),
                "prompt_mel": pfeat.flatten().tolist(), "prompt_frames": prompt_frames,
                "noise": noise.flatten().tolist(), "mel": mel.transpose(1, 2).flatten().tolist(),
                "frames": mel.shape[2]}

    return {"config": FLOW, "initial": initial, "tokens": tokens.flatten().tolist(), "frames": frames,
            "mel": feat.flatten().tolist(), "embedding": embedding.flatten().tolist(), "prompt": prompt, "t": 0.37,
            "noise": z.flatten().tolist(), "lr": 0.05,
            "squared": {"loss": squared_loss, "final": squared_final},
            "absolute": {"loss": absolute_loss, "final": absolute_final},
            "generate_short": generate(6, 3, 5, 31), "generate_long": generate(45, 2, 4, 32)}


HIFT = dict(mel=8, base=16, harmonics=2, rates=[2, 2], kernels=[4, 4], n_fft=8, hop=2, res_kernels=[3, 5],
            res_dilations=[[1, 3], [1, 3]], src_kernels=[3, 5], src_dilations=[[1, 3], [1, 3]], f0_channels=8, frames=60)


def run_hift():
    import numpy as np
    from scipy.signal import get_window
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.hifigan.generator import HiFTGenerator
    c = HIFT
    gen = torch.Generator().manual_seed(41)
    model = HiFTGenerator(in_channels=c["mel"], base_channels=c["base"], nb_harmonics=c["harmonics"], sampling_rate=22050,
                          nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10, upsample_rates=c["rates"],
                          upsample_kernel_sizes=c["kernels"], istft_params={"n_fft": c["n_fft"], "hop_len": c["hop"]},
                          resblock_kernel_sizes=c["res_kernels"], resblock_dilation_sizes=c["res_dilations"],
                          source_resblock_kernel_sizes=c["src_kernels"], source_resblock_dilation_sizes=c["src_dilations"],
                          lrelu_slope=0.1, audio_limit=0.99,
                          f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=c["mel"], cond_channels=c["f0_channels"]))
    # SineGen's voicing mask is float32, which rounds its noise amplitudes (0.003, 0.1 / 3) to float32; float64 here.
    from cosyvoice.hifigan import generator as cv_gen
    cv_gen.SineGen._f02uv = lambda self, f0: (f0 > self.voiced_threshold).to(torch.float64)
    # The reference casts its Hann window to float32; the fixture keeps it in float64.
    model.stft_window = torch.from_numpy(get_window("hann", c["n_fft"], fftbins=True))
    randomize(model, gen, 0.3)
    with torch.no_grad():
        model.f0_predictor.classifier.bias.add_(10.235)     # an F0 near the 10 Hz voicing threshold: voiced and unvoiced frames
    initial = flat(model)
    mel = torch.randn(1, c["mel"], c["frames"], generator=gen)
    samples = c["frames"] * c["rates"][0] * c["rates"][1] * c["hop"]
    h = c["harmonics"] + 1
    phases = (torch.rand(1, h, 1, generator=gen) * 2 - 1) * math.pi
    noise = torch.randn(1, h, samples, generator=gen)   # SineGen draws it as [B, harmonics, samples]
    weights = torch.randn(samples, generator=gen)
    pitch = torch.rand(1, c["frames"], generator=gen) * 80
    from cosyvoice.hifigan import generator as cv_generator

    def run(net):
        f0 = net.f0_predictor(mel)
        s = net.f0_upsamp(f0[:, None]).transpose(1, 2)
        with mock.patch.object(cv_generator.Uniform, "sample", return_value=phases.clone()),                 mock.patch("torch.randn_like", side_effect=[noise.clone(), torch.zeros(1, samples, 1)]):
            source, _, _ = net.m_source(s)
        audio = net.decode(x=mel, s=source.transpose(1, 2))
        return f0, source, audio

    import copy as _copy
    run_hift.model = _copy.deepcopy(model)
    run_hift.inputs = (mel, pitch, phases, noise)
    f0, source, audio = run(model)
    loss = (audio.squeeze() * weights).sum() + torch.nn.functional.l1_loss(f0, pitch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"config": HIFT, "initial": initial, "mel": mel.flatten().tolist(), "phases": phases.flatten().tolist(),
            "noise": noise[0].transpose(0, 1).flatten().tolist(), "weights": weights.tolist(), "pitch": pitch.flatten().tolist(),
            "f0": f0.flatten().tolist(), "source": source.flatten().tolist(), "audio": audio.flatten().tolist(),
            "loss": loss.item(), "lr": 0.01, "final": flat(model)}


GAN = dict(fft=[16, 32, 8], hop=[4, 8, 2], win=[12, 24, 6], mel_fft=32, mel_hop=8, mel_win=32, mel_bins=8,
           period_channels=[4, 6, 8, 8, 8])


def run_hift_gan():
    import copy
    import functools
    import librosa
    import numpy as np
    from cosyvoice.hifigan.discriminator import MultipleDiscriminator, MultiResSpecDiscriminator
    from cosyvoice.hifigan.hifigan import HiFiGan
    from cosyvoice.hifigan import generator as cv_generator
    from matcha.hifigan.models import MultiPeriodDiscriminator
    from matcha.utils import audio as matcha_audio
    g = GAN
    hift = run_hift.model                                   # the HiFT fixture's generator, before its SGD step
    gen = torch.Generator().manual_seed(51)
    from matcha.hifigan import models as matcha_models
    widths = {1: 1, 32: 4, 128: 6, 512: 8, 1024: 8}         # DiscriminatorP's 32/128/512/1024 channels, narrowed

    def narrow_conv(cin, cout, *args, **kwargs):
        return torch.nn.Conv2d(widths[cin], widths[cout], *args, **kwargs)

    def make_discriminator():
        with mock.patch.object(matcha_models, "Conv2d", narrow_conv):
            mpd = MultiPeriodDiscriminator()
        return MultipleDiscriminator(mpd=mpd, mrd=MultiResSpecDiscriminator(fft_sizes=g["fft"], hop_sizes=g["hop"],
                                                                            win_lengths=g["win"]))

    discriminator = make_discriminator()
    randomize(discriminator, gen, 0.1)
    # matcha caches a float32 mel basis and window; the fixture's are float64.
    key = f"None_cpu"
    matcha_audio.mel_basis[key] = torch.from_numpy(librosa.filters.mel(sr=22050, n_fft=g["mel_fft"], n_mels=g["mel_bins"], fmin=0, fmax=None, dtype=np.float64))
    matcha_audio.hann_window["cpu"] = torch.hann_window(g["mel_win"])
    transform = functools.partial(matcha_audio.mel_spectrogram, n_fft=g["mel_fft"], num_mels=g["mel_bins"], sampling_rate=22050,
                                  hop_size=g["mel_hop"], win_size=g["mel_win"], fmin=0, fmax=None, center=False)
    c = HIFT
    samples = c["frames"] * c["rates"][0] * c["rates"][1] * c["hop"]
    real = torch.randn(1, samples, generator=gen) * 0.3
    mel, pitch, phases, noise = run_hift.inputs
    initial_d = flat(discriminator)

    def step(turn):
        g_net = copy.deepcopy(hift)
        d_net = make_discriminator()                      # matcha's weight_norm modules cannot be deep-copied
        d_net.load_state_dict(discriminator.state_dict())
        model = HiFiGan(generator=g_net, discriminator=d_net, mel_spec_transform=[transform])
        batch = {"speech": real, "speech_feat": mel.transpose(1, 2), "pitch_feat": pitch, "turn": turn}
        with mock.patch.object(cv_generator.Uniform, "sample", return_value=phases.clone()),                 mock.patch("torch.randn_like", side_effect=[noise.clone(), torch.zeros(1, samples, 1)]):
            out = model(batch, torch.device("cpu"))
        params = list((g_net if turn == "generator" else d_net).parameters())
        optimizer = torch.optim.SGD(params, lr=0.01)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        return {k: float(v) for k, v in out.items()}, flat(g_net if turn == "generator" else d_net)

    gen_losses, gen_final = step("generator")
    disc_losses, disc_final = step("discriminator")
    return {"config": GAN, "real": real.flatten().tolist(), "discriminator": initial_d, "lr": 0.01,
            "generator_losses": gen_losses, "generator_final": gen_final,
            "discriminator_losses": disc_losses, "discriminator_final": disc_final}


def main():
    out = {"cosyvoice_commit": COSYVOICE_COMMIT, "matcha_commit": MATCHA_COMMIT, "language_model": run_language_model(),
           "flow": run_flow(), "hift": run_hift(), "hift_gan": run_hift_gan()}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "cosyvoice_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), "lm loss", out["language_model"]["loss"], "flow", out["flow"]["squared"]["loss"], out["flow"]["absolute"]["loss"])


if __name__ == "__main__":
    main()
