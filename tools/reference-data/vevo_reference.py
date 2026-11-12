"""Reference fixture for Vevo's networks (Zhang et al., ICLR 2025; Amphion models/vc at 26f6883), built small, in float64:
HuBERT (torchaudio's wav2vec2_model in HuBERT-Large's layout, read at a middle layer), the content-style tokenizer
(RepCodec over normalized features), the content tokenizer (VevoRepCodec over raw features, duplicates removed), the two
autoregressive transformers (phones to content-style tokens for Vevo-TTS; content tokens to content-style tokens for
Vevo-Voice and Vevo-Style), the flow-matching transformer (content-style tokens to mel, with its guided Euler solver and
its training loss) and the Vocos vocoder, each from the release's own classes, plus the mel features.

Weights are closed forms of each parameter's name, shape and flat index (maskgct_reference.py's), so the test rebuilds
them; the content tokenizer's EMA codebook (a buffer) is filled the same way. The precision patches are
maskgct_reference.py's: the rotary angles, the adaptive and plain RMSNorms and the attention softmax are kept in float64. Decoding is
greedy (the release samples), the solver's noise and the loss's draws are fixed, and everything is recorded in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vevo_reference.json.

Usage: python vevo_reference.py [--write], with transformers 4.52.1 (the flow model's OldLlamaAttention needs 4.46 or
later, past the requirements' 4.41.2 pin), torch, torchaudio, einops and librosa, and Amphion cloned at
../refsrc/Amphion-full.
"""
import argparse
import math
import os
import sys
import types

import torch
import torchaudio

import _fixture

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "Amphion-full"))
torch.set_default_dtype(torch.float64)
sys.modules["pyworld"] = types.ModuleType("pyworld")   # melspec.py imports it for pitch tools the mel never calls

import transformers.models.llama.modeling_llama as llama  # noqa: E402
import models.vc.flow_matching_transformer.llama_nar as llama_nar  # noqa: E402
from models.vc.autoregressive_transformer.ar_model import AutoregressiveTransformer  # noqa: E402
from models.vc.flow_matching_transformer.fmt_model import FlowMatchingTransformer  # noqa: E402
from models.codec.kmeans.repcodec_model import RepCodec  # noqa: E402
from models.codec.vevo.vevo_repcodec import VevoRepCodec  # noqa: E402
from models.codec.melvqgan.melspec import MelSpectrogram  # noqa: E402
from models.codec.amphion_codec.vocos import Vocos  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vevo_reference.json"

HUBERT = dict(conv=[(8, 10, 5), (8, 3, 2), (8, 2, 2)], dim=16, layers=3, heads=2, ff=32, pos_kernel=8, pos_groups=4, read_layer=2)
CODES, CODE_DIM = 16, 4                       # content-style tokens
CONTENT_CODES = 4                             # content tokens
TTS_VOCAB = 20
MEL = dict(n_fft=64, num_mels=8, sampling_rate=24000, hop_size=30, win_size=64, fmin=0, fmax=12000)
MEL_MEAN, MEL_VAR = -4.92, 8.14
SAMPLES_24K = 1200
PHONES = [3, 7, 11, 4, 9, 2]
STYLE_PHONES = [5, 1, 8]


def rotary64(self, x, position_ids):
    config = self.config
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].double()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def adaptive_norm64(self, hidden_states, cond_embedding):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    weight = self.to_weight(cond_embedding)
    if len(weight.shape) == 2:
        weight = weight.unsqueeze(1)
    return weight * hidden_states


def rms_norm64(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


llama.LlamaRotaryEmbedding.forward = rotary64
llama.LlamaRMSNorm.forward = rms_norm64
llama_nar.LlamaAdaptiveRMSNorm.forward = adaptive_norm64
_softmax = torch.nn.functional.softmax


def softmax64(x, dim=None, _stacklevel=3, dtype=None):
    # OldLlamaAttention takes its softmax in float32; in float64 the same softmax in the input's dtype.
    return _softmax(x, dim=dim) if x.dtype == torch.float64 else _softmax(x, dim=dim, dtype=dtype)


torch.nn.functional.softmax = softmax64


def closed_form(name, shape):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    i = k.to(torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(("to_weight.bias", "norm.weight")):
        return 1 + 0.2 * s
    if name.endswith("weight_g"):
        return 0.5 + 0.2 * s
    if name.endswith(".alpha"):
        return 1 + 0.5 * s
    if name.endswith("in_project.weight_v"):
        return 8 * s / math.sqrt(count / shape[0])
    if len(shape) >= 2:
        return 0.6 * s / math.sqrt(count / shape[0])
    return 0.05 * s


def fill(model, buffers=()):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel():
                parameter.copy_(closed_form(name, list(parameter.shape)).reshape(parameter.shape))
        for name, buffer in model.named_buffers():
            if any(name.endswith(b) for b in buffers):
                buffer.copy_(closed_form(name, list(buffer.shape)).reshape(buffer.shape))


class Fixed:
    """Replaces torch's samplers with fixed values while a call runs."""

    def __init__(self, randn=(), rand=(), randint=()):
        self.randn, self.rand, self.randint = list(randn), list(rand), list(randint)

    def __enter__(self):
        self.saved = torch.randn, torch.rand, torch.randint
        torch.randn = lambda *a, **k: self.randn.pop(0).clone()
        torch.rand = lambda *a, **k: self.rand.pop(0).clone()
        torch.randint = lambda *a, **k: self.randint.pop(0).clone()
        return self

    def __exit__(self, *exc):
        torch.randn, torch.rand, torch.randint = self.saved


def greedy(ar):
    # The release samples (do_sample=True); greedy decoding exercises the same loop deterministically.
    generate = ar.model.generate
    ar.model.generate = lambda *a, **k: generate(*a, **{**k, "do_sample": False})
    return ar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)

    h = HUBERT
    hubert = torchaudio.models.wav2vec2_model(
        extractor_mode="layer_norm", extractor_conv_layer_config=h["conv"], extractor_conv_bias=False, encoder_embed_dim=h["dim"],
        encoder_projection_dropout=0.0, encoder_pos_conv_kernel=h["pos_kernel"], encoder_pos_conv_groups=h["pos_groups"],
        encoder_num_layers=h["layers"], encoder_num_heads=h["heads"], encoder_attention_dropout=0.0,
        encoder_ff_interm_features=h["ff"], encoder_ff_interm_dropout=0.0, encoder_dropout=0.0, encoder_layer_norm_first=True,
        encoder_layer_drop=0.0, aux_num_out=None)
    fill(hubert)
    content_style = RepCodec(codebook_size=CODES, hidden_size=h["dim"], codebook_dim=CODE_DIM, vocos_dim=8, vocos_intermediate_dim=16,
                             vocos_num_layers=1)
    fill(content_style)
    content = VevoRepCodec(input_channels=h["dim"], output_channels=h["dim"], encode_channels=h["dim"], decode_channels=h["dim"],
                           code_dim=h["dim"], codebook_num=1, codebook_size=CONTENT_CODES)
    fill(content, buffers=("embed",))
    content.quantizer.initial()
    ar_tts = AutoregressiveTransformer(input_vocab_size=TTS_VOCAB, output_vocab_size=CODES, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=2, num_attention_heads=2)
    ar_vc = AutoregressiveTransformer(input_vocab_size=TTS_VOCAB + CONTENT_CODES, output_vocab_size=CODES, hidden_size=16,
                                      intermediate_size=24, num_hidden_layers=2, num_attention_heads=2)
    fmt = FlowMatchingTransformer(mel_dim=MEL["num_mels"], hidden_size=16, num_layers=2, num_heads=2, cfg_scale=0.2, cond_codebook_size=CODES,
                                  sigma=1e-5, time_scheduler="cos")
    vocos = Vocos(input_channels=MEL["num_mels"], dim=16, intermediate_dim=32, num_layers=2, n_fft=64, hop_size=30, padding="same")
    for m in (ar_tts, ar_vc, fmt, vocos):
        fill(m)
    for m in (hubert, content_style, content, ar_tts, ar_vc, fmt, vocos):
        m.eval()
    # The release stores librosa's filterbank as float32 (mel_basis .float()); converted, its values stay float32-rounded.
    mel_model = MelSpectrogram(**MEL).double()

    def tensor(key, values):
        return torch.tensor(stored[key]) if stored is not None else values
    k = torch.arange(SAMPLES_24K, dtype=torch.float64)
    wave = tensor("wave24k", (0.3 * torch.sin(0.05 * k) * torch.sin(0.0031 * k * k / SAMPLES_24K * 7) + 0.05 * torch.sin(1.3 * k)).float().double())
    timbre = tensor("timbre24k", (0.25 * torch.sin(0.08 * k + 1.0) * torch.cos(0.002 * k) + 0.04 * torch.cos(2.1 * k)).float().double())
    stat_mean = tensor("hubert_mean", (0.1 * torch.sin(torch.arange(h["dim"]) * 0.7)).float().double())
    stat_std = tensor("hubert_std", (1.0 + 0.3 * torch.cos(torch.arange(h["dim"]) * 0.9)).float().double())

    out, results = {}, {}
    with torch.no_grad():
        def mel(x):
            return ((mel_model(x.unsqueeze(0)).transpose(1, 2) - MEL_MEAN) / math.sqrt(MEL_VAR))[0]

        def to16k(x):
            return torchaudio.functional.resample(x.unsqueeze(0), 24000, 16000)[0]

        def hubert_feats(x16):
            feats, _ = hubert.extract_features(x16.unsqueeze(0), num_layers=h["read_layer"])
            return feats[-1]                                                         # [1, T, D]

        def content_style_codes(x16):
            codes, _ = content_style.quantize((hubert_feats(x16) - stat_mean) / stat_std)
            return codes                                                             # [1, T]

        def content_codes(x16):
            x = content.encoder(hubert_feats(x16).transpose(1, 2))
            z = content.projector(x)
            _, idx = content.quantizer.codebook.forward_index(z.transpose(2, 1))
            raw = idx[0][0]
            keep = torch.cat([torch.tensor([True]), raw[1:] != raw[:-1]])
            return raw, raw[keep]

        out["mel"] = mel(wave)
        wave16 = to16k(wave)
        timbre16 = to16k(timbre)
        out["wave16k"] = wave16
        out["hubert"] = hubert_feats(wave16)[0]
        cs = content_style_codes(wave16)
        results["content_style_codes"] = cs[0].tolist()
        raw, reduced = content_codes(wave16)
        results["content_codes"] = raw.tolist()
        results["content_codes_reduced"] = reduced.tolist()

        # Vevo-TTS's AR: the style reference's phones then the text's, prompted by the reference's content-style tokens.
        out_ids = torch.tensor([results["content_style_codes"][:10]])
        loss = ar_tts(torch.tensor([PHONES]), torch.ones(1, len(PHONES), dtype=torch.long), out_ids,
                      torch.ones_like(out_ids))
        out["ar_loss"] = loss.loss.reshape(1)
        out["ar_logits"] = loss.logits[0]
        prompt = cs[:, :6]
        tts_codes = greedy(ar_tts).generate(torch.tensor([STYLE_PHONES + PHONES]), prompt_output_ids=prompt,
                                            max_length=len(STYLE_PHONES + PHONES) + 2 + 1 + 6 + 14, min_new_tokens=5)
        results["ar_tts_codes"] = tts_codes[0].tolist()
        # Vevo-Voice's AR: the content tokens of [style reference, source], duplicates removed, shifted past the phones.
        _, voice_input = content_codes(torch.cat([timbre16, wave16]))
        vc_codes = greedy(ar_vc).generate((voice_input + TTS_VOCAB).unsqueeze(0), prompt_output_ids=content_style_codes(timbre16),
                                          max_length=voice_input.shape[0] + 2 + 1 + 39 + 10, min_new_tokens=3)
        results["voice_input"] = voice_input.tolist()
        results["ar_vc_codes"] = vc_codes[0].tolist()

        # The flow as Vevo-Timbre runs it (inference_fm): the timbre reference's tokens then the source's, prompted by the
        # reference's mel. (The tiny AR's greedy codes stray outside the content-style codes, which the flow cannot embed.)
        timbre_codes = content_style_codes(timbre16)
        cond_codes = torch.cat([timbre_codes, cs], dim=1)
        results["fm_codes"] = cond_codes[0].tolist()
        prompt_mel = mel(timbre).unsqueeze(0)
        target = cond_codes.shape[1] - prompt_mel.shape[1]
        steps = 4
        noise = tensor("fm_noise", torch.sin(torch.arange(target * MEL["num_mels"], dtype=torch.float64) * 0.77).float().double().reshape(1, target, MEL["num_mels"]))
        with Fixed(randn=[noise.reshape(1, target, MEL["num_mels"])]):
            solved = fmt.reverse_diffusion(cond=fmt.cond_emb(cond_codes), prompt=prompt_mel, n_timesteps=steps, cfg=1.0, rescale_cfg=0.75)
        out["fm_mel"] = solved[0]
        out["vocoder_wave"] = vocos(solved.transpose(1, 2))[0, 0]

        # The flow's training loss (fmt_trainer): x the recording's mel cut to its token count, t = 0.3 before the cos schedule,
        # a prompt of 7 frames, fixed noise.
        x = mel(wave).unsqueeze(0)[:, : cs.shape[1]]
        x_noise = tensor("loss_noise", torch.cos(torch.arange(x.numel(), dtype=torch.float64) * 0.41).float().double().reshape(x.shape))
        with Fixed(randn=[x_noise.reshape(x.shape)], rand=[torch.tensor([0.3]), torch.tensor([0.9])], randint=[torch.tensor([7])]):
            noise_used, x0, flow_pred, final_mask, _ = fmt(x, torch.ones(1, x.shape[1]), cond_code=cs)
        final_mask = final_mask.squeeze(-1)
        flow_gt = x0 - (1 - 1e-5) * noise_used
        diff = torch.nn.functional.l1_loss(flow_pred, flow_gt, reduction="none") * final_mask.unsqueeze(-1)
        out["fm_loss"] = (torch.mean(diff, dim=2).sum() / final_mask.sum()).reshape(1)

    shapes = {}
    for prefix, m, buffers in (("hubert.", hubert, ()), ("content_style.", content_style, ()), ("content.", content, ("embed",)),
                               ("ar_tts.", ar_tts, ()), ("ar_vc.", ar_vc, ()), ("fmt.", fmt, ()), ("vocos.", vocos, ())):
        for name, p in m.named_parameters():
            if p.numel():
                shapes[prefix + name] = list(p.shape)
        for name, b in m.named_buffers():
            if any(name.endswith(x) for x in buffers):
                shapes[prefix + name] = list(b.shape)
    if args.write:
        data = {"shapes": shapes, "wave24k": _fixture.flat(wave), "timbre24k": _fixture.flat(timbre), "hubert_mean": _fixture.flat(stat_mean),
                "hubert_std": _fixture.flat(stat_std), "fm_noise": _fixture.flat(noise), "loss_noise": _fixture.flat(x_noise), **results}
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: " + ", ".join(f"{k} {v}" for k, v in results.items()))
        return
    check = _fixture.Comparison("Vevo")
    check.exact("shapes", stored["shapes"], shapes)
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
