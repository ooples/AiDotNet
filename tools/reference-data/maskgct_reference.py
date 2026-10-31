"""Reference fixture for MaskGCT's networks (Wang et al. 2024; Amphion models/tts/maskgct at 26f6883), built small, in
float64: the T2S model (MaskGCT_T2S over DiffLlamaPrefix), the two S2A models (MaskGCT_S2A over DiffLlama: one predicting
only the first acoustic layer, one predicting the rest), the semantic codec (RepCodec) and the acoustic codec
(CodecEncoder and CodecDecoder with Vocos), each from the release's own classes.

Every parameter is a closed form of its name, shape and flat index i, with
s = sin(0.37 * i + 0.11 * len(name) + 0.0013 * (i * i mod 9973)): a layer norm's weight (a name ending in norm.weight) and
an adaptive norm's bias (to_weight.bias) are 1 + 0.2 s,
a weight-norm gain (weight_g) 0.5 + 0.2 s, a Snake alpha 1 + 0.5 s, a factorized quantizer's input projection direction
(in_project.weight_v) 8 s / sqrt(fan_in) (so its outputs spread over the codebook), any other tensor of two or more
dimensions 0.6 s / sqrt(fan_in) (fan_in = numel / shape[0]), and any other vector 0.05 s.

transformers 4.41's LlamaRotaryEmbedding computes its angles in float32 whatever the model's dtype, and Amphion's
LlamaAdaptiveRMSNorm casts to float32 for its variance, and the eager attention takes its softmax in float32; the first
two are replaced by the same formulas in float64 and the attention by transformers' SDPA attention (the same
computation in the model's dtype), so the comparison isolates the networks. The training forwards' random draws (t, the prompt
drop, the prompt length, the mask, the S2A layer) are fixed by replacing torch's samplers while they run, and decoding's
Gumbel draws are taken from a seeded generator and recorded in order (the uniforms, before the Gumbel transform).

Usage: python maskgct_reference.py, with transformers 4.41.2 (the release's pin), torch, einops and librosa, and Amphion
cloned next to this repository (../refsrc/Amphion). Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_reference.json.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "Amphion"))
torch.set_default_dtype(torch.float64)

import transformers.models.llama.modeling_llama as llama  # noqa: E402
import models.tts.maskgct.llama_nar as llama_nar  # noqa: E402
import models.tts.maskgct.maskgct_s2a as s2a_module  # noqa: E402
import models.tts.maskgct.maskgct_t2s as t2s_module  # noqa: E402
from models.codec.amphion_codec.codec import CodecDecoder, CodecEncoder  # noqa: E402
from models.codec.kmeans.repcodec_model import RepCodec  # noqa: E402
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A  # noqa: E402
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S  # noqa: E402


def rotary64(self, x, position_ids):
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float64) / self.dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].double()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


llama.LlamaRotaryEmbedding.forward = rotary64


def adaptive_norm64(self, hidden_states, cond_embedding):
    # LlamaAdaptiveRMSNorm.forward without its cast of the variance to float32.
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    weight = self.to_weight(cond_embedding)
    if len(weight.shape) == 2:
        weight = weight.unsqueeze(1)
    return weight * hidden_states


llama_nar.LlamaAdaptiveRMSNorm.forward = adaptive_norm64

CONFIG = {"hidden": 32, "layers": 2, "heads": 4, "semantic_codes": 256, "acoustic_codes": 128, "quantizers": 3,
          "semantic_codec": {"hidden": 32, "dim": 16, "intermediate": 32, "layers": 2, "codes": 64, "code_dim": 4},
          "acoustic_codec": {"encoder_dim": 4, "strides": [3, 4, 5, 8], "latent": 16, "code_dim": 4, "vocos_dim": 16,
                             "vocos_intermediate": 32, "vocos_layers": 2, "fft": 1920, "hop": 480}}


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


def fill(model):
    # transformers 4.41's eager attention takes its softmax in float32; its SDPA attention is the same computation in the
    # model's dtype.
    for module in model.modules():
        if type(module) is llama.LlamaAttention:
            module.__class__ = llama.LlamaSdpaAttention
    shapes = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel() == 0:
                continue
            shapes[name] = list(parameter.shape)
            parameter.copy_(closed_form(name, list(parameter.shape)).reshape(parameter.shape))
    return shapes


class Fixed:
    """Replaces torch's samplers with fixed values while a training forward runs."""

    def __init__(self, rand, randint=None, bernoulli=None, multinomial=None):
        self.rand, self.randint, self.bernoulli, self.multinomial = list(rand), randint, bernoulli, multinomial

    def __enter__(self):
        self.saved = torch.rand, torch.randint, torch.bernoulli, torch.multinomial
        torch.rand = lambda *a, **k: torch.tensor([self.rand.pop(0)] * (a[0] if isinstance(a[0], int) else a[0][0]))
        torch.randint = lambda *a, **k: torch.tensor(self.randint)
        torch.bernoulli = lambda p: torch.tensor(self.bernoulli, dtype=p.dtype).reshape(p.shape)
        torch.multinomial = lambda w, n: torch.tensor([self.multinomial])
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randint, torch.bernoulli, torch.multinomial = self.saved


def recording_gumbel(module, generator, draws):
    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
        draws.append(noise.flatten().tolist())
        return -module.log(-module.log(noise))
    module.gumbel_noise = gumbel_noise


def main():
    c = CONFIG
    result = {"config": c}

    # ---- T2S
    t2s = MaskGCT_T2S(hidden_size=c["hidden"], num_layers=c["layers"], num_heads=c["heads"], cfg_scale=0.15,
                      cond_codebook_size=c["semantic_codes"], cond_dim=c["hidden"]).eval()
    result["t2s_shapes"] = fill(t2s)
    phones = [5, 17, 3, 900, 42, 7]
    x0 = [3, 61, 212, 12, 140, 7, 0, 233, 25, 9]
    bern = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    with torch.no_grad(), Fixed(rand=[0.37, 0.5], randint=[3], bernoulli=bern):
        logits, final_mask, _, prompt_len, _ = t2s(torch.tensor([x0]), torch.ones(1, len(x0)), torch.tensor([phones]),
                                                  torch.ones(1, len(phones)))
    result["t2s_training"] = {"phones": phones, "x0": x0, "t": 0.37, "prompt": int(prompt_len[0]),
                              "mask": final_mask.flatten().tolist(), "logits": logits[0].tolist()}
    draws = []
    recording_gumbel(t2s_module, torch.Generator().manual_seed(1), draws)
    prompt = [3, 61, 212, 40]
    with torch.no_grad():
        generated = t2s.reverse_diffusion(torch.tensor([prompt]), 7, torch.tensor([phones]), temp=0.9, filter_thres=0.98,
                                          n_timesteps=4, cfg=2.5, rescale_cfg=0.75)
    result["t2s_generation"] = {"phones": phones, "prompt": prompt, "length": 7, "steps": 4, "draws": draws,
                                "tokens": generated.flatten().tolist()}

    # ---- S2A (first layer, and the rest)
    s2a_shapes = {}
    models = {}
    for key, quantizers, first in [("s2a_1layer", 1, True), ("s2a_full", c["quantizers"], False)]:
        model = MaskGCT_S2A(num_quantizer=quantizers, hidden_size=c["hidden"], num_layers=c["layers"], num_heads=c["heads"],
                            codebook_size=c["acoustic_codes"], cond_codebook_size=c["semantic_codes"], cond_dim=c["hidden"],
                            predict_layer_1=first).eval()
        s2a_shapes[key] = fill(model)
        models[key] = model
    result["s2a_shapes"] = s2a_shapes
    semantic = [3, 61, 212, 12, 140, 7, 0, 233, 25, 9, 14, 2]
    codes = [[(5 * t + 3 * q + 1) % c["acoustic_codes"] for q in range(c["quantizers"])] for t in range(len(semantic))]
    bern = [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    with torch.no_grad(), Fixed(rand=[0.61, 0.5], randint=[3], bernoulli=bern, multinomial=2):
        model = models["s2a_full"]
        cond = model.cond_emb(torch.tensor([semantic]))
        logits, layer, final_mask, _, prompt_len, _ = model.compute_loss(torch.tensor([codes]), torch.ones(1, len(semantic)), cond)
    result["s2a_training"] = {"semantic": semantic, "codes": codes, "t": 0.61, "layer": int(layer), "prompt": int(prompt_len[0]),
                              "mask": final_mask.flatten().tolist(), "logits": logits[0].tolist()}
    draws = []
    recording_gumbel(s2a_module, torch.Generator().manual_seed(2), draws)
    prompt_codes = torch.tensor([codes[:5]])
    with torch.no_grad():
        first = models["s2a_1layer"].reverse_diffusion(cond=models["s2a_1layer"].cond_emb(torch.tensor([semantic])), prompt=prompt_codes,
                                                       temp=1.5, filter_thres=0.98, n_timesteps=[3], cfg=2.5, rescale_cfg=0.75)
        full = models["s2a_full"].reverse_diffusion(cond=models["s2a_full"].cond_emb(torch.tensor([semantic])), prompt=prompt_codes,
                                                    temp=1.5, filter_thres=0.98, n_timesteps=[3, 2, 1], cfg=2.5, rescale_cfg=0.75,
                                                    gt_code=first)
    result["s2a_generation"] = {"semantic": semantic, "prompt": codes[:5], "steps_first": [3], "steps_full": [3, 2, 1],
                                "draws": draws, "first": first[0].tolist(), "full": full[0].tolist()}

    # ---- semantic codec
    sc = c["semantic_codec"]
    codec = RepCodec(codebook_size=sc["codes"], hidden_size=sc["hidden"], codebook_dim=sc["code_dim"], vocos_dim=sc["dim"],
                     vocos_intermediate_dim=sc["intermediate"], vocos_num_layers=sc["layers"]).eval()
    result["semantic_codec_shapes"] = fill(codec)
    frames = 9
    tt = torch.arange(frames * sc["hidden"], dtype=torch.float64).reshape(frames, sc["hidden"])
    feat = torch.sin(0.13 * tt) + 0.5 * torch.cos(0.031 * tt * tt / 50)
    mean = 0.1 * torch.sin(torch.arange(sc["hidden"], dtype=torch.float64))
    var = 1.0 + 0.5 * torch.cos(torch.arange(sc["hidden"], dtype=torch.float64)) ** 2
    with torch.no_grad():
        semantic_codes, _ = codec.quantize(((feat - mean) / torch.sqrt(var)).unsqueeze(0))
    result["semantic_codec"] = {"features": feat.tolist(), "mean": mean.tolist(), "var": var.tolist(),
                                "codes": semantic_codes.flatten().tolist()}

    # ---- acoustic codec
    ac = c["acoustic_codec"]
    encoder = CodecEncoder(d_model=ac["encoder_dim"], up_ratios=ac["strides"], out_channels=ac["latent"], use_tanh=False).eval()
    decoder = CodecDecoder(in_channels=ac["latent"], num_quantizers=c["quantizers"], codebook_size=c["acoustic_codes"],
                           codebook_dim=ac["code_dim"], quantizer_type="fvq", use_l2_normlize=True, use_vocos=True,
                           vocos_dim=ac["vocos_dim"], vocos_intermediate_dim=ac["vocos_intermediate"],
                           vocos_num_layers=ac["vocos_layers"], n_fft=ac["fft"], hop_size=ac["hop"], padding="same").eval()
    result["acoustic_encoder_shapes"] = fill(encoder)
    result["acoustic_decoder_shapes"] = fill(decoder)
    n = 4800
    t = torch.arange(n, dtype=torch.float64)
    audio = 0.5 * torch.sin(0.02 * t + 3 * torch.sin(0.003 * t)) * (1 + torch.sin(0.002 * t))
    with torch.no_grad():
        _, vq, _, _, _ = decoder.quantizer(encoder(audio.reshape(1, 1, n)))
        acoustic_codes = vq.permute(1, 2, 0)[0]
        decoded = decoder(decoder.vq2emb(vq, n_quantizers=c["quantizers"]))
    result["acoustic_codec"] = {"audio": audio.tolist(), "codes": acoustic_codes.tolist(), "decoded": decoded.flatten().tolist(),
                                "window": decoder.model.head.istft.window.tolist()}
    print("t2s tokens", result["t2s_generation"]["tokens"], "s2a full", result["s2a_generation"]["full"][:3],
          "semantic codes", result["semantic_codec"]["codes"], "acoustic codes", acoustic_codes.tolist()[:3],
          "distinct acoustic", len({tuple(r) for r in acoustic_codes.tolist()}))
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "maskgct_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")


if __name__ == "__main__":
    main()
