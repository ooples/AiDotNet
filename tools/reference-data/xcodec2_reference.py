"""Reference fixture for X-codec2 (Ye et al. 2025, the Llasa paper's speech tokenizer), with its semantic encoder
Wav2Vec2-BERT 2.0 (transformers' Wav2Vec2BertModel) and SeamlessM4T's feature extractor, in float64.

The networks are the official classes of the xcodec2 0.1.5 package (the one Llasa's model card imports), built small:
CodecEncoder_Transformer(ngf=2) keeps the 320-sample hop, a 17-layer, 32-wide Wav2Vec2BertModel (relative_key
positions, as w2v-bert-2.0) supplies hidden_states[16], SemanticEncoder(32, 32, 32), fc_prior 64 -> 64, ResidualFSQ over
64 with levels [4] * 8, fc_post_a 64 -> 32 and CodecDecoderVocos(hidden_dim=32, depth=2, heads=4, pos_meb_dim=8).
XCodec2Model.encode_code and decode_code run unchanged on an instance holding these modules.

Every parameter is a closed form of its name, shape and flat index i, with
s = sin(0.37 * i + 0.11 * len(name) + 0.0013 * (i * i mod 9973)) (the quadratic term keeps the matrices from being
nearly rank two, which collapsed every frame onto one code):
a norm's weight (a name ending in norm.weight, norm1.weight, norm2.weight or layer_norm.weight) is 1 + 0.2 s, a
weight-norm gain (weight_g) 0.5 + 0.2 s, a SnakeBeta log-scale alpha or beta (act.alpha, act.bias) 0.1 s, the FSQ input
projection's weight 8 s / sqrt(fan_in) (so its outputs span several quantization levels), any other
tensor of two or more dimensions 0.6 s / sqrt(fan_in) (fan_in = numel / shape[0]), and any other vector 0.05 s.

The decoder's rotary embedding is applied over the head axis (torchtune reads axis 1 of [batch, heads, time, dim] as
the sequence), so queries and keys of one head get the same rotation and the attention scores are unchanged. The
fixture decodes twice, with torchtune's class and with the rotation removed, and checks the two agree (torchtune
computes in float32, so they differ by its rounding).

Records the ISTFT's window buffer (computed in float32, as stored in the checkpoint), and, for two signals (3840 samples, a whole number of hops, which encode_code still pads by a full hop, and 4000):
the extractor's input_features, hidden_states[16], the codes and the decoded audio; and each parameter's name and shape.
The extractor casts the waveform and its spectrogram to float32, which moves quiet bins by up to 0.01 after
normalization, so it also records the extractor's features computed in float64 (its _extract_fbank_features with the
spectrogram in float64, then its normalization, truncation and stacking) and the codes encode_code makes from them;
hidden_states[16] is that of the float64 features.

Usage: python xcodec2_reference.py. Needs ../refsrc/xcodec2-pip/xcodec2-0.1.5 (pip download xcodec2==0.1.5), the
torchtune stub in ../refsrc/stubs (torchtune 0.3.1's RotaryPositionalEmbeddings, verbatim), vector-quantize-pytorch
1.17.8, einx, einops and transformers. Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xcodec2_reference.json.
"""
import functools
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.join(HERE, "..", "..", "..", "refsrc")
sys.path.insert(0, os.path.join(REFSRC, "stubs"))
sys.path.insert(0, os.path.join(REFSRC, "xcodec2-pip", "xcodec2-0.1.5"))

import transformers.models.seamless_m4t.feature_extraction_seamless_m4t as seamless_extraction  # noqa: E402
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel  # noqa: E402
from xcodec2.modeling_xcodec2 import XCodec2Model  # noqa: E402
from xcodec2.vq.codec_decoder_vocos import CodecDecoderVocos  # noqa: E402
from xcodec2.vq.codec_encoder import CodecEncoder_Transformer  # noqa: E402
from xcodec2.vq.module import SemanticEncoder  # noqa: E402
import xcodec2.vq.bs_roformer5 as roformer  # noqa: E402

SEMANTIC = dict(hidden_size=32, num_hidden_layers=17, num_attention_heads=4, intermediate_size=64,
                feature_projection_input_dim=160, conv_depthwise_kernel_size=31, position_embeddings_type="relative_key",
                left_max_position_embeddings=64, right_max_position_embeddings=8, hidden_act="swish", layer_norm_eps=1e-5,
                add_adapter=False, apply_spec_augment=False, mask_time_prob=0.0, layerdrop=0.0, hidden_dropout=0.0,
                attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, conformer_conv_dropout=0.0)
CONFIG = {"encoder_dim": 2, "encoder_rates": [2, 2, 4, 4, 5], "acoustic_dim": 32, "semantic_dim": 32, "semantic_layer": 16,
          "fsq_levels": [4] * 8, "decoder_dim": 32, "decoder_depth": 2, "decoder_heads": 4, "hop": 320, "semantic": SEMANTIC}


def closed_form(name, shape):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    i = k.to(torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(("norm.weight", "norm1.weight", "norm2.weight")):
        return 1 + 0.2 * s
    if name.endswith("weight_g") or name.endswith("original0"):
        return 0.5 + 0.2 * s
    if name.endswith(("act.alpha", "act.bias")):
        return 0.1 * s
    if name.endswith("project_in.weight"):
        return 8 * s / math.sqrt(count / shape[0])
    if len(shape) >= 2:
        return 0.6 * s / math.sqrt(count / shape[0])
    return 0.05 * s


def build():
    model = XCodec2Model.__new__(XCodec2Model)
    nn.Module.__init__(model)
    config = Wav2Vec2BertConfig(**SEMANTIC)
    config._attn_implementation = "eager"
    model.semantic_model = Wav2Vec2BertModel(config).eval()
    model.SemanticEncoder_module = SemanticEncoder(32, 32, 32)
    model.CodecEnc = CodecEncoder_Transformer(ngf=CONFIG["encoder_dim"], hidden_dim=CONFIG["acoustic_dim"])
    model.generator = CodecDecoderVocos(hidden_dim=32, depth=2, heads=4, pos_meb_dim=8, hop_length=320, vq_dim=64)
    model.fc_prior = nn.Linear(64, 64)
    model.fc_post_a = nn.Linear(64, 32)
    model.feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, padding_value=1, stride=2,
                                                          sampling_rate=16000)
    model = model.double().eval()
    shapes = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            shapes[name] = list(parameter.shape)
            parameter.copy_(closed_form(name, list(parameter.shape)).reshape(parameter.shape))
    return model, shapes


def features64(extractor, waveform):
    # SeamlessM4TFeatureExtractor.__call__'s steps for one waveform, with no float32 casts.
    spectrogram = seamless_extraction.spectrogram
    seamless_extraction.spectrogram = functools.partial(spectrogram, dtype=np.float64)
    try:
        x = extractor._extract_fbank_features(np.asarray(waveform, dtype=np.float64))
    finally:
        seamless_extraction.spectrogram = spectrogram
    x = (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
    frames = x.shape[0] - x.shape[0] % extractor.stride
    return torch.from_numpy(x[:frames].reshape(frames // extractor.stride, -1)).unsqueeze(0)


class NoRotation(nn.Module):
    def forward(self, x, *, input_pos=None):
        return x


def main():
    torch.manual_seed(0)
    model, shapes = build()
    cases = []
    for n in (3840, 4000):
        t = torch.arange(n, dtype=torch.float64)
        audio = (0.4 * torch.sin(0.031 * t) + 0.2 * torch.sin(0.17 * t + 0.3) + 0.05 * torch.cos(0.9 * t)).reshape(1, n)
        captured = {}
        real_extractor = model.feature_extractor

        use64 = {"on": False}

        def extractor(*args, **kwargs):
            out = real_extractor(*args, **kwargs)
            captured["features"] = out.input_features.clone()
            captured["features64"] = features64(real_extractor, args[0])
            out["input_features"] = captured["features64"] if use64["on"] else out.input_features.double()
            return out

        real_semantic = model.semantic_model.forward

        def semantic(features, **kwargs):
            out = real_semantic(features, output_hidden_states=True, **kwargs)
            captured["hidden16"] = out.hidden_states[16].clone()
            return out

        fsq = model.generator.quantizer.layers[0]
        hook = fsq.register_forward_pre_hook(lambda module, inputs: captured.__setitem__("bounded", fsq.bound(inputs[0])))
        model.feature_extractor = extractor
        model.semantic_model.forward = semantic
        try:
            codes = model.encode_code(audio)
            margin = ((captured["bounded"] - captured["bounded"].floor()) - 0.5).abs().min().item()
            use64["on"] = True
            codes64 = model.encode_code(audio)
            margin = min(margin, ((captured["bounded"] - captured["bounded"].floor()) - 0.5).abs().min().item())
        finally:
            model.feature_extractor = real_extractor
            model.semantic_model.forward = real_semantic
            hook.remove()
        # No value the quantizer rounds may sit near a rounding boundary, or a port's last-digit differences could flip a code.
        assert margin > 1e-6, margin
        decoded_rotary = model.decode_code(codes)
        rotaries = []
        for block in model.generator.backbone.transformers:
            rotaries.append(block.att.rotary_embed)
            block.att.rotary_embed = NoRotation()
        decoded = model.decode_code(codes)
        for block, rotary in zip(model.generator.backbone.transformers, rotaries):
            block.att.rotary_embed = rotary
        gap = (decoded - decoded_rotary).abs().max().item()
        assert gap < 1e-5 * max(1.0, decoded.abs().max().item()), gap
        cases.append({
            "audio": audio.flatten().tolist(),
            "features": captured["features"][0].tolist(),
            "features64": captured["features64"][0].tolist(),
            "codes64": codes64.flatten().tolist(),
            "hidden16": captured["hidden16"][0].tolist(),
            "codes": codes.flatten().tolist(),
            "decoded": decoded.flatten().tolist(),
        })
        print(n, "frames", codes.shape[-1], "distinct codes", len(set(codes.flatten().tolist())), "codes differ in float64",
              int((codes != codes64).sum()), "decoded", decoded.shape[-1],
              "rotary gap", gap, "rounding margin", margin)
    # The ISTFT window is a float32 buffer the checkpoint stores; the port loads it.
    buffers = {"generator.head.istft.window": model.generator.head.istft.window.tolist()}
    result = {"config": CONFIG, "cases": cases, "shapes": shapes, "buffers": buffers}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "xcodec2_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")


if __name__ == "__main__":
    main()
