"""Reference fixture for Mimi (Defossez et al. 2024), the codec CSM speaks through: moshi 0.2.2's MimiModel built as
moshi.models.loaders.get_mimi builds it (the released configuration), run without streaming as Sesame's generator runs it,
in float64. The attention context is shortened from 250 to 12 frames so that two seconds of audio (50 frames at 25 Hz, 25 codes)
reach past the window.

Every parameter and buffer is set by a closed form of its name and flat index i, so the C# test rebuilds the full-size
codec without a stored checkpoint: s = sin(0.37 * i + 0.11 * len(name) + 0.0013 * (i * i mod 9973)); layer-norm weights are 1 + 0.2 s, cluster usage
1 + 0.5 s, codebook sums 0.5 s, layer scales 0.05 s, the codebooks' _initialized flags 1, and everything else (convolution
and linear weights, biases) s * sqrt(2 / fan-in) for weights and 0.05 s for biases. moshi's rotary embedding computes in
float32; it is replaced by the same rotation in float64, and attention by its masked softmax (torch 2.4's CPU kernel
returns NaN in float64 with a boolean mask).

Records the input audio, its codes (all 32 codebooks), the unquantized latent at 12.5 Hz, and the audio decoded from the
codes (MimiModel.decode, which keeps the causal padding's length), with each tensor's name and shape.

Run with ~/source/repos/.venv-csm (moshi 0.2.2, torch 2.4.0).
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mimi_reference.json.
"""
import json
import math
import os
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
# moshi.utils.quantize imports bitsandbytes on every linear map but uses it only for 8-bit weights, which Mimi has none of;
# the package does not install on Windows.
sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")

import moshi.modules.rope as rope  # noqa: E402
from moshi.models import loaders  # noqa: E402
from moshi.models.compression import MimiModel  # noqa: E402
from moshi.modules import SEANetDecoder, SEANetEncoder, transformer  # noqa: E402
from moshi.quantization import SplitResidualVectorQuantizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT = 12


def apply_rope64(q, k, offset, max_period=10_000, time_before_heads=False):
    B, H, T, D = q.shape
    ds = torch.arange(D // 2, dtype=torch.float64)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = (offset.double() + torch.arange(T, dtype=torch.float64)).view(1, -1, 1)
    q = q.view(B, H, T, D // 2, 2)
    k = k.view(B, H, T, D // 2, 2)
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([q[..., 0] * rotr - q[..., 1] * roti, q[..., 0] * roti + q[..., 1] * rotr], dim=-1)
    ko = torch.stack([k[..., 0] * rotr - k[..., 1] * roti, k[..., 0] * roti + k[..., 1] * rotr], dim=-1)
    return qo.view(B, H, T, D), ko.view(B, H, T, D)


def attention64(q, k, v, mask=None, dropout_p=0.0):
    # torch 2.4's CPU scaled_dot_product_attention returns NaN for float64 inputs with a boolean mask; this is its math.
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


rope.apply_rope = apply_rope64
transformer.F.scaled_dot_product_attention = attention64


def closed_form(name, shape):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    s = torch.sin(0.37 * k.to(torch.float64) + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(("norm1.weight", "norm2.weight")):
        return 1 + 0.2 * s
    if name.endswith("cluster_usage"):
        return 1 + 0.5 * s
    if name.endswith("embedding_sum"):
        return 0.5 * s
    if name.endswith("_initialized"):
        return torch.ones(count)
    if name.endswith(".scale"):
        return 0.05 * s
    if len(shape) >= 2:
        return s * math.sqrt(2 * shape[0] / count)
    return 0.05 * s


def build():
    kwargs = dict(loaders._transformer_kwargs, context=CONTEXT)
    encoder = SEANetEncoder(**loaders._seanet_kwargs)
    model = MimiModel(
        encoder, SEANetDecoder(**loaders._seanet_kwargs),
        SplitResidualVectorQuantizer(**loaders._quantizer_kwargs), channels=1, sample_rate=loaders.SAMPLE_RATE,
        frame_rate=loaders.FRAME_RATE, encoder_frame_rate=loaders.SAMPLE_RATE / encoder.hop_length, causal=True, resample_method="conv",
        encoder_transformer=transformer.ProjectedTransformer(**kwargs),
        decoder_transformer=transformer.ProjectedTransformer(**kwargs)).eval()
    model.set_num_codebooks(32)
    return model


def main():
    model = build()
    state = model.state_dict()
    names = {}
    with torch.no_grad():
        for name, value in state.items():
            if name.endswith("._embedding"):
                continue
            names[name] = list(value.shape)
            value.copy_(closed_form(name, list(value.shape)).reshape(value.shape))
        for module in model.modules():
            if hasattr(module, "_embedding"):
                module._embedding = None
        n = 48000
        i = torch.arange(n, dtype=torch.float64)
        generator = torch.Generator().manual_seed(0)
        envelope = 0.5 + 0.5 * torch.sin(2 * math.pi * 3 * i / 24000)
        audio = envelope * (0.3 * torch.sin(0.0137 * i * (1 + i / 9600)) + 0.2 * torch.randn(n, generator=generator))
        x = audio.view(1, 1, -1)
        latent = model._encode_to_unquantized_latent(x)[0]
        codes = model.encode(x)[0]
        decoded = model.decode(codes[None])[0, 0]
    result = {
        "context": CONTEXT, "shapes": names, "audio": audio.tolist(),
        "codes": codes.T.tolist(), "latent": latent.T.reshape(-1).tolist(), "latent_shape": [latent.shape[1], latent.shape[0]],
        "decoded": decoded.tolist(),
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "mimi_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(codes.shape, latent.shape, decoded.shape, float(decoded.abs().max()), float(latent.abs().max()))


if __name__ == "__main__":
    main()
