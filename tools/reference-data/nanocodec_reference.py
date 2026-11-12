"""NVIDIA NeMo's NanoCodec modules (NeMo 2.4.0: HiFiGANEncoder, GroupFiniteScalarQuantizer, CausalHiFiGANDecoder) and
AudioCodecModel's encode and decode steps, on a tiny random codec.

The encoder (widths 4-16, downsampling 2 then 3, replicate padding, leaky ReLU) encodes 50 samples zero-padded to whole
6-sample frames; two FSQ groups of levels (9, 8, 8, 7) quantize; the causal decoder (widths 20, 10, 5: HalfSnake over an
odd width, grouped transposed convolutions, clamped output) decodes the codes. Records in
tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/nanocodec_reference.json the encoding, the codes and the decoded audio.
NeMo's module file is loaded from its source with stubs for the framework pieces it imports (typing decorators,
logging), which change no computation. AudioCodecModel's pad_audio, encode and decode are restated below from
nemo/collections/tts/models/audio_codec.py. Pass --nemo (the nemo_toolkit-2.4.0 source tree). Default: verify; --write:
regenerate. Needs the .venv-kani environment.
"""
import argparse
import importlib.util
import logging
import os
import sys
import types

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Audio/Codecs/ReferenceData/nanocodec_reference.json"
ENCODER = dict(encoded_dim=8, down_sample_rates=[2, 3], base_channels=4, activation="lrelu", pad_mode="replicate")
DECODER = dict(input_dim=8, up_sample_rates=[3, 2], base_channels=20, activation="half_snake", output_activation="clamp",
               pad_mode="zeros", n_groups_equal_to_out_channels=True)
QUANTIZER = dict(num_groups=2, num_levels_per_group=[9, 8, 8, 7])
SAMPLES = 50


def load_nemo(root):
    """NeMo's audio_codec_modules.py with its framework imports stubbed."""
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

    def typecheck(*a, **k):
        return lambda f: f

    module("nemo")
    module("nemo.collections")
    module("nemo.collections.asr")
    module("nemo.collections.asr.modules", AudioToMelSpectrogramPreprocessor=_Any)
    module("nemo.core")
    module("nemo.core.classes")
    module("nemo.core.classes.common", typecheck=typecheck)
    module("nemo.core.classes.module", NeuralModule=torch.nn.Module)
    module("nemo.core.neural_types")
    elements = module("nemo.core.neural_types.elements")
    def element(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any

    elements.__getattr__ = element
    module("nemo.core.neural_types.neural_type", NeuralType=_Any)
    module("nemo.utils", logging=logging.getLogger("nemo"))
    module("nemo.collections.common")
    module("nemo.collections.common.parts")

    def load_file(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    load_file("nemo.collections.common.parts.utils", os.path.join(root, "nemo/collections/common/parts/utils.py"))
    return load_file("nemo.collections.tts.modules.audio_codec_modules", os.path.join(root, "nemo/collections/tts/modules/audio_codec_modules.py"))


def build(m, stored=None):
    torch.manual_seed(61)
    encoder = m.HiFiGANEncoder(**ENCODER)
    decoder = m.CausalHiFiGANDecoder(**DECODER)
    quantizer = m.GroupFiniteScalarQuantizer(**QUANTIZER)
    with torch.no_grad():
        for prefix, net in (("audio_encoder", encoder), ("audio_decoder", decoder)):
            for name, p in net.named_parameters():
                key = f"{prefix}.{name}"
                if stored is not None:
                    p.copy_(torch.tensor(stored[key], dtype=torch.float32).reshape(p.shape))
                elif name.endswith("alpha"):
                    p.copy_(0.5 + torch.rand_like(p))
                elif name == "post_conv.conv.parametrizations.weight.original0":
                    # Larger output gains spread the codes over the levels and drive some samples into the clamp.
                    p.copy_(torch.full_like(p, 6.0))
                elif name.endswith("original0"):
                    p.copy_(0.5 + 0.5 * torch.rand_like(p))
                elif name.endswith("bias"):
                    p.copy_(0.05 * torch.randn_like(p))
                else:
                    p.copy_(torch.randn_like(p))
                p.copy_(p.float())
    return encoder.double().eval(), decoder.double().eval(), quantizer.double().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    m = load_nemo(os.path.abspath(args.nemo))
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    encoder, decoder, quantizer = build(m, None if args.write else stored["state"])
    samples_per_frame = 6
    torch.manual_seed(67)
    audio = (0.5 * torch.randn(1, SAMPLES)).float().double() if args.write else torch.tensor([stored["audio"]])
    audio_len = torch.tensor([SAMPLES])
    with torch.no_grad():
        # AudioCodecModel.pad_audio
        padded_len = samples_per_frame * torch.ceil(audio_len / samples_per_frame).int()
        padded = torch.nn.functional.pad(audio, (0, padded_len.max().item() - audio.shape[1]))
        # encode: encode_audio, then quantize ([C, B, T] -> [B, C, T])
        encoded, encoded_len = encoder(audio=padded, audio_len=padded_len)
        tokens = quantizer.encode(inputs=encoded, input_len=encoded_len).permute(1, 0, 2)
        # decode: dequantize ([B, C, T] -> [C, B, T]), then decode_audio
        dequantized = quantizer.decode(indices=tokens.permute(1, 0, 2), input_len=encoded_len)
        decoded, decoded_len = decoder(inputs=dequantized, input_len=encoded_len)
    torch.set_default_dtype(torch.float32)

    data = {"encoder": ENCODER, "decoder": DECODER, "quantizer": QUANTIZER, "audio": audio[0].tolist(),
            "encoded": encoded[0].reshape(-1).tolist(), "codes": tokens[0].tolist(), "decoded": decoded[0].tolist()}
    if args.write:
        state = {f"audio_encoder.{k}": v.float().reshape(-1).tolist() for k, v in encoder.named_parameters()}
        state.update({f"audio_decoder.{k}": v.float().reshape(-1).tolist() for k, v in decoder.named_parameters()})
        data["state"] = state
        _fixture.save(FIXTURE, data, __file__)
        clamped = sum(1 for v in data["decoded"] if abs(v) >= 1.0)
        print("wrote", FIXTURE, "codes", data["codes"], "samples", len(data["decoded"]), "clamped", clamped)
        return
    check = _fixture.Comparison("NanoCodec")
    check.floats("encoded", stored["encoded"], data["encoded"], 1e-12, float)
    check.exact("codes", stored["codes"], data["codes"])
    check.floats("decoded", stored["decoded"], data["decoded"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
