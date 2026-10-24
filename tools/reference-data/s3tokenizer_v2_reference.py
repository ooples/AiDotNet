"""S3 speech tokenizer v2 reference: xingchensong/S3Tokenizer (commit 9bf5d845b5e043ffaf4657f4942939091c7697a2,
s3tokenizer/model_v2.py), the PyTorch form of CosyVoice 2's speech_tokenizer_v2_25hz.onnx. Pass the checkout with
--s3tokenizer.

Builds a tiny tokenizer (heads 64 wide, as v2's rotary table is) whose weights follow an exact integer formula of each
parameter's name and index (weights(), which the C# test repeats, so no weights are stored) and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/s3tokenizer_v2_reference.json the encoder states and FSQ tokens of a
padded batch of two log-mels (lengths 13 and 10), and the tokens of a 3100-frame log-mel, which takes the 30 s windowing
path. The float32 steps s3tokenizer_reference.py removes are removed here too, and the rotary table is float64.
Default: verify; --write: regenerate.
"""
import argparse

import torch

import _fixture
from s3tokenizer_reference import load_package

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/s3tokenizer_v2_reference.json"
CONFIG = dict(n_mels=4, n_audio_ctx=1500, n_audio_state=128, n_audio_head=2, n_audio_layer=2, n_codebook_size=3 ** 8)
BATCH_LENGTHS = [13, 10]
LONG_FRAMES = 3100


def weights(name, count):
    h = sum((j + 1) * ord(ch) for j, ch in enumerate(name)) % 100003
    base = 1.0 if ("ln" in name and name.endswith("weight")) else 0.0
    # The FSQ projection is scaled up so its 8 values spread over all three levels, not only 0.
    scale = 8.0 if "project_down" in name else 0.2
    # A quadratic index term, so the weights look random rather than repeat a linear ramp.
    return [base + scale * (((i * i * 7919 + i * 104729 + h) % 2003) / 2003.0 - 0.5) for i in range(count)]


def build(root):
    load_package(root)
    import s3tokenizer.model_v2 as v2

    def mask_to_bias(mask, dtype):
        assert mask.dtype == torch.bool
        return (1.0 - mask.to(dtype)) * -1.0e+10
    v2.mask_to_bias = mask_to_bias
    torch.set_default_dtype(torch.float64)
    model = v2.S3TokenizerV2("speech_tokenizer_v2_25hz", v2.ModelConfig(**CONFIG)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.tensor(weights(name, p.numel())).reshape(p.shape))
        # precompute_freqs_cis builds the table in float32; the same formula in float64.
        freqs = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        angles = torch.outer(torch.arange(2048, dtype=torch.float64), freqs)
        table = torch.polar(torch.ones_like(angles), angles)
        model.encoder.freqs_cis = torch.cat((table, table), dim=-1)
    return model


def run(model, batch, long):
    lengths = torch.tensor(BATCH_LENGTHS)
    with torch.no_grad():
        hidden, hidden_lengths = model.encoder(batch, lengths)
    codes, code_lengths = model.quantize(batch, lengths)
    long_codes, long_length = model.quantize(long, torch.tensor([LONG_FRAMES]))
    hidden = [[float(v) for v in hidden[i, :hidden_lengths[i]].reshape(-1).tolist()] for i in range(len(BATCH_LENGTHS))]
    codes = [codes[i, :code_lengths[i]].tolist() for i in range(len(BATCH_LENGTHS))]
    return hidden, codes, long_codes[0, :long_length[0]].tolist()


def inputs():
    generator = torch.Generator().manual_seed(7)
    batch = torch.randn(len(BATCH_LENGTHS), CONFIG["n_mels"], max(BATCH_LENGTHS), generator=generator, dtype=torch.float64)
    for i, length in enumerate(BATCH_LENGTHS):
        batch[i, :, length:] = 0
    long = torch.randn(1, CONFIG["n_mels"], LONG_FRAMES, generator=generator, dtype=torch.float64)
    return batch, long


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3tokenizer", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    model = build(args.s3tokenizer)
    batch, long = inputs()
    hidden, codes, long_codes = run(model, batch, long)
    if args.write:
        data = {"config": CONFIG, "batch_lengths": BATCH_LENGTHS,
                "parameters": [[n, list(p.shape)] for n, p in model.named_parameters()],
                "batch": [float(v) for v in batch.reshape(-1).tolist()], "hidden": hidden, "codes": codes,
                "long": [float(v) for v in long.reshape(-1).tolist()], "long_codes": long_codes}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE, "distinct codes:", len(set(sum(codes, []) + long_codes)))
        return
    data = _fixture.load(FIXTURE)
    check = _fixture.Comparison("S3 tokenizer v2")
    for i, h in enumerate(hidden):
        check.floats(f"hidden[{i}]", data["hidden"][i], h, 1e-12, float)
    check.exact("codes", data["codes"], codes)
    check.exact("long_codes", data["long_codes"], long_codes)
    check.report(exact=False)


if __name__ == "__main__":
    main()
