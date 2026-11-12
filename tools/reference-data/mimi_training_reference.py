"""Mimi training-forward reference (Défossez et al. 2024 §3.3; the release's MimiModel.forward and split RVQ in training mode,
moshi/quantization/core_vq.py): a small Mimi (SEANet ratios 4·2, width 16, one 16-wide Transformer layer per side with 2
heads, 4 codebooks of 8 entries in 8 dimensions), every parameter and buffer by the closed form of mimi_reference.py (the
codebooks initialized, so no k-means runs), two training steps on a second of audio with no quantizer dropout and full
quantization.

Records each step's reconstruction (cut to the input), the semantic quantizer's output, the commitment penalty, and every
codebook's embedding_sum and cluster_usage after the step (the exponential moving averages; the dead-code check falls on
the fifth step, so neither step replaces codes). The release's rotary embedding (float32) and its float64 attention mask
kernel are replaced as in mimi_reference.py.
Default: verify; --write: regenerate.
"""
import math
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

torch.set_default_dtype(torch.float64)
sys.modules.setdefault("bitsandbytes", types.ModuleType("bitsandbytes"))
import moshi.modules.rope as rope  # noqa: E402
from moshi.models.compression import MimiModel  # noqa: E402
from moshi.modules import SEANetDecoder, SEANetEncoder, transformer  # noqa: E402
from moshi.quantization import SplitResidualVectorQuantizer  # noqa: E402
from mimi_reference import apply_rope64, attention64, closed_form  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mimi_training_reference.json"
rope.apply_rope = lambda q, k, offset, max_period=10_000, interleave=True, time_before_heads=False: apply_rope64(q, k, offset, max_period, time_before_heads)
transformer.F.scaled_dot_product_attention = attention64

SEANET = dict(channels=1, dimension=16, causal=True, n_filters=4, n_residual_layers=1, activation="ELU", compress=2, dilation_base=2,
              disable_norm_outer_blocks=0, kernel_size=7, residual_kernel_size=3, last_kernel_size=3, norm="none", pad_mode="constant",
              ratios=[4, 2], true_skip=True)
TRANSFORMER = dict(d_model=16, num_heads=2, num_layers=1, causal=True, layer_scale=0.01, context=6, conv_layout=True, max_period=10000,
                   gating="none", norm="layer_norm", positional_embedding="rope", dim_feedforward=32, input_dimension=16, output_dimensions=[16])
QUANTIZER = dict(dimension=8, n_q=4, bins=8, input_dimension=16, output_dimension=16)
SAMPLE_RATE, FRAME_RATE = 1600, 100


def build():
    encoder = SEANetEncoder(**SEANET)
    model = MimiModel(encoder, SEANetDecoder(**SEANET), SplitResidualVectorQuantizer(**QUANTIZER), channels=1, sample_rate=SAMPLE_RATE,
                      frame_rate=FRAME_RATE, encoder_frame_rate=SAMPLE_RATE / encoder.hop_length, causal=True, resample_method="conv",
                      encoder_transformer=transformer.ProjectedTransformer(**TRANSFORMER),
                      decoder_transformer=transformer.ProjectedTransformer(**TRANSFORMER))
    model.set_num_codebooks(4)
    return model


def main():
    args = _fixture.parse_args(__doc__)
    model = build()
    shapes = {}
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name.endswith("._embedding"):
                continue
            shapes[name] = list(value.shape)
            value.copy_(closed_form(name, list(value.shape)).reshape(value.shape))
        for module in model.modules():
            if hasattr(module, "_embedding"):
                module._embedding = None
            if hasattr(module, "_cached_initialized"):
                module._cached_initialized = False
    model.train()
    n = SAMPLE_RATE
    i = torch.arange(n, dtype=torch.float64)
    audio = 0.4 * torch.sin(2 * math.pi * 110 * i / SAMPLE_RATE * (1 + i / 4000)) + 0.1 * torch.sin(0.37 * i)
    steps = []
    for step in range(2):
        captured = {}
        hook = model.quantizer.rvq_first.register_forward_hook(lambda m, inp, out: captured.update(semantic=out.x.detach()))
        result = model(audio.view(1, 1, -1))
        hook.remove()
        books = []
        for rvq in (model.quantizer.rvq_first, model.quantizer.rvq_rest):
            for layer in rvq.vq.layers:
                cb = layer._codebook
                books.append({"sum": cb.embedding_sum.reshape(-1).tolist(), "usage": cb.cluster_usage.tolist()})
        steps.append({"audio": result.x.detach().reshape(-1).tolist(), "semantic": captured["semantic"].reshape(-1).tolist(),
                      "penalty": float(result.penalty), "codebooks": books})
    data = {"steps": steps}
    if args.write:
        _fixture.save(FIXTURE, {"shapes": shapes, "input": audio.tolist(), **data}, __file__)
        print("wrote", FIXTURE, [s["penalty"] for s in steps], len(steps[0]["audio"]))
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Mimi training")
    for s, (want, got) in enumerate(zip(stored["steps"], data["steps"])):
        for key in ("audio", "semantic"):
            check.floats(f"step {s} {key}", want[key], got[key], 1e-12, float)
        check.floats(f"step {s} penalty", [want["penalty"]], [got["penalty"]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
