"""SoundStorm Conformer reference (Pheme's acoustic stage): Pheme's own modules/conformer.py.

Builds a tiny randomly initialized Conformer with Pheme's code (PolyAI-LDN/pheme, modules/conformer.py, after
lucidrains/soundstorm-pytorch; pass a checkout with --pheme, the code is not vendored here), stores its weights in
float32, runs it in float64 on a float32-representable input, and records weights, input and output in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/soundstorm_conformer_reference.json.

ChanLayerNorm switches its epsilon to 1e-4 for any dtype other than float32; the released models run in float32 with
1e-6, so the float64 run here keeps 1e-6 (see _chan_layer_norm_float32_epsilon). The attention uses the einsum path
(attn_flash=False), the same arithmetic as the flash kernel. Default: verify; --write: regenerate.
"""
import argparse
import base64
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/soundstorm_conformer_reference.json"
CONFIG = dict(dim=16, num_layers=2, heads=2, dim_head=8, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=5)
TIME = 7


def _chan_layer_norm_float32_epsilon(self, x):
    var = torch.var(x, dim=1, unbiased=False, keepdim=True)
    mean = torch.mean(x, dim=1, keepdim=True)
    return (x - mean) * var.clamp(min=1e-6).rsqrt() * self.gamma


def build(conformer_module, state=None):
    torch.manual_seed(2294)
    model = conformer_module.Conformer(**CONFIG, attn_dropout=0.0, ff_dropout=0.0, conv_dropout=0.0,
                                       attn_flash=False, t5_rel_pos_bias=False).eval()
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pheme", required=True, help="a checkout of PolyAI-LDN/pheme")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    sys.path.insert(0, args.pheme)
    from modules import conformer as conformer_module
    conformer_module.ChanLayerNorm.forward = _chan_layer_norm_float32_epsilon

    if args.write:
        model = build(conformer_module)
        state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
        torch.manual_seed(7)
        x = torch.randn(1, TIME, CONFIG["dim"])
        with torch.no_grad():
            y = model.double()(x.double(), None)
        data = {"config": CONFIG, "time": TIME, "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "input": [float(v) for v in x.reshape(-1).tolist()], "output": [float(v) for v in y.reshape(-1).tolist()]}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(conformer_module, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    x = torch.tensor(data["input"], dtype=torch.float64).reshape(1, data["time"], CONFIG["dim"])
    with torch.no_grad():
        y = model.double()(x, None)
    check = _fixture.Comparison("SoundStorm Conformer")
    check.floats("output", data["output"], y.reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
