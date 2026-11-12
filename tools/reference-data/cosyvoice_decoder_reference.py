"""CosyVoice flow estimator reference: FunAudioLLM/CosyVoice's ConditionalDecoder (cosyvoice/flow/decoder.py), a Matcha-TTS
U-Net built from the Matcha components CosyVoice pins as a submodule (shivammehta25/Matcha-TTS at
dd9105b34bf2be2230f4aa1e4769fb586a3c824e) and diffusers' BasicTransformerBlock. Pass checkouts with --cosyvoice and
--matcha.

Builds a tiny randomly initialized estimator (seeded; GELU feed-forward and several Transformer blocks per level, as
CosyVoice configures it), runs it in float64 and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_decoder_reference.json the vector field it gives for one
x, mask, mu, t, speaker embedding and condition, and that field after one SGD step (lr 0.1) on its mean square.
Default: verify; --write: regenerate.
"""
import argparse
import base64
import math
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_decoder_reference.json"
MEL = 4
CONFIG = dict(in_channels=4 * MEL, out_channels=MEL, channels=[8, 8], dropout=0.0, attention_head_dim=4, n_blocks=2,
              num_mid_blocks=2, num_heads=2, act_fn="gelu")
FRAMES = 8
T = 0.37
LEARNING_RATE = 0.1


def build(cosyvoice_root, matcha_root, state=None):
    sys.path.insert(0, matcha_root)
    sys.path.insert(0, cosyvoice_root)
    import cosyvoice.flow.decoder as decoder
    from cosyvoice.flow.decoder import ConditionalDecoder

    def mask_to_bias(mask, dtype):
        # cosyvoice.utils.common.mask_to_bias without its float32/16 assertion, so the estimator can run in float64.
        assert mask.dtype == torch.bool
        return (1.0 - mask.to(dtype)) * -1.0e+10
    decoder.mask_to_bias = mask_to_bias

    def sinusoidal(self, x, scale=1000):
        # Matcha's SinusoidalPosEmb takes its frequencies in float32 (.float()); in float64 here.
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        step = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float64) * -step)
        emb = scale * x.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    decoder.SinusoidalPosEmb.forward = sinusoidal
    torch.manual_seed(2294)
    model = ConditionalDecoder(**CONFIG).eval()
    with torch.no_grad():
        # Biases and norms start at constants (initialize_weights); spread them so each is exercised.
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name or name.endswith("block.1.weight"):
                p.add_(0.1 * torch.randn_like(p))
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def run(model, x, mask, mu, spks, cond):
    t = torch.tensor([T], dtype=torch.float64)
    field = model(x, mask, mu, t, spks=spks, cond=cond)
    loss = (field ** 2).mean()
    parameters = list(model.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
        after = model(x, mask, mu, t, spks=spks, cond=cond)
    return field, after


def inputs(values=None):
    if values is None:
        torch.manual_seed(7)
        return [torch.randn(1, MEL, FRAMES), torch.randn(1, MEL, FRAMES), torch.randn(1, MEL), torch.randn(1, MEL, FRAMES)]
    return [torch.tensor(values["x"], dtype=torch.float64).reshape(1, MEL, FRAMES),
            torch.tensor(values["mu"], dtype=torch.float64).reshape(1, MEL, FRAMES),
            torch.tensor(values["spks"], dtype=torch.float64).reshape(1, MEL),
            torch.tensor(values["cond"], dtype=torch.float64).reshape(1, MEL, FRAMES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True, help="a checkout of FunAudioLLM/CosyVoice")
    parser.add_argument("--matcha", required=True, help="a checkout of shivammehta25/Matcha-TTS at CosyVoice's pinned commit")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture")
    args = parser.parse_args()
    mask = torch.ones(1, 1, FRAMES, dtype=torch.float64)
    if args.write:
        model = build(args.cosyvoice, args.matcha)
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        x, mu, spks, cond = inputs()
        field, after = run(model.double(), x.double(), mask, mu.double(), spks.double(), cond.double())
        data = {"config": CONFIG, "mel": MEL, "frames": FRAMES, "t": T, "learning_rate": LEARNING_RATE,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "x": [float(v) for v in x.double().reshape(-1).tolist()], "mu": [float(v) for v in mu.double().reshape(-1).tolist()],
                "spks": [float(v) for v in spks.double().reshape(-1).tolist()], "cond": [float(v) for v in cond.double().reshape(-1).tolist()],
                "field": [float(v) for v in field.detach().reshape(-1).tolist()],
                "field_after_sgd": [float(v) for v in after.detach().reshape(-1).tolist()]}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(args.cosyvoice, args.matcha, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    x, mu, spks, cond = inputs(data)
    field, after = run(model.double(), x, mask, mu, spks, cond)
    check = _fixture.Comparison("CosyVoice decoder")
    check.floats("field", data["field"], field.detach().reshape(-1).tolist(), 1e-12, float)
    check.floats("field_after_sgd", data["field_after_sgd"], after.detach().reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
