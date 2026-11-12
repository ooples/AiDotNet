"""StyleTTS-ZS conformer reference: wenet's ConformerEncoderLayer with ESPnet relative positional attention (as vendored in
FunAudioLLM/CosyVoice, cosyvoice/transformer), the reproduction of Gulati et al.'s Conformer that StyleTTS-ZS's
conformer stacks follow (the paper names the block, Li et al. 2024 App. D, without code).

Builds a small stack (EspnetRelPositionalEncoding, then blocks with a macaron Swish feed-forward, relative-position
attention, the convolution module with BatchNorm and Swish, kernels 5 and 3) from a checkout passed with --cosyvoice,
stores its weights in float32 (batch-norm statistics and the u/v biases off their defaults) and records the stack's
output for a random input in float64 and evaluation mode. The relative position table is built in float64 (the
reference builds it in float32 whatever the default dtype). in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts_zs_conformer_reference.json. Default: verify; --write.
"""
import argparse
import base64
import inspect
import os
import sys
import textwrap
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/styletts_zs_conformer_reference.json"
DIM, HEADS, FEED_FORWARD, KERNELS, LENGTH = 16, 2, 24, [5, 3], 9


def load(root):
    package = types.ModuleType("cosyvoice")
    package.__path__ = [os.path.join(root, "cosyvoice")]
    sys.modules["cosyvoice"] = package
    from cosyvoice.transformer import attention, convolution, embedding, encoder_layer, positionwise_feed_forward, activation
    source = textwrap.dedent(inspect.getsource(embedding.EspnetRelPositionalEncoding.extend_pe))
    namespace = {}
    exec(source.replace("dtype=torch.float32", "dtype=torch.float64"), embedding.__dict__, namespace)
    embedding.EspnetRelPositionalEncoding.extend_pe = namespace["extend_pe"]
    return attention, convolution, embedding, encoder_layer, positionwise_feed_forward, activation


def attention(state=None):
    torch.manual_seed(77)
    module = torch.nn.MultiheadAttention(DIM, HEADS, batch_first=True)
    if state is not None:
        module.load_state_dict(state, strict=True)
    return module.eval()


def attention_inputs():
    g = torch.Generator().manual_seed(8)
    return torch.randn(1, 4, DIM, generator=g, dtype=torch.float32), torch.randn(1, LENGTH, DIM, generator=g, dtype=torch.float32)


def run_attention(module):
    query, memory = attention_inputs()
    with torch.no_grad():
        out, _ = module(query.double(), memory.double(), memory.double(), need_weights=False)
    return [float(v) for v in out.reshape(-1).tolist()]


class Stack(torch.nn.Module):
    def __init__(self, loaded):
        super().__init__()
        attention, convolution, embedding, encoder_layer, ffn, activation = loaded
        self.pos = embedding.EspnetRelPositionalEncoding(DIM, 0.0, max_len=64)
        self.encoders = torch.nn.ModuleList([
            encoder_layer.ConformerEncoderLayer(
                DIM, attention.RelPositionMultiHeadedAttention(HEADS, DIM, 0.0),
                ffn.PositionwiseFeedForward(DIM, FEED_FORWARD, 0.0, activation.Swish()),
                ffn.PositionwiseFeedForward(DIM, FEED_FORWARD, 0.0, activation.Swish()),
                convolution.ConvolutionModule(DIM, k, activation.Swish(), "batch_norm", False, True), 0.0, True)
            for k in KERNELS])

    def forward(self, x):
        x, pos = self.pos(x)
        mask = torch.ones(1, x.size(1), x.size(1), dtype=torch.bool)
        for layer in self.encoders:
            x = layer(x, mask, pos)[0]
        return x


def build(loaded, state=None):
    torch.manual_seed(2294)
    stack = Stack(loaded)
    if state is None:
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, p in stack.named_parameters():
                if "pos_bias" in name:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            for m in stack.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                    m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                    m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
    else:
        stack.load_state_dict(state, strict=True)
    return stack.eval()


def inputs():
    return torch.randn(1, LENGTH, DIM, generator=torch.Generator().manual_seed(7), dtype=torch.float32)


def run(stack):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.no_grad():
            stack.pos.pe = None
            stack.pos.extend_pe(torch.zeros(1, 64))
            return [float(v) for v in stack(inputs().double()).reshape(-1).tolist()]
    finally:
        torch.set_default_dtype(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True, help="a checkout of FunAudioLLM/CosyVoice")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    loaded = load(args.cosyvoice)
    if args.write:
        state = {k: v.float().clone().contiguous() for k, v in build(loaded).state_dict().items()}
        data = {"dim": DIM, "heads": HEADS, "feed_forward": FEED_FORWARD, "kernels": KERNELS,
                "input": [float(v) for v in inputs().reshape(-1).tolist()],
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "output": run(build(loaded, state).double())}
        astate = {k: v.float().clone().contiguous() for k, v in attention().state_dict().items()}
        query, memory = attention_inputs()
        data["attention"] = {"query": [float(v) for v in query.reshape(-1).tolist()], "memory": [float(v) for v in memory.reshape(-1).tolist()],
                             "safetensors_base64": base64.b64encode(save_safetensors(astate)).decode("ascii"),
                             "output": run_attention(attention(astate).double())}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    again = run(build(loaded, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double())
    check = _fixture.Comparison("StyleTTS-ZS conformer")
    check.floats("output", data["output"], again, 1e-12, float)
    astate = load_safetensors(base64.b64decode(data["attention"]["safetensors_base64"]))
    check.floats("attention", data["attention"]["output"], run_attention(attention(astate).double()), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
