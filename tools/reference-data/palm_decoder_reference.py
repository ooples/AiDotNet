"""The PaLM-family decoders AudioPaLM builds on: lucidrains/PaLM-pytorch's PaLM and transformers' Gemma.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/palm_decoder_reference.json, in float64 on float32 weights,
the logits of token ids through a tiny PaLM (PaLM-pytorch 0.2.2, its logits scaled by 1/sqrt(d) as PaLM §5 states) and a
tiny GemmaForCausalLM (transformers 4.57.1, multi-query, eager attention), with their states. The float32 casts of
both references (PaLM-pytorch's rotary frequencies; Gemma's RMSNorm, rotary tables and attention softmax) are
replaced by float64 so the comparison holds at 1e-9.

Usage (default: verify; --write: regenerate), in a venv with torch, einops, PaLM-pytorch and transformers:
  python tools/reference-data/palm_decoder_reference.py [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/palm_decoder_reference.json"


def floats(t):
    return [float(x) for x in t.detach().reshape(-1).tolist()]


def fill(module, stored, key, scale=0.3, norm_offset=None):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if stored:
                p.copy_(torch.tensor(stored[key][name]).reshape(p.shape))
            elif norm_offset is not None and ("norm" in name or "gamma" in name):
                p.copy_((norm_offset + 0.2 * torch.randn_like(p)).float())
            else:
                p.copy_((scale * torch.randn_like(p)).float())


def palm(stored):
    import palm_pytorch.palm_pytorch as lucid

    class Rotary(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)))

        def forward(self, max_seq_len, *, device):
            seq = torch.arange(max_seq_len, device=device, dtype=torch.float64)
            freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
            return torch.cat((freqs, freqs), dim=-1)
    lucid.RotaryEmbedding = Rotary
    net = lucid.PaLM(dim=32, num_tokens=50, depth=2, dim_head=8, heads=4, ff_mult=4)
    fill(net, stored, "palm_state", norm_offset=1.0)
    net = net.double().eval()
    ids = torch.tensor([stored["palm_ids"]]) if stored else torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        logits = net(ids) * 32 ** -0.5
    return ids, net, logits


def gemma(stored):
    import transformers.models.gemma.modeling_gemma as modeling
    from transformers import GemmaConfig, GemmaForCausalLM

    def norm_forward(self, x):
        return self._norm(x) * (1.0 + self.weight)

    def rotary_forward(self, x, position_ids):
        dim = 2 * self.inv_freq.shape[0]
        inv = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        freqs = position_ids[0].to(torch.float64)[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)[None]
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    source = modeling.eager_attention_forward

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = modeling.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1)
        out = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return out, weights
    assert source is not None
    modeling.GemmaRMSNorm.forward = norm_forward
    modeling.GemmaRotaryEmbedding.forward = rotary_forward
    modeling.eager_attention_forward = eager
    config = GemmaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
                         attn_implementation="eager", tie_word_embeddings=True)
    model = GemmaForCausalLM(config)
    fill(model, stored, "gemma_state")
    model = model.double().eval()
    ids = torch.tensor([stored["gemma_ids"]]) if stored else torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    return ids, model, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(21)
    palm_ids, palm_net, palm_logits = palm(stored)
    gemma_ids, gemma_net, gemma_logits = gemma(stored)
    out = {"palm_logits": palm_logits[0], "gemma_logits": gemma_logits[0]}
    if args.write:
        data = {"palm_ids": palm_ids[0].tolist(), "gemma_ids": gemma_ids[0].tolist(),
                "palm_state": {n: _fixture.flat(p) for n, p in palm_net.named_parameters()},
                "gemma_state": {n: _fixture.flat(p) for n, p in gemma_net.named_parameters()}}
        data.update({k: floats(v) for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, {k: list(v.shape) for k, v in out.items()})
        return
    check = _fixture.Comparison("PaLM-family decoders")
    for k, v in out.items():
        check.floats(k, stored[k], floats(v), dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
