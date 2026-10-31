"""GLM-4's decoder as THUDM's modeling_chatglm.py builds it (GLM-4-Voice-9B's remote code).

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_decoder_reference.json, in float64 on float32 weights,
a tiny ChatGLMForConditionalGeneration's state and the logits of token ids (eager CoreAttention, no cache). The remote
code's float32 casts (RMSNorm's variance, the rotary cache, the attention softmax) are replaced by float64 so the
comparison holds at 1e-9; query-key layer scaling, which multiplies and divides by the layer number, is turned off.

Usage (default: verify; --write: regenerate), with the model's remote code in <dir> (glm-4-voice-9b's
configuration_chatglm.py and modeling_chatglm.py):
  python tools/reference-data/glm_decoder_reference.py --code <dir> [--write]
"""
import argparse
import importlib.util
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_decoder_reference.json"


def load(code):
    package = types.ModuleType("chatglm_remote")
    package.__path__ = [code]
    sys.modules["chatglm_remote"] = package
    modules = {}
    for name in ("configuration_chatglm", "modeling_chatglm"):
        spec = importlib.util.spec_from_file_location(f"chatglm_remote.{name}", os.path.join(code, name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"chatglm_remote.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["configuration_chatglm"], modules["modeling_chatglm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--code", required=True)
    args = parser.parse_args()
    configuration, modeling = load(args.code)

    def norm_forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.eps))

    def rotary_impl(self, seq_len, n_elem, dtype, device, base=10000):
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float64, device=device) / n_elem))
        idx_theta = torch.outer(torch.arange(seq_len, dtype=torch.float64, device=device), theta)
        return torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    modeling.RMSNorm.forward = norm_forward
    modeling.RotaryEmbedding.forward_impl = rotary_impl
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(13)
    config = configuration.ChatGLMConfig(
        num_layers=2, padded_vocab_size=50, hidden_size=32, ffn_hidden_size=40, kv_channels=8, num_attention_heads=4,
        multi_query_attention=True, multi_query_group_num=2, add_qkv_bias=True, add_bias_linear=False, rmsnorm=True,
        layernorm_epsilon=1e-5, post_layer_norm=True, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=False,
        original_rope=True, rope_ratio=1, hidden_dropout=0.0, attention_dropout=0.0, seq_length=64, torch_dtype="float64",
        fp32_residual_connection=False, apply_residual_connection_post_layernorm=False)
    config._attn_implementation = "eager"
    model = modeling.ChatGLMForConditionalGeneration(config, empty_init=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored:
                p.copy_(torch.tensor(stored["state"][name]).reshape(p.shape))
            elif "layernorm" in name:
                p.copy_((1 + 0.2 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
    model = model.double().eval()
    ids = torch.tensor([stored["ids"]]) if stored else torch.randint(0, 50, (1, 9))
    with torch.no_grad():
        hidden = model.transformer(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        logits = model.transformer.output_layer(hidden)
    out = {"logits": logits.reshape(-1)}
    if args.write:
        data = {"ids": ids[0].tolist(), "state": {n: _fixture.flat(p) for n, p in model.named_parameters()}}
        data.update({k: [float(x) for x in v.tolist()] for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, list(logits.shape))
        return
    check = _fixture.Comparison("GLM-4 decoder")
    check.floats("logits", stored["logits"], [float(x) for x in out["logits"].tolist()], dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
