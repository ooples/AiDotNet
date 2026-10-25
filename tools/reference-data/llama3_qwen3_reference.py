"""Llama 3 and Qwen3 reference: Hugging Face transformers' LlamaForCausalLM with Llama 3's rotary scaling and tied
embeddings (Llama-3.2-1B's form) and Qwen3ForCausalLM (a head width that is not width / heads, per-head query and key
RMSNorm, tied embeddings; Qwen3-0.6B's form), run in float64 on CPU.

Small random models: Llama width 32, two layers of four query heads and two key-value heads, SwiGLU width 64, rotary
base 500000 with rope_type "llama3" (factor 32, low/high frequency factors 1 and 4, original context 16 so that the
scaling's three bands all occur at these sizes); Qwen3 width 32, two layers of four query heads of width 16 (64 in all)
and two key-value heads, base 1000000. Both tie the head to the token table. The reference computes RMSNorm and the
rotary tables in float32 and softmaxes attention in float32; these are patched to float64 (the rotary frequencies with
Llama 3's scaling re-derived in float64 and checked against the reference's float32 values), as the C# side computes
them. For two sequences of seven tokens (the second right-padded by two) the fixture records the final hidden states,
the logits and the causal language-model loss; and, for the first sequence, the logits of its last position computed
one position at a time with the model's KV cache (DynamicCache), which the C# cache must reproduce.
Default: verify; --write: regenerate tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/llama3_qwen3_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/llama3_qwen3_reference.json"
IDS = [[1, 17, 45, 99, 3, 120, 64], [1, 88, 5, 31, 77, 0, 0]]
MASK = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]
LLAMA3 = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16}


def inverse_frequencies(dim, base, scaling):
    import torch
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if scaling is None:
        return inv
    factor, low, high, old = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"], scaling["original_max_position_embeddings"]
    low_wavelen, high_wavelen = old / low, old / high
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > low_wavelen, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < high_wavelen) * ~(wavelen > low_wavelen)
    return torch.where(medium, smoothed, scaled)


def patch_float64(module, prefix):
    import torch

    def rms_forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    def rotary_forward(self, x, position_ids):
        rp = getattr(self.config, "rope_parameters", None)
        if rp is None:  # transformers 4.x: rope_theta and rope_scaling
            rp = {"rope_theta": self.config.rope_theta, **(getattr(self.config, "rope_scaling", None) or {})}
        scaling = LLAMA3 if rp.get("rope_type") == "llama3" else None
        inv = inverse_frequencies(self.inv_freq.shape[0] * 2, rp["rope_theta"], scaling)
        assert torch.allclose(inv.float(), self.inv_freq.float().cpu(), rtol=1e-6), "the float64 frequencies differ from the reference's"
        freqs = position_ids[:, :, None].double() * inv[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def eager_attention(module_, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = module.repeat_kv(key, module_.num_key_value_groups)
        value_states = module.repeat_kv(value, module_.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights

    getattr(module, prefix + "RMSNorm").forward = rms_forward
    getattr(module, prefix + "RotaryEmbedding").forward = rotary_forward
    module.eager_attention_forward = eager_attention


def causal_lm_loss(logits, labels):
    import torch
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)


def build(kind):
    import torch
    torch.manual_seed(23 if kind == "llama3" else 29)
    if kind == "llama3":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, rms_norm_eps=1e-5, max_position_embeddings=64, tie_word_embeddings=True,
                             rope_parameters={"rope_theta": 500000.0, **LLAMA3}, attn_implementation="eager")
        model = LlamaForCausalLM(config)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, max_position_embeddings=64,
                             tie_word_embeddings=True, rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                             attn_implementation="eager")
        model = Qwen3ForCausalLM(config)
    model = model.double().eval()
    with torch.no_grad():  # unit norms would hide a missing weight
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
    return model


def run_case(kind):
    import torch
    model = build(kind)
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    ids = torch.tensor(IDS)
    mask = torch.tensor(MASK)
    labels = ids.masked_fill(mask == 0, -100)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        loss = causal_lm_loss(out.logits, labels)
        # The first sequence one position at a time through the cache.
        from transformers import DynamicCache
        cache = DynamicCache()
        step = None
        for t in range(len(IDS[0])):
            step = model(input_ids=ids[:1, t:t + 1], past_key_values=cache, use_cache=True)
            cache = step.past_key_values
    return {"state": state, "hidden": out.hidden_states[-1].flatten().tolist(), "logits": out.logits.flatten().tolist(),
            "loss": loss.item(), "cached_last_logits": step.logits[0, -1].tolist()}


def compute():
    import torch
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.qwen3.modeling_qwen3 as qwen3
    torch.set_default_dtype(torch.float64)
    patch_float64(llama, "Llama")
    patch_float64(qwen3, "Qwen3")
    return {"ids": IDS, "mask": MASK, "llama3": run_case("llama3"), "qwen3": run_case("qwen3")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["llama3"]["loss"], result["qwen3"]["loss"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(stored[m]["loss"] - result[m]["loss"]) < 1e-12 for m in ("llama3", "qwen3"))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
