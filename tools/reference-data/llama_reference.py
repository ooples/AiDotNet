"""LLaMA reference: Hugging Face transformers' LlamaForCausalLM, run in float64 on CPU.

A small model with the LLaMA 2 architecture: width 32, two decoder layers of four query heads and two key-value heads
(grouped-query attention, which LLaMA 2 70B uses; 7B has equal counts), SwiGLU width 64, vocabulary 128, RMSNorm epsilon
1e-5 and rotary base 10000. All weights are random (state dict recorded). The reference computes RMSNorm and the rotary
tables in float32 even in a float64 model, softmaxes attention in float32, and casts the logits to float32 for the
loss. The norm, the rotary tables and the softmax are patched to float64 here, and the loss (the same shifted
cross-entropy) is computed in float64, as the C# side computes them, so the comparison is not limited to float32 precision.

For two sequences of seven tokens (the second right-padded by two), the fixture records the final normalized hidden
states (the last hidden_states entry), the logits, and the causal language-model loss on labels equal to the ids with
padding ignored (-100), and that loss after one plain SGD step (lr 0.01) on every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/llama_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/llama_reference.json"
IDS = [[1, 17, 45, 99, 3, 120, 64], [1, 88, 5, 31, 77, 0, 0]]
MASK = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]]


def patch_float64():
    import torch
    import transformers.models.llama.modeling_llama as m

    def rms_forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    def rotary_forward(self, x, position_ids):
        dim = self.inv_freq.shape[0] * 2
        base = self.config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        freqs = position_ids[:, :, None].double() * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = m.repeat_kv(key, module.num_key_value_groups)
        value_states = m.repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = torch.nn.functional.softmax(weights, dim=-1)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
        return output, weights

    m.LlamaRMSNorm.forward = rms_forward
    m.LlamaRotaryEmbedding.forward = rotary_forward
    m.eager_attention_forward = eager_attention


def causal_lm_loss(logits, labels):
    import torch
    shifted = labels[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), shifted, ignore_index=-100)


def compute():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_default_dtype(torch.float64)
    patch_float64()
    torch.manual_seed(11)
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=64,
                         tie_word_embeddings=False, attn_implementation="eager")
    model = LlamaForCausalLM(config).double().eval()
    with torch.no_grad():  # unit norms would hide a missing weight
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
    ids = torch.tensor(IDS)
    mask = torch.tensor(MASK)
    labels = ids.masked_fill(mask == 0, -100)

    def run():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return out.hidden_states[-1], out.logits, causal_lm_loss(out.logits, labels)

    hidden, logits, loss = run()
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        _, _, after = run()
    return {"state": state, "ids": IDS, "mask": MASK, "hidden": hidden.detach().flatten().tolist(),
            "logits": logits.detach().flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["loss"], result["loss_after_sgd"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(stored[k] - result[k]) < 1e-12 for k in ("loss", "loss_after_sgd"))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
