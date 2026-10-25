"""LFM2 (Liquid AI) as transformers 4.57.1's Lfm2ForCausalLM computes it, with peft's rank-stabilized LoRA.

A tiny model (width 16, four heads over two key-value heads, layers conv, attention, conv, attention, a 300-token
vocabulary, the head tied to the token table) with random weights, run in float64 with the reference's float32 casts
(RMSNorm, the rotary tables, the attention softmax) patched as in llama3_qwen3_reference.py. Records in
tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/lfm2_reference.json:
- the logits of a 9-token sequence;
- greedy generation of 6 tokens through generate() (the hybrid cache: keys and values, and the convolution states);
- with LoRA (rank 4, alpha 8, rsLoRA, on q_proj, in_proj, out_proj, w1 and w2, B random so the adapters act): the
  logits, the causal LM loss on the sequence and the gradients of two adapters' weights.
Default: verify; --write: regenerate. Needs the .venv-kani environment (torch 2.8.0, transformers 4.57.1, peft 0.17.1).
"""
import argparse

import numpy as np
import torch

import _fixture
from llama3_qwen3_reference import patch_float64

FIXTURE = "tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/lfm2_reference.json"
CONFIG = dict(vocab_size=300, hidden_size=16, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
              layer_types=["conv", "full_attention", "conv", "full_attention"], conv_L_cache=3, conv_bias=False, norm_eps=1e-5,
              rope_theta=1000000.0, max_position_embeddings=128, block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.0,
              block_multiple_of=8, tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=7)
LORA = dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "in_proj", "out_proj", "w1", "w2"], use_rslora=True,
            bias="none", task_type="CAUSAL_LM")
IDS = [1, 17, 250, 3, 99, 42, 120, 8, 77]
GRADIENTS = ["model.layers.0.conv.in_proj.lora_A.weight", "model.layers.1.self_attn.q_proj.lora_B.weight"]


def build(state=None):
    from transformers import Lfm2Config, Lfm2ForCausalLM
    torch.manual_seed(41)
    model = Lfm2ForCausalLM(Lfm2Config(**CONFIG, attn_implementation="eager"))
    with torch.no_grad():
        if state is None:
            for name, p in model.named_parameters():
                if name.endswith("norm.weight"):
                    p.copy_(1 + 0.1 * torch.randn_like(p))
                elif name.endswith("conv.conv.weight"):
                    p.copy_(0.3 * torch.randn_like(p))
                else:
                    p.copy_(0.2 * torch.randn_like(p))
        else:
            model.load_state_dict(state, strict=False)
    return model.double().eval()


def lora_state(model, stored=None):
    """peft's adapter weights by module name without the adapter name; random A and B so the adapters act."""
    from peft import LoraConfig, get_peft_model
    peft_model = get_peft_model(model, LoraConfig(**LORA))
    torch.manual_seed(43)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_" not in name:
                continue
            key = name.replace("base_model.model.", "").replace(".default", "")
            if stored is None:
                p.copy_(0.2 * torch.randn_like(p))
            else:
                p.copy_(torch.tensor(stored[key], dtype=torch.float64).reshape(p.shape))
    return peft_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import transformers.models.lfm2.modeling_lfm2 as lfm2
    patch_float64(lfm2, "Lfm2")
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    if args.write:
        state32 = {k: v.float().clone() for k, v in build().state_dict().items() if k != "lm_head.weight"}
    else:
        shapes = {k: v.shape for k, v in build().state_dict().items()}
        state32 = {k: torch.tensor(v, dtype=torch.float32).reshape(shapes[k]) for k, v in stored["state"].items()}
    model = build({k: v for k, v in state32.items()})
    ids = torch.tensor([IDS])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
        generated = model.generate(ids, do_sample=False, max_new_tokens=6, min_new_tokens=6, eos_token_id=None, pad_token_id=0)[0, len(IDS):].tolist()

    peft_model = lora_state(model, None if args.write else stored["lora"])
    lora = {name.replace("base_model.model.", "").replace(".default", ""): p for name, p in peft_model.named_parameters() if "lora_" in name}
    if args.write:
        with torch.no_grad():
            for p in lora.values():
                p.copy_(p.float().double())
    out = peft_model(input_ids=ids).logits
    # Precision patch: transformers' loss casts the logits to float32; the same shifted cross-entropy in float64.
    loss = torch.nn.functional.cross_entropy(out[0, :-1], ids[0, 1:])
    loss.backward()
    lora_logits = out[0].detach()
    gradients = {name: lora[name].grad.reshape(-1).tolist() for name in GRADIENTS}
    torch.set_default_dtype(torch.float32)

    data = {"config": CONFIG, "lora_config": LORA, "ids": IDS, "logits": logits.reshape(-1).tolist(), "generated": generated,
            "lora_logits": lora_logits.reshape(-1).tolist(), "loss": float(loss.detach()), "gradients": gradients}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        data["lora"] = {k: v.detach().reshape(-1).tolist() for k, v in lora.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "generated", generated, "loss", float(loss))
        return
    check = _fixture.Comparison("LFM2")
    check.floats("logits", stored["logits"], data["logits"], 1e-12, float)
    check.exact("generated", stored["generated"], generated)
    check.floats("lora logits", stored["lora_logits"], data["lora_logits"], 1e-12, float)
    check.floats("loss", [stored["loss"]], [data["loss"]], 1e-12, float)
    for name in GRADIENTS:
        check.floats(name, stored["gradients"][name], gradients[name], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
