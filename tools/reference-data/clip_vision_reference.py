"""CLIP vision reference: Hugging Face transformers' CLIPVisionModel, run in float64 on CPU.

A small model with CLIP ViT's architecture (quick-GELU, pre-norm layers, a LayerNorm before the transformer): images of
28 px in 7 px patches (16 patches and a class token), width 32, four layers of four heads, feed-forward width 64. All
weights are random (state dict recorded). The reference softmaxes attention in float32 even in a float64 model; it is
patched to float64 here, as the C# side computes it.

For two images the fixture records every hidden state (the embeddings after the first LayerNorm, then each layer's
output) and the loss sum over states of the mean square, and that loss after one plain SGD step (lr 0.01).
Default: verify; --write: regenerate tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/clip_vision_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/NeuralNetworks/ReferenceData/clip_vision_reference.json"


def patch_float64():
    import torch
    import transformers.models.clip.modeling_clip as m

    def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = torch.nn.functional.softmax(weights, dim=-1)
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights

    m.eager_attention_forward = eager_attention


def compute():
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel
    torch.set_default_dtype(torch.float64)
    patch_float64()
    torch.manual_seed(13)
    config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, image_size=28,
                              patch_size=7, hidden_act="quick_gelu", attn_implementation="eager")
    model = CLIPVisionModel(config).double().eval()
    with torch.no_grad():  # unit norms and zero biases would hide wiring mistakes
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.copy_(0.05 * torch.randn_like(p))
            elif "norm" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items() if not k.endswith("position_ids")}
    images = torch.randn(2, 3, 28, 28)

    def run():
        states = model(pixel_values=images, output_hidden_states=True).hidden_states
        return states, sum((s ** 2).mean() for s in states)

    states, loss = run()
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        _, after = run()
    return {"state": state, "images": images.flatten().tolist(), "states": [s.detach().flatten().tolist() for s in states],
            "loss": loss.item(), "loss_after_sgd": after.item()}


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
