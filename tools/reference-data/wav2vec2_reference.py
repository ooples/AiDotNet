"""wav2vec 2.0 reference: transformers' Wav2Vec2Model (the port of fairseq's model).

Two small configurations run in float64 in eval mode: BASE-style (group-norm feature encoder, post-norm encoder) and
LARGE-LV60-style (layer-norm feature encoder, stable pre-norm encoder), each with a 3-layer feature encoder (dims 8,
kernels 10/3/2, strides 5/2/2, conv bias on in the second), hidden size 16, 2 layers of 2 heads, intermediate 32, a
positional convolution of kernel 8 in 4 groups. For two 400-sample waveforms the fixture records the state dict, the
feature-encoder output after its layer norm (extract_features), every hidden state (after the position embedding,
then after each layer), the last hidden state, the loss mean(last_hidden_state^2), and that loss after one plain SGD
step (lr 0.1) on every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/Audio/ReferenceData/wav2vec2_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/wav2vec2_reference.json"


def compute():
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    torch.set_default_dtype(torch.float64)

    def case(name, layer_norm, seed):
        config = Wav2Vec2Config(conv_dim=(8, 8, 8), conv_kernel=(10, 3, 2), conv_stride=(5, 2, 2), conv_bias=layer_norm,
                                feat_extract_norm="layer" if layer_norm else "group", do_stable_layer_norm=layer_norm,
                                hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                                num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=4, hidden_act="gelu",
                                feat_extract_activation="gelu", layer_norm_eps=1e-5)
        torch.manual_seed(seed)
        model = Wav2Vec2Model(config).double().eval()
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.float())
        g = torch.Generator().manual_seed(seed + 1)
        audio = torch.randn(2, 400, generator=g)
        state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
        out = model(audio, output_hidden_states=True)
        loss = (out.last_hidden_state ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.1 * p.grad
            after = (model(audio).last_hidden_state ** 2).mean()
        return {"name": name, "layer_norm": layer_norm, "state": state, "audio": audio.flatten().tolist(),
                "extract_features": out.extract_features.detach().flatten().tolist(),
                "hidden_states": [h.detach().flatten().tolist() for h in out.hidden_states],
                "last_hidden_state": out.last_hidden_state.detach().flatten().tolist(),
                "frames": out.last_hidden_state.shape[1], "loss": loss.item(), "loss_after_sgd": after.item()}

    return {"cases": [case("base", False, 3), case("large", True, 5)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", [(c["name"], c["frames"], round(c["loss"], 9), round(c["loss_after_sgd"], 9)) for c in result["cases"]])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a["loss"] - b["loss"]) < 1e-12 and abs(a["loss_after_sgd"] - b["loss_after_sgd"]) < 1e-12
               for a, b in zip(stored["cases"], result["cases"]))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
