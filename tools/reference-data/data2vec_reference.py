"""data2vec-audio reference: transformers' Data2VecAudioForCTC in the released BASE form.

The facebook/data2vec-audio-base-960h form at test width: the full 7-layer, 512-channel feature encoder with layer
norm and no conv bias, a post-norm encoder of hidden size 16, 2 layers of 2 heads, intermediate 32, the
128-tap positional convolution in 16 groups, and a 12-token CTC head, in float64 eval mode.

The feature encoder alone holds ~4M weights, so the state dict is not stored: every parameter is filled from a
formula of its name and index that the C# test reproduces (see Fill). The fixture records two 1600-sample
waveforms and the CTC logits.

Default: verify; --write: regenerate tests/AiDotNet.Tests/Audio/ReferenceData/data2vec_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/data2vec_reference.json"


def fill(name, count):
    """0.05 sin(0.7 (i + 1) + 0.013 s), s the sum of the name's character codes; norm gains and the weight-norm
    magnitude are offset by 1 so the signal survives the stack."""
    s = sum(ord(c) for c in name)
    offset = 1.0 if name.endswith("norm.weight") or name.endswith("original0") else 0.0
    return [offset + 0.05 * math.sin(0.7 * (i + 1) + 0.013 * s) for i in range(count)]


def compute():
    import torch
    from transformers import Data2VecAudioConfig, Data2VecAudioForCTC
    torch.set_default_dtype(torch.float64)
    config = Data2VecAudioConfig(conv_bias=False,
                            hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                            num_conv_pos_embeddings=5, conv_pos_kernel_size=19, num_conv_pos_embedding_groups=16, vocab_size=12,
                            hidden_act="gelu", feat_extract_activation="gelu", layer_norm_eps=1e-5)
    model = Data2VecAudioForCTC(config).double().eval()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            tensor.copy_(torch.tensor(fill(name, tensor.numel()), dtype=torch.float64).reshape(tensor.shape))
    g = torch.Generator().manual_seed(37)
    audio = torch.randn(2, 1600, generator=g)
    with torch.no_grad():
        logits = model(audio).logits
    return {"audio": audio.flatten().tolist(), "frames": logits.shape[1], "vocab": logits.shape[2],
            "logits": logits.flatten().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w") as f:
            json.dump(result, f)
        print("wrote", FIXTURE)
        return 0
    with open(FIXTURE) as f:
        stored = json.load(f)
    worst = max(abs(a - b) for a, b in zip(stored["logits"], result["logits"]))
    print("max |stored - recomputed| =", worst)
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
