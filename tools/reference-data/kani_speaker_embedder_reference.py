"""KaniTTS 2's speaker embedder (kani_tts/speaker_embedder.py's EmbeddingsModel) on a tiny WavLM in WavLM-Large's layout.

transformers 4.57.1's WavLMModel with feat_extract_norm="layer" and do_stable_layer_norm=True (a layer norm after every
feature-encoder convolution, pre-norm encoder layers, the encoder's layer norm last), at width 16 with two layers; the
head (1×1 convolution, ReLU, non-affine batch norm, twice, then L2 normalization) with its first convolution resized from
the release's fixed 2048 inputs to this width's 32 (a test-size patch; nothing else changes). Random weights, random
running statistics, evaluation mode, float64. Records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kani_speaker_embedder_reference.json the WavLM's last hidden state and
the embedding of 4000 samples. Pass --kani2 (nineninesix-ai/kani-tts-2 checkout). Default: verify; --write: regenerate.
Needs the .venv-kani environment.
"""
import argparse
import importlib.util
import os
import sys

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kani_speaker_embedder_reference.json"
WAVLM = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, conv_dim=[8] * 7,
             conv_kernel=[10, 3, 3, 3, 3, 2, 2], conv_stride=[5, 2, 2, 2, 2, 2, 2], conv_bias=False, feat_extract_norm="layer",
             do_stable_layer_norm=True, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=320,
             max_bucket_distance=800, embd_size=4, top_interm_size=6)
SAMPLES = 4000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kani2", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("kani_speaker_embedder", os.path.join(os.path.abspath(args.kani2), "kani_tts/speaker_embedder.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transformers import WavLMConfig

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(83)
    model = module.EmbeddingsModel(WavLMConfig(**WAVLM))
    # Test-size patch: the release's head reads 2048 pooled values (WavLM-Large's 1024, mean and standard deviation).
    model.top_layers.affine1 = torch.nn.Conv1d(2 * WAVLM["hidden_size"], WAVLM["top_interm_size"], kernel_size=1)
    with torch.no_grad():
        state = model.state_dict()
        for name, value in state.items():
            if name.endswith("num_batches_tracked"):
                continue
            if stored is not None:
                value.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(value.shape))
            elif name.endswith("running_var"):
                value.copy_(0.5 + torch.rand_like(value))
            elif name.endswith("running_mean"):
                value.copy_(0.1 * torch.randn_like(value))
            elif "layer_norm.weight" in name or name.endswith("gru_rel_pos_const"):
                value.copy_(0.5 + torch.rand_like(value))
            elif value.is_floating_point():
                value.copy_((0.3 * torch.randn_like(value)).float())
        state32 = {k: v.float().clone() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    model = model.double().eval()
    torch.manual_seed(89)
    audio = (0.4 * torch.randn(1, SAMPLES)).float().double() if args.write else torch.tensor([stored["audio"]], dtype=torch.float64)
    with torch.no_grad():
        x_norm = (audio - audio.mean(dim=1, keepdim=True)) / (audio.std(dim=1, keepdim=True) + 1e-10)
        hidden = model.wavlm(input_values=x_norm).last_hidden_state[0]
        embedding = model(audio)[0]

    data = {"wavlm": WAVLM, "audio": audio[0].tolist(), "hidden": hidden.reshape(-1).tolist(), "embedding": embedding.tolist()}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "hidden", tuple(hidden.shape), "embedding", embedding.tolist())
        return
    check = _fixture.Comparison("Kani speaker embedder")
    check.floats("hidden", stored["hidden"], data["hidden"], 1e-12, float)
    check.floats("embedding", stored["embedding"], data["embedding"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
