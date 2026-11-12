"""Reference fixture for MaskGCT's semantic features: the release's extract_features (SeamlessM4TFeatureExtractor with
its defaults: padding to an even frame count, an attention mask over the stacked rows) and Wav2Vec2BertModel run with that
mask (transformers 4.41.2, the release's pin), in float64 with closed-form weights on a small w2v-BERT. One waveform has
an odd frame count, so its last row is padding: zeroed after the projection, hidden from attention and from the
depthwise convolution, and still read out.

Run with ~/source/repos/.venv-maskgct (transformers 4.41.2).
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_semantic_reference.json.
"""
import functools
import json
import math
import os

import numpy as np
import torch

import transformers.models.seamless_m4t.feature_extraction_seamless_m4t as seamless_extraction
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel

HERE = os.path.dirname(os.path.abspath(__file__))
SEMANTIC = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, intermediate_size=64,
                feature_projection_input_dim=160, conv_depthwise_kernel_size=31, position_embeddings_type="relative_key",
                left_max_position_embeddings=64, right_max_position_embeddings=8, hidden_act="swish", layer_norm_eps=1e-5,
                add_adapter=False, apply_spec_augment=False, mask_time_prob=0.0, layerdrop=0.0, hidden_dropout=0.0,
                attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, conformer_conv_dropout=0.0)
LAYER = 3


def closed_form(name, shape):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    i = k.to(torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(("norm.weight", "layer_norm.weight")):
        return 1 + 0.2 * s
    if len(shape) >= 2:
        return 0.6 * s / math.sqrt(count / shape[0])
    return 0.05 * s


def features64(extractor, waveform):
    # SeamlessM4TFeatureExtractor.__call__ for one waveform with its defaults (padding=True, pad_to_multiple_of=2),
    # without the float32 casts.
    spectrogram = seamless_extraction.spectrogram
    seamless_extraction.spectrogram = functools.partial(spectrogram, dtype=np.float64)
    try:
        x = extractor._extract_fbank_features(np.asarray(waveform, dtype=np.float64))
    finally:
        seamless_extraction.spectrogram = spectrogram
    x = (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
    frames = x.shape[0]
    mask = np.ones(frames + frames % 2, dtype=np.int64)
    if frames % 2:
        x = np.concatenate([x, np.full((1, x.shape[1]), extractor.padding_value)], 0)
        mask[-1] = 0
    rows = x.shape[0] // 2
    features = x.reshape(rows, -1)
    row_mask = mask[np.arange(x.shape[0]) % 2 == 1]
    return features, row_mask


def main():
    torch.manual_seed(0)
    config = Wav2Vec2BertConfig(**SEMANTIC)
    config._attn_implementation = "eager"
    model = Wav2Vec2BertModel(config).double().eval()
    weights = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = closed_form(name, list(parameter.shape)).reshape(parameter.shape)
            parameter.copy_(value)
            weights[name] = {"shape": list(parameter.shape), "values": value.flatten().tolist()}
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(os.path.join(HERE, "..", "..", "..", "refsrc", "w2vbert2-hf")) \
        if os.path.isdir(os.path.join(HERE, "..", "..", "..", "refsrc", "w2vbert2-hf")) else SeamlessM4TFeatureExtractor()
    cases = []
    for n in (4000, 4160):
        t = np.arange(n, dtype=np.float64)
        # Broadband (a chirp and a deterministic noise over two tones), so no filterbank bin is near-constant over time:
        # per-bin normalization divides by each bin's deviation, and a narrowband signal's would amplify last-digit
        # differences in the filterbank a thousandfold.
        noise = ((np.arange(n) * 7919) % 1009) / 504.5 - 1.0
        audio = 0.3 * np.sin(0.031 * t) + 0.15 * np.sin(0.17 * t + 0.3) + 0.2 * np.sin(1.5e-4 * t * t) + 0.1 * noise
        features, mask = features64(extractor, audio)
        # The real extractor (float32) must agree on the shapes and the mask.
        real = extractor(audio, sampling_rate=16000, return_tensors="pt")
        assert list(real["input_features"].shape[1:]) == list(features.shape), (real["input_features"].shape, features.shape)
        assert real["attention_mask"][0].tolist() == mask.tolist()
        with torch.no_grad():
            out = model(input_features=torch.from_numpy(features).unsqueeze(0), attention_mask=torch.from_numpy(mask).unsqueeze(0),
                        output_hidden_states=True)
        cases.append({"samples": n, "audio": audio.tolist(), "mask": mask.tolist(), "features": features.flatten().tolist(),
                      "rows": features.shape[0], "hidden": out.hidden_states[LAYER][0].flatten().tolist()})
        print(n, features.shape, int(mask.sum()))
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "maskgct_semantic_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"config": SEMANTIC, "layer": LAYER, "weights": weights, "cases": cases}, handle)
        handle.write("\n")


if __name__ == "__main__":
    main()
