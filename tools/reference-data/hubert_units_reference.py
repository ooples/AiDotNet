"""mHuBERT units reference: layer-11 features of fairseq's mhubert_base_vp_en_es_fr_it3 and its 1000-cluster k-means
(mhubert_base_vp_en_es_fr_it3_L11_km1000.bin), as fairseq's HubertFeatureReader and ApplyKmeans label them.

fairseq is not importable here, so the checkpoint runs in transformers' HubertModel with fairseq's names mapped as
transformers' convert_hubert_original_pytorch_checkpoint_to_pytorch maps them (the fairseq config objects in the pickle are
stubbed). Records in tests/AiDotNet.Tests/Audio/ReferenceData/hubert_units_reference.json, in float64: three seconds of
CosyVoice's zero_shot_prompt.wav at 16 kHz, its layer-11 features, the k-means labels (scikit-learn's predict on the
float32 features, as the release feeds it) and those labels with consecutive repeats merged; the first centroids as
joblib reads them; and a five-cluster k-means saved by the installed joblib, with its centroids.

Usage (default: verify; --write: regenerate):
  python tools/reference-data/hubert_units_reference.py --hubert <mhubert .pt> --kmeans <km1000 .bin> --wav <wav> [--write]
"""
import argparse
import os
import pickle
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/hubert_units_reference.json"


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("fairseq", "omegaconf")):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)


def load_fairseq_state(path):
    module = types.ModuleType("stub_pickle")
    module.Unpickler = _Unpickler
    module.load = pickle.load
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=module)["model"]


def to_transformers(state):
    mapped = {}
    for k, v in state.items():
        if k in ("mask_emb", "label_embs_concat") or k.startswith("final_proj"):
            continue
        n = k
        if n.startswith("feature_extractor.conv_layers."):
            parts = n.split(".")
            n = f"feature_extractor.conv_layers.{parts[2]}." + ("conv." if parts[3] == "0" else "layer_norm.") + parts[4]
        elif n.startswith("layer_norm."):
            n = "feature_projection." + n
        elif n.startswith("post_extract_proj."):
            n = n.replace("post_extract_proj.", "feature_projection.projection.")
        elif n.startswith("encoder.pos_conv.0."):
            n = n.replace("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv.")
        elif n.startswith("encoder.layers."):
            n = (n.replace(".self_attn_layer_norm.", ".layer_norm.").replace(".self_attn.", ".attention.")
                 .replace(".fc1.", ".feed_forward.intermediate_dense.").replace(".fc2.", ".feed_forward.output_dense."))
        mapped[n] = v
    return mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hubert", required=True)
    parser.add_argument("--kmeans", required=True)
    parser.add_argument("--wav")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import joblib
    from transformers import HubertConfig, HubertModel
    stored = None if args.write else _fixture.load(FIXTURE)
    if stored:
        audio = np.array(stored["audio"])
    else:
        import soundfile
        wave, rate = soundfile.read(args.wav)
        if wave.ndim == 2:
            wave = wave.mean(-1)
        if rate != 16000:
            import torchaudio
            wave = torchaudio.transforms.Resample(rate, 16000)(torch.tensor(wave, dtype=torch.float32)[None])[0].numpy()
        audio = wave[:48000].astype(np.float32).astype(np.float64)

    config = HubertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, feat_extract_norm="group",
                          do_stable_layer_norm=False, conv_bias=False, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
                          hidden_act="gelu", feat_proj_layer_norm=True)
    model = HubertModel(config)
    missing, unexpected = model.load_state_dict(to_transformers(load_fairseq_state(args.hubert)), strict=False)
    assert not unexpected and all("masked_spec_embed" in m for m in missing), (missing, unexpected)
    model = model.double().eval()
    with torch.no_grad():
        hidden = model(torch.tensor(audio)[None], output_hidden_states=True).hidden_states[11][0]
    kmeans = joblib.load(args.kmeans)
    labels = kmeans.predict(hidden.float().numpy()).tolist()
    merged = [u for i, u in enumerate(labels) if i == 0 or u != labels[i - 1]]
    data = {"features": [float(v) for v in hidden.reshape(-1).tolist()], "labels": labels, "merged": merged,
            "centroids_head": [float(v) for v in kmeans.cluster_centers_[:3].reshape(-1).tolist()]}
    if args.write:
        # A small k-means saved by this joblib (which aligns its arrays, unlike the 2021 release file) for the reader.
        import base64
        import io
        from sklearn.cluster import KMeans
        tiny = KMeans(n_clusters=5, n_init=1, random_state=0).fit(np.random.default_rng(3).normal(size=(40, 3)).astype(np.float32))
        buffer = io.BytesIO()
        joblib.dump(tiny, buffer)
        data.update(tiny_joblib_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                    tiny_centers=[float(v) for v in tiny.cluster_centers_.reshape(-1).tolist()])
        _fixture.save(FIXTURE, {"audio": audio.tolist(), **data}, __file__)
        print("wrote", FIXTURE, len(labels), "frames", merged[:12])
        return
    check = _fixture.Comparison("mHuBERT units")
    check.floats("features", stored["features"], data["features"], 1e-12, float)
    check.exact("labels", stored["labels"], labels)
    check.report(exact=False)


if __name__ == "__main__":
    main()
