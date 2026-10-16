"""LAION-CLAP reference: the HTSAT + RoBERTa CLAP of LAION-AI/CLAP (Wu et al., ICASSP 2023) at a tiny size.

The network is the reference's own code: clap_module.model.CLAP with its HTSAT_Swin_Transformer audio branch (built
through a patched create_htsat_model so the size can be tiny), the AFF feature-fusion patch embedding, and a tiny
Hugging Face RobertaModel text branch (RobertaModel.from_pretrained is patched to build one from a config instead of
downloading roberta-base). The reference's text_projection hardcodes roberta-base's 768-wide input, so it is rebuilt
with the same Linear -> ReLU -> Linear structure at the tiny width. The audio input pipeline is
training/data.py's get_mel and get_audio_features, executed from the reference source.

The fixture records:
- the weights (CLAP state dict, float32 values), and the sizes;
- a short waveform (shorter than the clip): its non-fusion log-mel (torchlibrosa, librosa's Slaney filterbank) and
  fusion log-mel (torchaudio, HTK filterbank) as the model reads them, and the audio embeddings of the non-fusion and
  the fusion model;
- a long waveform (longer than the clip): its four fusion views with the crop offsets the reference drew, and the
  fusion model's embedding of those views;
- two padded captions' token ids and their text embeddings, and the logit matrix logit_scale_a.exp() * a @ t.T.

Everything runs in float64 (torch.set_default_dtype) from float32-rounded weights. Two front-end tensors the reference
builds in float32 regardless are rebuilt in float64 from the same formulas: torchlibrosa's melW (librosa.filters.mel
defaults to dtype=float32) and its STFT convolution kernels (built with the legacy float32 torch.Tensor constructor).
Upstream data.py calls np.random.integers, which the legacy numpy module does not have; it only draws
the waveform crop a fusion model never reads, and is aliased to randint.

Pass a LAION-AI/CLAP checkout with --clap. Default: verify; --write: regenerate the fixture.
"""
import argparse
import ast
import importlib
import json
import os
import sys
import types

import numpy as np
import torch

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/clap_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

AUDIO = dict(sample_rate=48000, window_size=512, hop_size=64, mel_bins=16, fmin=50, fmax=14000, clip_samples=8192,
             audio_length=1024, class_num=5, model_type="HTSAT", model_name="tiny")
HTSAT = dict(spec_size=64, patch_size=4, patch_stride=(4, 4), embed_dim=8, depths=[2, 2, 2], num_heads=[1, 2, 4],
             window_size=8)
ROBERTA = dict(vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
               max_position_embeddings=32, type_vocab_size=1, pad_token_id=1, bos_token_id=0, eos_token_id=2,
               layer_norm_eps=1e-5, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
JOINT = 12
CAPTIONS = [[0, 5, 9, 17, 23, 2, 1, 1, 1, 1], [0, 33, 41, 2, 1, 1, 1, 1, 1, 1]]
SHORT, LONG = 5000, 30000


def waveform(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / AUDIO["sample_rate"]
    return 0.4 * np.sin(2 * np.pi * 880.0 * t) + 0.2 * np.sin(2 * np.pi * 3100.0 * t) + 0.05 * rng.standard_normal(n)


def load_reference(clap):
    src = os.path.join(clap, "src", "laion_clap")
    for name in ("h5py",):  # imported by clap_module.utils, unused here; timm is optional (guarded import)
        sys.modules.setdefault(name, types.ModuleType(name))
    package = types.ModuleType("laion_clap.clap_module")
    package.__path__ = [os.path.join(src, "clap_module")]
    root = types.ModuleType("laion_clap")
    root.__path__ = [src]
    sys.modules["laion_clap"] = root
    sys.modules["laion_clap.clap_module"] = package
    model = importlib.import_module("laion_clap.clap_module.model")
    htsat = importlib.import_module("laion_clap.clap_module.htsat")

    # training/data.py's audio pipeline, executed from the reference source (its module imports pull in the whole
    # training stack, so only the two functions are compiled).
    import torchaudio
    import torchvision
    path = os.path.join(src, "training", "data.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("get_mel", "get_audio_features")]
    from contextlib import suppress
    import torch.nn.functional as F
    namespace = dict(torch=torch, torchaudio=torchaudio, torchvision=torchvision, np=np, F=F, suppress=suppress)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), namespace)
    return model, htsat, namespace


def build(model, htsat, fusion):
    from transformers import RobertaConfig, RobertaModel

    def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type="None"):
        return htsat.HTSAT_Swin_Transformer(num_classes=audio_cfg.class_num, config=audio_cfg,
                                            enable_fusion=enable_fusion, fusion_type=fusion_type, **HTSAT)

    model.create_htsat_model = create_htsat_model
    model.RobertaModel = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: RobertaModel(RobertaConfig(**ROBERTA)))
    width = HTSAT["embed_dim"] * 2 ** (len(HTSAT["depths"]) - 1)
    text_cfg = dict(context_length=len(CAPTIONS[0]), vocab_size=ROBERTA["vocab_size"], width=ROBERTA["hidden_size"],
                    heads=ROBERTA["num_attention_heads"], layers=ROBERTA["num_hidden_layers"], model_type="roberta")
    clap = model.CLAP(embed_dim=width, audio_cfg=AUDIO, text_cfg=text_cfg, enable_fusion=fusion,
                      fusion_type="aff_2d" if fusion else "None", joint_embed_shape=JOINT)
    clap.text_projection = torch.nn.Sequential(torch.nn.Linear(ROBERTA["hidden_size"], JOINT), torch.nn.ReLU(),
                                               torch.nn.Linear(JOINT, JOINT))
    return clap


def float64_slaney(clap):
    import librosa
    a = AUDIO
    melW = librosa.filters.mel(sr=a["sample_rate"], n_fft=a["window_size"], n_mels=a["mel_bins"], fmin=a["fmin"],
                               fmax=a["fmax"], dtype=np.float64).T
    clap.audio_branch.logmel_extractor.melW.data = torch.tensor(melW, dtype=torch.float64)
    # torchlibrosa builds its STFT kernels with the legacy torch.Tensor(ndarray) constructor, which yields float32
    # whatever the default dtype is; rebuild them in float64 from the same DFT matrix and window.
    import librosa.util
    stft = clap.audio_branch.spectrogram_extractor.stft
    window = librosa.util.pad_center(librosa.filters.get_window("hann", a["window_size"], fftbins=True), size=a["window_size"])
    bins = a["window_size"] // 2 + 1
    kernel = stft.W[:, 0:bins] * window[:, None]
    stft.conv_real.weight.data = torch.tensor(np.real(kernel).T, dtype=torch.float64)[:, None, :]
    stft.conv_imag.weight.data = torch.tensor(np.imag(kernel).T, dtype=torch.float64)[:, None, :]


def compute(clap_dir):
    torch.set_default_dtype(torch.float64)
    model, htsat, data = load_reference(clap_dir)
    torch.manual_seed(2313)
    fused = build(model, htsat, fusion=True)
    with torch.no_grad():
        for p in fused.parameters():
            p.copy_(p.float().double())
        # Non-trivial running statistics, so loading them is exercised.
        g = torch.Generator().manual_seed(5)
        for m in fused.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.running_mean.copy_((0.5 * torch.randn(m.running_mean.shape, generator=g)).float().double())
                m.running_var.copy_((0.5 + torch.rand(m.running_var.shape, generator=g)).float().double())
                m.weight.copy_((1 + 0.1 * torch.randn(m.weight.shape, generator=g)).float().double())
                m.bias.copy_((0.1 * torch.randn(m.bias.shape, generator=g)).float().double())
        fused.logit_scale_a.copy_(torch.tensor(np.log(1 / 0.07)).float().double())
    fused.eval()
    plain = build(model, htsat, fusion=False)
    missing, unexpected = plain.load_state_dict(fused.state_dict(), strict=False)
    assert not missing, missing
    assert all(".patch_embed.mel_conv2d" in k or ".patch_embed.fusion_model" in k for k in unexpected), unexpected
    float64_slaney(plain)
    plain.eval()

    cfg = dict(AUDIO)
    short = torch.tensor(waveform(SHORT, 1))
    long = torch.tensor(waveform(LONG, 2))
    with torch.no_grad():
        # Non-fusion: rand_trunc / repeatpad, then HTSAT's own torchlibrosa front end.
        s = data["get_audio_features"]({}, short.clone(), AUDIO["clip_samples"], "rand_trunc", "repeatpad", cfg)
        x = s["waveform"][None]
        plain_mel = plain.audio_branch.logmel_extractor(plain.audio_branch.spectrogram_extractor(x))   # [1, 1, T, F]
        plain_audio = torch.nn.functional.normalize(
            plain.audio_projection(plain.encode_audio({"waveform": x}, device="cpu")["embedding"]), dim=-1)

        # Fusion, short clip: four identical views, not longer.
        s = data["get_audio_features"]({}, short.clone(), AUDIO["clip_samples"], "fusion", "repeatpad", cfg)
        short_views = s["mel_fusion"][None]
        fused_short = torch.nn.functional.normalize(fused.audio_projection(fused.encode_audio(
            {"mel_fusion": short_views, "longer": s["longer"]}, device="cpu")["embedding"]), dim=-1)

        # Fusion, long clip: record the crop offsets np.random.choice draws.
        drawn = []
        choice = np.random.choice

        def recording(a, *args, **kwargs):
            value = choice(a, *args, **kwargs)
            drawn.append(int(value))
            return value

        np.random.seed(7)
        np.random.choice = recording
        np.random.integers = np.random.randint
        try:
            s = data["get_audio_features"]({}, long.clone(), AUDIO["clip_samples"], "fusion", "repeatpad", cfg)
        finally:
            np.random.choice = choice
            del np.random.integers
        long_views = s["mel_fusion"][None]
        long_mel = data["get_mel"](long.clone(), cfg)
        fused_long = torch.nn.functional.normalize(fused.audio_projection(fused.encode_audio(
            {"mel_fusion": long_views, "longer": s["longer"]}, device="cpu")["embedding"]), dim=-1)

        ids = torch.tensor(CAPTIONS)
        text = fused.get_text_embedding({"input_ids": ids, "attention_mask": (ids != ROBERTA["pad_token_id"]).long()})
        logits = fused.logit_scale_a.exp() * fused_short @ text.T

    # Buffers that are derived, not learned, and modules the CLAP embedding never reads (HTS-AT's AudioSet head,
    # the MLP-loss transforms and their text-side logit scale).
    skip = ("relative_position_index", "attn_mask", "num_batches_tracked", "spectrogram_extractor",
            "logmel_extractor", "position_ids", "tscam_conv", "audio_branch.head", "audio_transform",
            "text_transform", "logit_scale_t")
    state = {k: v.float().flatten().tolist() for k, v in fused.state_dict().items() if not any(t in k for t in skip)}
    return {
        "config": {"audio": AUDIO, "htsat": {k: list(v) if isinstance(v, tuple) else v for k, v in HTSAT.items()},
                   "roberta": ROBERTA, "joint": JOINT},
        "state_dict": state,
        "shapes": {k: list(v.shape) for k, v in fused.state_dict().items() if k in state},
        "short_waveform": short.tolist(),
        "long_waveform": long.tolist(),
        "plain_mel": plain_mel[0, 0].tolist(),
        "fusion_short_mel": short_views[0, 0].tolist(),
        "plain_audio_embedding": plain_audio[0].tolist(),
        "fusion_short_audio_embedding": fused_short[0].tolist(),
        "long_mel": long_mel.tolist(),
        "long_views": long_views[0].tolist(),
        "long_crop_offsets": drawn,
        "fusion_long_audio_embedding": fused_long[0].tolist(),
        "captions": CAPTIONS,
        "text_embeddings": text.tolist(),
        "logits": logits.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--clap", default=os.path.expanduser("~/source/repos/ref/CLAP"))
    args = parser.parse_args()
    result = compute(args.clap)
    path = os.path.join(REPO, *FIXTURE.split("/"))
    if args.write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        result["generator"] = "tools/reference-data/clap_reference.py"
        result["torch"] = torch.__version__
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print(f"wrote {path}: logits {result['logits']}")
        return
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    # Every generated key, not only the final outputs: the weights, the mels and the fusion views are what the C#
    # tests read, so drift in any of them must fail here too. Only the provenance written by --write is skipped.
    provenance = ("generator", "torch")
    missing = sorted(set(result) - set(stored))
    extra = sorted(set(stored) - set(result) - set(provenance))
    mismatched = [k for k in sorted(result) if k in stored and not same_value(stored[k], result[k])]
    for k in missing:
        print(f"missing from fixture: {k}")
    for k in extra:
        print(f"no longer generated: {k}")
    for k in mismatched:
        print(f"differs: {k}")
    ok = not (missing or extra or mismatched)
    print("match" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


def same_value(stored, generated):
    """Exact for structure and text, within 1e-12 for numbers, recursing through dicts (the state dict, the shapes)."""
    if isinstance(generated, dict):
        return (isinstance(stored, dict) and set(stored) == set(generated)
                and all(same_value(stored[k], generated[k]) for k in generated))
    if isinstance(generated, str):
        return stored == generated
    stored_array, generated_array = np.asarray(stored), np.asarray(generated)
    if stored_array.shape != generated_array.shape:
        return False
    if generated_array.dtype.kind in "biuf":
        return bool(np.allclose(stored_array, generated_array, rtol=0, atol=1e-12))
    return stored == generated


if __name__ == "__main__":
    main()
