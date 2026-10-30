"""SpiritLM speech tokenizer and vocoder reference (facebookresearch/spiritlm's spiritlm/speech_tokenizer, torchfcpe 0.0.4
and AMFM_decompy's pYAAPT), run in float64 on tiny seeded modules and, with --dir, on the released checkpoints.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spiritlm_speech_reference.json:
  - the 25 Hz mHuBERT (the release's vendored fairseq HubertModel: batch-normed position convolution, 640-sample hop) at a
    tiny size, its layer features and the CTC linear quantizer's units (LinearQuantizerModel), with crafted logits for the
    blank filling (a blank first frame takes the last frame's unit, as torch.roll wraps);
  - FCPE (CFNaiveMelPE, conv-only) at a tiny size: the log-mel, the saliences, the F0 and SpiritLM's interpolated F0 and
    voicing at 5 ms (FCPEF0Extractor's arguments);
  - pYAAPT through the release's pYAAPTF0Extractor (normalized to 0.95, padded, interpolated) on a LibriSpeech excerpt;
  - the F0 VQ-VAE at a tiny size: codes of a normalized F0 and voicing track;
  - the CodeGenerator vocoders at tiny sizes: Base-like (speaker and utterance style) and Expressive-like (pitch and style
    tokens), with duration prediction, and the release's upsample of lower-rate tokens;
  - the wav2vec 2.0 style classifier at a tiny size (Wav2Vec2ForSequenceClassification, the release's per-50-frame pooling);
  - units_to_string and string_to_units cases.
With --dir (the released speech_tokenizer directory) it also records the real 25 Hz mHuBERT features and units, the
bundled FCPE's F0, the real F0 VQ-VAE codes, both real vocoders' waveforms and the real style tokens.

Float32 steps the release takes are kept in float64: fairseq's Fp32GroupNorm and Fp32LayerNorm (the feature encoder's
norms, which cast to float32) and its encoder layers' gelu (F.gelu(x.float())), librosa's mel bank (float32 by default) and FCPE's float32 cast of the waveform;
Wav2Vec2FeatureExtractor's float32 normalization is recomputed in float64. Default: verify; --write:
regenerate. Run with .venv-ttsref312 (torch 2.14, transformers 5.18, torchfcpe 0.0.4 and AMFM-decompy installed no-deps).
"""
import argparse
import base64
import json
import os
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spiritlm_speech_reference.json"


def flat(t):
    return [float(v) for v in torch.as_tensor(t).detach().reshape(-1).tolist()]


def pack(module):
    state = {k: v.detach().contiguous().to(torch.float64 if v.is_floating_point() else v.dtype) for k, v in module.state_dict().items()}
    return base64.b64encode(save_safetensors(state)).decode("ascii")


def unpack(module, text):
    module.load_state_dict(load_safetensors(base64.b64decode(text)), strict=True)


def tones(seconds, rate=16000, seed=3):
    g = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f = 140 + 60 * np.sin(2 * np.pi * 1.3 * t)
    phase = 2 * np.pi * np.cumsum(f) / rate
    return (0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase + 0.3) + 0.02 * g.standard_normal(t.shape)).astype(np.float64)


def librispeech(path, seconds):
    import soundfile
    wave, rate = soundfile.read(path)
    assert rate == 16000
    return wave[: int(seconds * rate)].astype(np.float32).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiritlm", required=True, help="the facebookresearch/spiritlm checkout")
    parser.add_argument("--dir", help="the released speech_tokenizer directory (real weights)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.spiritlm)
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    data = {}
    saved = {}

    # ------------------------------------------------------------------ 25 Hz HuBERT and the linear quantizer
    from spiritlm.speech_tokenizer.hubert.hubert_model.hubert_model import HubertConfig, HubertModel, HubertPretrainingConfig
    from spiritlm.speech_tokenizer.hubert.quantizer_model import LinearQuantizerModel
    from spiritlm.speech_tokenizer.hubert.hubert_model import wav2vec2_model
    wav2vec2_model.Fp32GroupNorm.forward = torch.nn.GroupNorm.forward
    wav2vec2_model.Fp32LayerNorm.forward = torch.nn.LayerNorm.forward

    def double_gelu(model):
        for layer in model.encoder.layers:
            layer.activation_fn = torch.nn.functional.gelu
        return model

    hubert_cfg = dict(conv_feature_layers="[(12,10,5)] + [(12,3,2)] * 2 + [(12,2,2)]", encoder_embed_dim=16, encoder_ffn_embed_dim=32,
                      encoder_attention_heads=2, encoder_layers=3, extractor_mode="default", conv_pos=16, conv_pos_groups=4,
                      conv_pos_batch_norm=True, layer_norm_first=False, conv_bias=False, dropout=0.0, attention_dropout=0.0,
                      activation_dropout=0.0, encoder_layerdrop=0.0, dropout_input=0.0, dropout_features=0.0, required_seq_len_multiple=2)
    torch.manual_seed(2294)
    cfg = HubertConfig(label_rate=50.0)
    for k, v in hubert_cfg.items():
        setattr(cfg, k, v)
    hubert = HubertModel(cfg, HubertPretrainingConfig(), dictionaries=None)
    with torch.no_grad():
        bn = hubert.encoder.pos_conv[0]
        bn.running_mean.copy_(torch.randn(16) * 0.3)
        bn.running_var.copy_(torch.rand(16) + 0.5)
        bn.weight.copy_(torch.rand(16) + 0.5)
        bn.bias.copy_(torch.randn(16) * 0.1)
    if stored:
        unpack(hubert, stored["hubert_state"])
    hubert = double_gelu(hubert.double().eval())
    audio = tones(0.5) if not stored else np.array(stored["hubert_audio"])
    with torch.no_grad():
        features, _ = hubert.extract_features(torch.tensor(audio)[None], padding_mask=None, mask=False, output_layer=2)
    features = features[0]
    # LinearQuantizerModel reads its widths and weights from a checkpoint: write a tiny one (16 → 14 → 12 → 7, 6 units).
    torch.manual_seed(7)
    encoder = torch.nn.Sequential(torch.nn.Linear(16, 14), torch.nn.LeakyReLU(), torch.nn.Linear(14, 12), torch.nn.LeakyReLU(), torch.nn.Linear(12, 7))
    with torch.no_grad():
        encoder[4].bias[6] += 0.4   # blanks now and then
    if stored:
        encoder.load_state_dict(load_safetensors(base64.b64decode(stored["quantizer_state"])))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "q.pt")
        torch.save({"model_cfg": {"vocab_size": 6, "dim": 16, "upstream_dim": 16}, "model_weight": encoder.state_dict()}, path)
        quantizer = LinearQuantizerModel(path).double().eval()
    with torch.no_grad():
        logits = quantizer.encoder(features)
        units = quantizer(features)
    data.update(hubert_features=flat(features), quantizer_logits=flat(logits), units=[int(u) for u in units])
    saved.update(hubert_audio=audio.tolist(), hubert_state=pack(hubert), hubert_config=hubert_cfg,
                 quantizer_state=base64.b64encode(save_safetensors({k: v.contiguous() for k, v in encoder.state_dict().items()})).decode("ascii"))
    identity = LinearQuantizerModel.__new__(LinearQuantizerModel)
    torch.nn.Module.__init__(identity)
    identity.vocab_size = 4
    identity.encoder = torch.nn.Identity()
    blank_cases = [[4, 4, 1, 4, 2, 2], [4, 3, 4, 4, 0], [4, 4, 4, 4]]
    filled = []
    for case in blank_cases:
        crafted = torch.full((len(case), 5), -1.0)
        for i, c in enumerate(case):
            crafted[i, c] = 1.0
        filled.append([int(u) for u in identity(crafted)])
    data["blank_filled"] = filled
    saved["blank_cases"] = blank_cases

    # ------------------------------------------------------------------ FCPE
    import librosa
    from torchfcpe.models_infer import InferCFNaiveMelPE
    from torchfcpe.tools import DotDict
    fcpe_cfg = {"mel": {"sr": 16000, "num_mels": 16, "n_fft": 1024, "win_size": 1024, "hop_size": 160, "fmin": 0, "fmax": 8000, "type": "default"},
                "model": {"type": "CFNaiveMelPE", "out_dims": 30, "hidden_dims": 16, "n_layers": 2, "n_heads": 2, "f0_min": 32.70, "f0_max": 1975.5,
                          "use_fa_norm": True, "conv_only": True, "conv_dropout": 0.0, "atten_dropout": 0.0}}
    from torchfcpe.models_infer import spawn_model
    torch.manual_seed(11)
    fcpe_model = spawn_model(DotDict(fcpe_cfg))
    with torch.no_grad():
        fcpe_model.output_proj.bias.add_(0.5)
    if stored:
        unpack(fcpe_model, stored["fcpe_state"])
    fcpe = InferCFNaiveMelPE(DotDict(fcpe_cfg), fcpe_model.state_dict()).double()

    def double_mel(module, mels):
        basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=mels, fmin=0, fmax=8000, dtype=np.float64)
        module.wav2mel.mel_extractor.mel_basis = torch.tensor(basis)

    double_mel(fcpe, 16)
    fcpe_audio = tones(0.6, seed=5) if not stored else np.array(stored["fcpe_audio"])

    def fcpe_track(model, wave):
        x = torch.tensor(wave)[None, :, None]
        mel = model.wav2mel(x, 16000)
        latent = model.model.forward(mel)
        f0, uv = model.infer(x, sr=16000, decoder_mode="local_argmax", threshold=0.05, f0_min=50, f0_max=1100, interp_uv=True,
                             output_interp_target_length=len(wave) // 80 + 1, retur_uv=True)
        raw = model(x, 16000, "local_argmax", 0.05)
        return mel, latent, raw, f0, 1 - uv

    with torch.no_grad():
        mel, latent, raw, f0, vuv = fcpe_track(fcpe, fcpe_audio)
    data.update(fcpe_mel=flat(mel), fcpe_saliences=flat(latent), fcpe_f0=flat(raw), fcpe_f0_interpolated=flat(f0), fcpe_voiced=flat(vuv))
    saved.update(fcpe_audio=fcpe_audio.tolist(), fcpe_state=pack(fcpe_model), fcpe_config=fcpe_cfg)

    # ------------------------------------------------------------------ pYAAPT
    from spiritlm.speech_tokenizer.f0.f0_extractor import pYAAPTF0Extractor
    speech_path = os.path.join(args.spiritlm, "examples", "audio", "7143-88743-0029.flac")
    yaapt_audio = librispeech(speech_path, 1.2) if not stored else np.array(stored["yaapt_audio"])
    yaapt = pYAAPTF0Extractor(hop_length=80, sampling_rate=16000, interpolate=True)
    yf0, yvuv = yaapt.compute_f0_uv(yaapt_audio, interpolate=True)
    raw_pitch = yaapt.get_pitch(yaapt_audio)
    data.update(yaapt_f0=[float(v) for v in yf0], yaapt_voiced=[float(v) for v in np.asarray(yvuv, dtype=np.float64)],
                yaapt_raw=[float(v) for v in raw_pitch.samp_values], yaapt_energy=[float(v) for v in raw_pitch.energy])
    saved["yaapt_audio"] = yaapt_audio.tolist()

    # ------------------------------------------------------------------ F0 VQ-VAE
    from omegaconf import OmegaConf
    from spiritlm.speech_tokenizer.f0.vqvae import VQVAE
    vq_cfg = OmegaConf.create({"vq_params": {"l_bins": 8, "emb_width": 6, "mu": 0.99, "levels": 1},
                               "encoder_params": {"input_emb_width": 2, "output_emb_width": 6, "levels": 1, "downs_t": [4], "strides_t": [2],
                                                  "width": 5, "depth": 2, "m_conv": 1.0, "dilation_growth_rate": 3},
                               "decoder_params": {"input_emb_width": 2, "output_emb_width": 6, "levels": 1, "downs_t": [4], "strides_t": [2],
                                                  "width": 5, "depth": 2, "m_conv": 1.0, "dilation_growth_rate": 3}})
    torch.manual_seed(13)
    vqvae = VQVAE(vq_cfg)
    with torch.no_grad():
        vqvae.vq.level_blocks[0].k.copy_(torch.randn(8, 6) * 0.5)
    if stored:
        unpack(vqvae, stored["vqvae_state"])
    vqvae = vqvae.double().eval()
    track_f0 = np.array(data["fcpe_f0_interpolated"]) if not stored else np.array(stored["vq_f0"])
    track_vuv = np.array(data["fcpe_voiced"]) if not stored else np.array(stored["vq_voiced"])
    f0n = track_f0.copy()
    mask = f0n != 0
    f0n[mask] = f0n[mask] - f0n[track_vuv != 0].mean()
    x = torch.tensor(np.stack([f0n, track_vuv]))[None] / torch.tensor([[[50.0], [1.0]]])
    with torch.no_grad():
        codes = vqvae(x)[0].reshape(-1)
        latents = vqvae.encoder(x)[0]
        _, reconstruction = vqvae(x, compute_pred=True)
    data.update(vq_codes=[int(c) for c in codes], vq_latents=flat(latents), vq_reconstruction=flat(reconstruction[0]))
    saved.update(vq_f0=track_f0.tolist(), vq_voiced=track_vuv.tolist(), vq_input=flat(x), vqvae_state=pack(vqvae),
                 vq_config=OmegaConf.to_container(vq_cfg))

    # ------------------------------------------------------------------ vocoders
    from spiritlm.speech_tokenizer.hifigan.hifigan_vocoder import CodeGenerator, CodeHiFiGANVocoderModel
    common = {"resblock": "1", "upsample_rates": [5, 4], "upsample_kernel_sizes": [11, 8], "upsample_initial_channel": 16,
              "resblock_kernel_sizes": [3, 7], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "num_embeddings": 12, "embedding_dim": 4,
              "dur_predictor_params": {"encoder_embed_dim": 4, "var_pred_hidden_dim": 4, "var_pred_kernel_size": 3, "var_pred_dropout": 0.5},
              "multispkr": "from_input_file", "num_speakers": 4, "code_hop_size": 640}
    base_cfg = dict(common, multistyle="from_input_file", num_styles=5, model_in_dim=12)
    expressive_cfg = dict(common, multistyle=None, num_f0_tokens=6, num_style_tokens=7, f0_hop_size=1280, style_hop_size=16000, model_in_dim=16)
    vocoder_units = [3, 7, 1, 1, 9, 0, 11, 4]
    pitch_tokens = [2, 2, 5, 0, 0, 3, 1, 4]
    style_tokens = [6, 6, 6, 1, 1, 1, 1, 2]
    for name, config, seed in (("base", base_cfg, 17), ("expressive", expressive_cfg, 19)):
        torch.manual_seed(seed)
        generator = CodeGenerator(config)
        with torch.no_grad():
            generator.dur_predictor.proj.bias.fill_(1.3)
        if stored:
            unpack(generator, stored[f"vocoder_{name}_state"])
        generator = generator.double().eval()
        kwargs = {"code": torch.tensor([vocoder_units]), "spkr": torch.tensor([[1]]), "dur_prediction": True}
        if name == "base":
            kwargs["style"] = torch.tensor([[3]])
        else:
            kwargs["f0_code"] = torch.tensor([pitch_tokens])
            kwargs["style_code"] = torch.tensor([style_tokens])
        with torch.no_grad():
            wave = generator(**kwargs)
            dur = generator.dur_predictor(generator.dict(torch.tensor([vocoder_units])))
        data[f"vocoder_{name}_wave"] = flat(wave)
        data[f"vocoder_{name}_log_durations"] = flat(dur)
        saved[f"vocoder_{name}_state"] = pack(generator)
        saved[f"vocoder_{name}_config"] = config
    saved.update(vocoder_units=vocoder_units, pitch_tokens=pitch_tokens, style_tokens=style_tokens)
    upsample_cases = [([1, 2, 3], 6, 2), ([1, 2, 3], 7, 2), ([4, 5], 5, 2), ([9], 25, 25), ([1, 2], 49, 25)]
    data["upsampled"] = [CodeHiFiGANVocoderModel.upsample(None, torch.zeros(1, n, dtype=torch.long), torch.tensor([t]), r)[0].tolist()
                         for t, n, r in upsample_cases]
    saved["upsample_cases"] = [list(c) for c in upsample_cases]

    # ------------------------------------------------------------------ style classifier
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification
    style_cfg = Wav2Vec2Config(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, conv_dim=[16, 16, 16],
                               conv_kernel=[10, 3, 3], conv_stride=[5, 2, 2], num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
                               feat_extract_norm="group", do_stable_layer_norm=False, conv_bias=False, classifier_proj_size=8, num_labels=5,
                               hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
                               apply_spec_augment=False)
    torch.manual_seed(23)
    classifier = Wav2Vec2ForSequenceClassification(style_cfg)
    if stored:
        unpack(classifier, stored["style_state"])
    classifier = classifier.double().eval()
    style_audio = tones(1.3, seed=9) if not stored else np.array(stored["style_audio"])

    def style_tokens_of(model, wave, pool):
        # Wav2Vec2StyleEncoder.forward with the feature extractor's normalization in float64.
        normalized = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)
        hidden = model.wav2vec2(torch.tensor(normalized)[None])[0]
        hidden = model.projector(hidden)
        pooled = torch.cat([hidden[:, i:i + pool, :].mean(dim=1) for i in range(0, hidden.shape[1], pool)], dim=1)
        pooled = pooled.view(1, -1, hidden.shape[-1])
        logits = model.classifier(pooled)
        return logits[0], torch.argmax(torch.log_softmax(logits, -1), -1)[0]

    with torch.no_grad():
        style_logits, style_pred = style_tokens_of(classifier, style_audio, 10)
    data.update(style_logits=flat(style_logits), style_pred=[int(v) for v in style_pred])
    saved.update(style_audio=style_audio.tolist(), style_state=pack(classifier),
                 style_config={k: v for k, v in style_cfg.to_dict().items() if isinstance(v, (int, float, str, bool, list))})

    # ------------------------------------------------------------------ token strings
    from spiritlm.speech_tokenizer.spiritlm_tokenizer import string_to_units, units_to_string
    string_cases = [
        {"hubert": "78 42 81 159 316 259", "pitch": "13 13 13 13 13 3", "style": "81 81 81 81 81 81"},
        {"hubert": " ".join(str(v) for v in [5, 5, 5, 9, 9, 2, 2, 2, 2, 7, 7, 1] * 6),
         "pitch": " ".join(str(v) for v in [1, 1, 4, 4, 4, 2] * 6), "style": "3 3 8"},
        {"hubert": "1 1 2 3 3 3 4"},
    ]
    strings = []
    for case in string_cases:
        has_pitch, has_style = "pitch" in case, "style" in case
        strings.append(units_to_string(case, has_pitch=has_pitch, has_style=has_style))
    parse_cases = ["[St81][Hu78][Pi13][Hu42][Hu81][Hu159][Hu316][Pi3][Hu259]", "[Hu4][Hu4][St2][Hu9][Pi7][Hu1]", "[Hu3][Hu5][Text]hello[Speech][Hu6]",
                   "[Pi5][Pi6][Hu2][Hu3][St4][Pi1][Hu8]"]
    parsed = [string_to_units(s) for s in parse_cases]
    parsed_dup = [string_to_units(s, duplicate_hubert_for_multiple_pitch=True) for s in parse_cases]
    data.update(strings=strings, parsed=parsed, parsed_duplicated=parsed_dup)
    saved.update(string_cases=string_cases, parse_cases=parse_cases)

    # ------------------------------------------------------------------ released checkpoints
    if args.dir:
        from spiritlm.speech_tokenizer.hubert.hubert_model.hubert_model import load_hubert_model
        real_audio = librispeech(speech_path, 2.0)
        model, _, _ = load_hubert_model(os.path.join(args.dir, "hubert_25hz", "mhubert_base_25hz.pt"))
        model = double_gelu(model.double().eval())
        real_quantizer = LinearQuantizerModel(os.path.join(args.dir, "hubert_25hz", "L11_quantizer_500.pt")).double().eval()
        with torch.no_grad():
            real_features, _ = model.extract_features(torch.tensor(real_audio)[None], padding_mask=None, mask=False, output_layer=11)
            real_units = real_quantizer(real_features[0])
        data.update(real_hubert_head=flat(real_features[0, :4]), real_hubert_frames=int(real_features.shape[1]), real_units=[int(u) for u in real_units])
        from torchfcpe import spawn_bundled_infer_model
        bundled = spawn_bundled_infer_model(device="cpu").double()
        double_mel(bundled, 128)
        with torch.no_grad():
            _, _, real_raw, real_f0, real_vuv = fcpe_track(bundled, real_audio)
        data.update(real_fcpe_f0=flat(real_raw), real_fcpe_interpolated=flat(real_f0), real_fcpe_voiced=flat(real_vuv))
        from spiritlm.speech_tokenizer.f0.vqvae import load_vqvae
        real_vq, _ = load_vqvae(os.path.join(args.dir, "vqvae_f0_quantizer", "model.pt"))
        real_vq = real_vq.double().eval()
        rf0 = real_f0.reshape(-1).numpy().copy()
        rvuv = real_vuv.reshape(-1).numpy()
        m = rf0 != 0
        rf0[m] = rf0[m] - rf0[rvuv != 0].mean()
        with torch.no_grad():
            real_codes = real_vq(torch.tensor(np.stack([rf0, rvuv]))[None])[0].reshape(-1)
        data["real_pitch_codes"] = [int(c) for c in real_codes]
        for name, folder in (("base", "hifigan_spiritlm_base"), ("expressive", "hifigan_spiritlm_expressive_w2v2")):
            with open(os.path.join(args.dir, folder, "config.json")) as handle:
                config = json.load(handle)
            gen = CodeGenerator(config)
            gen.load_state_dict(torch.load(os.path.join(args.dir, folder, "generator.pt"), map_location="cpu", weights_only=True)["generator"])
            gen = gen.double().eval()
            gen.remove_weight_norm()
            real_kwargs = {"code": torch.tensor([[int(u) for u in real_units[:40]]]), "spkr": torch.tensor([[2]]), "dur_prediction": True}
            if name == "base":
                real_kwargs["style"] = torch.tensor([[8]])
            else:
                n = len(real_kwargs["code"][0])
                real_kwargs["f0_code"] = torch.tensor([[int(real_codes[min(i // 2, len(real_codes) - 1)]) for i in range(n)]])
                real_kwargs["style_code"] = torch.tensor([[71] * n])
            with torch.no_grad():
                real_wave = gen(**real_kwargs)
            data[f"real_vocoder_{name}_head"] = flat(real_wave.reshape(-1)[:2000])
            data[f"real_vocoder_{name}_length"] = int(real_wave.numel())
        style_model = Wav2Vec2ForSequenceClassification.from_pretrained(os.path.join(args.dir, "style_encoder_w2v2")).double().eval()
        with torch.no_grad():
            real_style_logits, real_style = style_tokens_of(style_model, real_audio, 50)
        data.update(real_style_logits=flat(real_style_logits), real_style=[int(v) for v in real_style])
        saved["real_audio"] = real_audio.tolist()

    if args.write:
        _fixture.save(FIXTURE, {**saved, **data}, __file__)
        print("wrote", FIXTURE, {k: (len(v) if isinstance(v, list) else v) for k, v in data.items() if k.startswith(("units", "vq_codes", "style_pred", "real_units", "real_style"))})
        return
    check = _fixture.Comparison("SpiritLM speech tokenizer")
    for k, v in data.items():
        if k not in stored:
            continue
        if isinstance(v, list) and v and isinstance(v[0], float):
            check.floats(k, stored[k], v, 1e-12, float)
        elif stored[k] != v:
            check.failures.append(f"{k}: differs")
    check.report(exact=False)


if __name__ == "__main__":
    main()
