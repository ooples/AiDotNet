"""FireRedTTS's networks from the release's code, on tiny sizes, in float64.

Builds, from the FireRedTTS-1 release (FireRedTeam/FireRedTTS, main branch) unless noted, and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_reference.json:
- the GPT (fireredtts/modules/gpt/gpt.py) with XTTS's training forward (coqui-ai/TTS TTS/tts/layers/xtts/gpt.py,
  set_inputs_and_targets/set_mel_padding/get_logits/forward, bound to the release's GPT): the text and audio
  cross-entropies and the audio logits of one utterance;
- its generate (Hugging Face sampling as do_gpt_inference calls it, with top_k 1 so the seven sequences are the most
  likely tokens) and greedy decoding with the repetition penalty, and Hugging Face's logits processors (repetition
  penalty 2, temperature 0.75, top-k 30, top-p 0.85) on fixed logits;
- the speaker encoder (fireredtts/modules/codec/speaker.py: TorchMelSpectrogram and ECAPA_TDNN, narrowed);
- the flow-matching Mel decoder (fireredtts/modules/flow): μ and the inference mel from fixed noise (3 steps), and the
  flow-matching loss at a fixed time and noise, with and without μ;
- BigVGAN-v2 (fireredtts/modules/bigvgan, narrowed) of that mel;
- the prompt mel (flow/mel_spectrogram.py: kaiser_best resampling, mel_spectrogram) and torchaudio's default resampling;
- the 1S release's speech tokenizer (FireRedTeam/FireRedTTS, fireredtts-1s branch: semantic_tokenizer.SemanticVQVAE,
  narrowed, with a tiny Hugging Face HubertModel standing in for fairseq's, run as HuBERT._extract runs it): the SSL
  features, encodings and tokens of a waveform, and one EMA step of its quantizer (dead-code replacement off).
Precision patches: the release computes Fp32BatchNorm, the relative position table, the flow's time-embedding frequencies
and the mel filterbank in float32; they are computed in float64 here (the filterbank's float32 values kept).

Usage: python fireredtts_reference.py --firered <main checkout> --firered1s <fireredtts-1s checkout> --coqui <coqui-ai/TTS checkout> [--write]
Needs .venv-firered (torch, torchaudio --no-deps, transformers 4.44.2, diffusers 0.27.2, librosa 0.10.2, einops).
"""
import argparse
import ast
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_reference.json"

# GPT
D, HEADS, LAYERS, MAX_TEXT, MAX_AUDIO, TEXT_TOKENS = 16, 2, 2, 20, 40, 30
START_TEXT, STOP_TEXT, AUDIO_TOKENS, START_AUDIO, STOP_AUDIO = 27, 28, 16386, 16384, 16385
SPEAKER, REFERENCE_HIDDEN = 8, 6
TEXT = [3, 7, 11, 4, 9]
CODES = [5, 17, 2, 2, 9, 13, 8, 6]
# ECAPA
ECAPA = dict(channels=[16, 16, 16, 16, 48], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1], attention_channels=8,
             res2net_scale=4, se_channels=8, global_context=True, batch_norm=True)
# flow
CODEBOOK_DIM = 4
FLOW = {
    "output_size": 100,
    "input_embedding": {"out_channels": 8, "freeze": True},
    "encoder": {"input_size": 8, "output_size": 8, "attention_heads": 2, "linear_units": 16, "num_blocks": 2, "dropout_rate": 0.01,
                "srcattention_start_index": 0, "srcattention_end_index": 0, "attention_dropout_rate": 0.01,
                "positional_dropout_rate": 0.01, "key_bias": True, "normalize_before": True},
    "length_regulator": {"channels": 8, "num_blocks": 1},
    "mel_encoder": {"in_channels": 100, "out_channels": 8, "hidden_channels": 8, "reduction_rate": 4, "n_layers": 1, "n_blocks": 1,
                    "kernel_size": 3},
    "decoder": {"t_scheduler": "cosine", "inference_cfg_rate": 0.7,
                "estimator": {"in_channels": 200, "out_channels": 100, "channels": [8, 8], "dropout": 0, "attention_head_dim": 4,
                              "n_blocks": 1, "num_mid_blocks": 1, "num_heads": 2, "act_fn": "gelu"}},
}
FLOW_TOKENS = [5, 300, 16383, 128, 77]
FLOW_STEPS = 3
FLOW_TIME = 0.37
# BigVGAN
BIGVGAN = {"num_mels": 100, "upsample_initial_channel": 64, "upsample_rates": [5, 3, 2, 2, 2, 2], "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4],
           "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3, 5]], "resblock_type": "1", "snake_logscale": True,
           "activation": "snakebeta", "use_tanh_at_final": False, "use_bias_at_final": False, "use_cuda_kernel": False}
# tokenizer
SSL_DIM, TOKENIZER_DIM = 16, 8
HUBERT = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, conv_dim=[8] * 7,
              num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, feat_extract_norm="layer", do_stable_layer_norm=True,
              conv_bias=False, hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
              layerdrop=0.0, final_dropout=0.0)
WAVE_SAMPLES = 9000
EMA_ROWS = 6


def methods_from(path, class_name, names, namespace):
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    exec(compile(ast.Module(body=[item], type_ignores=[]), path, "exec"), namespace)
                    found[item.name] = namespace[item.name]
    missing = set(names) - set(found)
    if missing:
        raise LookupError(f"{missing} not found in {class_name}")
    return found


def tensors(m):
    buffers = [(n, b) for n, b in m.named_buffers()
               if n.endswith("running_mean") or n.endswith("running_var") or n.endswith(".embed") or n.endswith(".embed_avg")
               or n.endswith(".cluster_size")]
    return [(n, p) for n, p in m.named_parameters()] + buffers


def randomize(m, stored, prefix):
    with torch.no_grad():
        for name, p in tensors(m):
            key = prefix + name
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][key], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("running_var"):
                p.copy_((0.5 + torch.rand_like(p)).float())
            elif name.endswith("running_mean"):
                p.copy_((0.3 * torch.randn_like(p)).float())
            elif name.endswith(".cluster_size"):
                p.copy_((0.5 + 2 * torch.rand_like(p)).float())
            elif name.endswith(".embed") or name.endswith(".embed_avg"):
                pass
            elif "norm" in name and name.endswith("weight") and p.dim() == 1:
                p.copy_((1.0 + 0.2 * torch.randn_like(p)).float())
            elif name.endswith(".alpha") or name.endswith(".beta"):
                p.copy_((0.3 * torch.randn_like(p)).float())
            elif name.endswith("pos_bias_u") or name.endswith("pos_bias_v"):
                p.copy_((0.2 * torch.randn_like(p)).float())
            else:
                scale = 0.5 if p.dim() == 1 else 1.0
                p.copy_((p + 0.05 * scale * torch.randn_like(p)).float())


def patch_source(owner, name, old, new):
    """Re-defines owner.name from its source with `old` replaced (the filterbank's float32 values kept in float64)."""
    import inspect
    import textwrap
    function = getattr(owner, name)
    source = textwrap.dedent(inspect.getsource(function))
    assert old in source, (name, old)
    namespace = dict(sys.modules[function.__module__].__dict__)
    exec(compile(source.replace(old, new), inspect.getsourcefile(function), "exec"), namespace)
    setattr(owner, name, namespace[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firered", required=True)
    parser.add_argument("--firered1s", required=True)
    parser.add_argument("--coqui", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2409)
    np.random.seed(2409)

    sys.path.insert(0, os.path.abspath(args.firered))
    import fireredtts.modules.flow.conformer as conformer
    import fireredtts.modules.codec.speaker as speaker_module
    from fireredtts.modules.gpt.gpt import GPT
    from fireredtts.modules.flow import get_flow_frontend
    from fireredtts.modules.flow.mel_spectrogram import mel_spectrogram, kaiser_best_resampling_fn
    from fireredtts.modules.bigvgan import get_bigvgan_backend
    import torchaudio

    # ---- precision patches
    def extend_pe64(self, x):
        if self.pe is not None and self.pe.size(1) >= x.size(1) * 2 - 1:
            self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1).to(device=x.device, dtype=x.dtype)
    conformer.EspnetRelPositionalEncoding.extend_pe = extend_pe64

    def batchnorm64(self, input):
        return self.bn(input)
    speaker_module.Fp32BatchNorm.forward = batchnorm64
    patch_source(speaker_module.TorchMelSpectrogram, "forward", "torch.from_numpy(mel).float()", "torch.from_numpy(mel).double()")
    import fireredtts.modules.flow.mel_spectrogram as flow_mel
    patch_source(flow_mel, "mel_spectrogram", "torch.from_numpy(mel).float()", "torch.from_numpy(mel).double()")
    import fireredtts.modules.flow.decoder as flow_decoder
    patch_source(flow_decoder.SinusoidalPosEmb, "forward", "torch.arange(half_dim, device=device).float()",
                 "torch.arange(half_dim, device=device).double()")

    # ---- the GPT, with XTTS's training forward
    namespace = {"torch": torch, "F": torch.nn.functional}
    xtts = methods_from(os.path.join(args.coqui, "TTS", "tts", "layers", "xtts", "gpt.py"), "GPT",
                        ["set_inputs_and_targets", "set_mel_padding", "get_logits", "forward"], namespace)
    for name, fn in xtts.items():
        setattr(GPT, name if name != "forward" else "xtts_forward", fn)
    gpt = GPT(start_text_token=START_TEXT, stop_text_token=STOP_TEXT, layers=LAYERS, model_dim=D, heads=HEADS, max_text_tokens=MAX_TEXT,
              max_mel_tokens=MAX_AUDIO, max_prompt_tokens=70, code_stride_len=640, number_text_tokens=TEXT_TOKENS,
              num_audio_tokens=AUDIO_TOKENS, start_audio_token=START_AUDIO, stop_audio_token=STOP_AUDIO)
    gpt.reference_embedding = torch.nn.Sequential(torch.nn.Linear(SPEAKER, REFERENCE_HIDDEN), torch.nn.Tanh(),
                                                  torch.nn.Linear(REFERENCE_HIDDEN, D))
    gpt.label_smoothing = 0.0
    speaker = speaker_module.ECAPA_TDNN(80, SPEAKER, **ECAPA)
    mel_extractor = speaker_module.TorchMelSpectrogram()

    # ---- the flow and BigVGAN
    codebook = (torch.randn(2, 128, CODEBOOK_DIM) * 1.5).float()
    if stored is not None:
        codebook = torch.tensor(stored["codebook"], dtype=torch.float32).reshape(2, 128, CODEBOOK_DIM)
    codebook_file = os.path.join(tempfile.mkdtemp(), "codebook.npy")
    np.save(codebook_file, codebook.numpy())
    flow_config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in FLOW.items()}
    flow_config["input_embedding"]["codebook_path"] = codebook_file
    flow = get_flow_frontend(flow_config)
    for table in flow.input_embedding.codebook:
        table.weight.data = table.weight.data.double()
    bigvgan = get_bigvgan_backend(BIGVGAN)

    # ---- the 1S tokenizer
    sys.path.insert(0, os.path.abspath(args.firered1s))
    for name in [k for k in sys.modules if k == "fireredtts" or k.startswith("fireredtts.")]:
        released = sys.modules.pop(name)
        sys.modules["firered_main_" + name] = released
    one_s = types.ModuleType("fireredtts")
    one_s.__path__ = [os.path.join(os.path.abspath(args.firered1s), "fireredtts")]
    sys.modules["fireredtts"] = one_s
    modules_1s = types.ModuleType("fireredtts.modules")
    modules_1s.__path__ = [os.path.join(os.path.abspath(args.firered1s), "fireredtts", "modules")]
    sys.modules["fireredtts.modules"] = modules_1s
    st_pkg = types.ModuleType("fireredtts.modules.semantic_tokenizer")
    st_pkg.__path__ = [os.path.join(os.path.abspath(args.firered1s), "fireredtts", "modules", "semantic_tokenizer")]
    sys.modules["fireredtts.modules.semantic_tokenizer"] = st_pkg
    hubert_stub = types.ModuleType("fireredtts.modules.semantic_tokenizer.hubert")
    hubert_stub.HuBERT = object
    sys.modules["fireredtts.modules.semantic_tokenizer.hubert"] = hubert_stub
    import fireredtts.modules.semantic_tokenizer.semantic_tokenizer as sast_module
    import fireredtts.modules.semantic_tokenizer.ecapa_tdnn as sast_ecapa
    sast_ecapa.Fp32BatchNorm.forward = batchnorm64
    import fireredtts.modules.semantic_tokenizer.audio as sast_audio
    patch_source(sast_audio.TorchMelSpectrogram, "forward", "torch.from_numpy(mel).float()", "torch.from_numpy(mel).double()")
    sast_module.ECAPA_TDNN = lambda out_dim, n_model_size, **_: sast_ecapa.ECAPA_TDNN(out_dim, n_model_size, **ECAPA)
    sast = sast_module.SemanticVQVAE(
        in_dim=SSL_DIM, out_dim=80, n_model_size=TOKENIZER_DIM, downsample_scales=[1, 1, 1, 2], upsample_scales=[[2, 1], [2, 1, 1, 1]],
        mel_config={"style": "BigVGAN", "filter_length": 1024, "hop_length": 160, "win_length": 640, "n_mel_channels": 80,
                    "sampling_rate": 16000},
        vq_config={"codebook_size": [128, 128], "codebook_dim": [CODEBOOK_DIM, CODEBOOK_DIM], "requires_projection": True},
        tree_config=[{"downsample_rate": 1, "n_groups": 1, "dropout": 0}], n_samples_per_token=640, checkpointing=False)
    from transformers import HubertConfig, HubertModel
    hubert = HubertModel(HubertConfig(**HUBERT))

    m = torch.nn.ModuleDict({"gpt": gpt, "speaker": speaker, "flow": flow, "bigvgan": bigvgan, "sast": sast, "hubert": hubert})
    m.eval()
    randomize(m, stored, "")
    with torch.no_grad():
        # the SAST codebooks are the flow's
        for b in range(2):
            cb = sast.quantizer.quantizers[0]._codebook.codebooks[b]
            cb.embed.copy_(codebook[b].double())
            if stored is None:
                cb.embed_avg.copy_((codebook[b].double() * 1.3).float())
    gpt.init_gpt_for_inference(kv_cache=True)
    initial_state = {n: _fixture.flat(p) for n, p in tensors(m)}       # before the EMA step changes the codebooks

    def tensor(key, shape, std=1.0):
        if stored is not None:
            return torch.tensor(stored[key], dtype=torch.float32).reshape(shape).to(torch.float64)
        return (std * torch.randn(shape)).float().to(torch.float64)
    t = torch.arange(16000, dtype=torch.float64)
    inputs = {
        "wave16k": tensor("wave16k", (16000,), 0.1),
        "wave24k": tensor("wave24k", (6000,), 0.1),
        "flow_noise": tensor("flow_noise", (1, 100, 4 * len(FLOW_TOKENS))),
        "loss_noise": tensor("loss_noise", (1, 100, 4 * len(FLOW_TOKENS))),
        "target_mel": tensor("target_mel", (1, 100, 4 * len(FLOW_TOKENS))),
        "logits": tensor("logits", (40,), 2.0),
        "ema_encodings": tensor("ema_encodings", (EMA_ROWS, TOKENIZER_DIM), 1.0),
    }
    out, results = {}, {}
    with torch.no_grad():
        # speaker
        speaker_mel = mel_extractor(inputs["wave16k"].unsqueeze(0))                       # [1, frames, 80]
        out["speaker_mel"] = speaker_mel[0]
        embedding = speaker(speaker_mel)[0]
        out["speaker_embedding"] = embedding

        # GPT training forward (cond latents from the speaker embedding)
        cond = gpt.reference_embedding(embedding.view(1, 1, SPEAKER))                     # [1, 1, D]
        out["cond_latent"] = cond[0]
        text = torch.tensor([TEXT])
        codes = torch.tensor([CODES])
        loss_text, loss_mel, mel_logits = gpt.xtts_forward(text, torch.tensor([len(TEXT)]), codes.clone(), torch.tensor([len(CODES) * 640]),
                                                           cond_mels=torch.zeros(1, 80, 1), cond_latents=cond)
        out["loss_text"] = loss_text.reshape(1)
        out["loss_mel"] = loss_mel.reshape(1)
        out["mel_logits"] = mel_logits[0].T

        # generation: HF sampling with top_k 1 (the seven sequences are the most likely tokens), and greedy
        text_tokens = torch.tensor([TEXT], dtype=torch.int32)
        torch.manual_seed(7)
        sampled = gpt.generate(cond_latents=cond, text_inputs=text_tokens, input_tokens=None, do_sample=True, top_p=0.85, top_k=1,
                               temperature=0.75, num_return_sequences=7, num_beams=1, length_penalty=1.0, repetition_penalty=2.0,
                               output_attentions=False)
        results["sampled"] = [s.tolist() for s in sampled]
        greedy = gpt.generate(cond_latents=cond, text_inputs=text_tokens, input_tokens=None, do_sample=False, num_beams=1,
                              repetition_penalty=2.0)
        results["greedy"] = greedy[0].tolist()

        # Hugging Face's processors on fixed logits
        from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                                                             TopKLogitsWarper, TopPLogitsWarper)
        history = torch.tensor([[1, 1, 1, 3, 7, 3, 12]])
        scores = inputs["logits"].clone().unsqueeze(0)
        for processor in [RepetitionPenaltyLogitsProcessor(2.0), TemperatureLogitsWarper(0.75), TopKLogitsWarper(30), TopPLogitsWarper(0.85)]:
            scores = processor(history, scores)
        out["processed"] = scores[0].clamp(min=-1e30)

        # the flow
        prompt_mel = flow_mel.mel_spectrogram(inputs["wave16k"].unsqueeze(0), 1024, 100, 16000, 160, 1024, 0, None)[0].T   # [F, 100]
        out["prompt_mel"] = prompt_mel
        token = torch.tensor([FLOW_TOKENS])
        token_len = torch.tensor([len(FLOW_TOKENS)])
        prompt_feat = flow.prompt_prenet(prompt_mel.unsqueeze(0))
        out["prompt_features"] = prompt_feat[0]
        prompt_feat_len = torch.ceil(torch.tensor([prompt_mel.shape[0]]) / flow.prompt_prenet.reduction_rate).long()
        embedded = flow.input_embedding(torch.clamp(token, min=0))
        out["embedded"] = embedded[0]
        h, _ = flow.encoder(embedded, token_len, prompt_feat, prompt_feat_len)
        out["encoded"] = h[0]
        h, _ = flow.length_regulator(h, (token_len * 4).int())
        mu = flow.encoder_proj(h)                                                                 # [1, T, 100]
        out["mu"] = mu[0]
        noise = inputs["flow_noise"]
        original_randn_like = torch.randn_like
        torch.randn_like = lambda x, **kw: noise.clone()
        mel = flow.inference(token, token_len, prompt_mel.unsqueeze(0), torch.tensor([prompt_mel.shape[0]]), n_timesteps=FLOW_STEPS)
        torch.randn_like = original_randn_like
        out["flow_mel"] = mel[0].T
        # flow-matching loss at a fixed time and noise (eq. 6, CosyVoice's compute_loss form)
        frames = mu.shape[1]
        mask = torch.ones(1, 1, frames)
        sigma = 1e-6
        x1 = inputs["target_mel"]
        z = inputs["loss_noise"]
        tt = torch.tensor([FLOW_TIME])
        y = (1 - (1 - sigma) * tt) * z + tt * x1
        u = x1 - (1 - sigma) * z
        for key, condition in [("flow_loss", mu.transpose(1, 2)), ("flow_loss_dropped", torch.zeros_like(mu.transpose(1, 2)))]:
            v = flow.decoder.estimator(y, mask, condition, tt)
            out[key] = (((v - u) ** 2).sum() / (frames * 100)).reshape(1)

        # BigVGAN
        out["waveform"] = bigvgan(mel)[0, 0]

        # resampling and the prompt mel's kaiser_best
        out["kaiser_up"] = kaiser_best_resampling_fn(inputs["wave16k"].unsqueeze(0), orig_freq=16000, new_freq=24000)[0]
        out["kaiser_down"] = kaiser_best_resampling_fn(inputs["wave24k"].unsqueeze(0), orig_freq=24000, new_freq=16000)[0]
        out["hann_down"] = torchaudio.functional.resample(inputs["wave24k"].unsqueeze(0), 24000, 16000)[0]

        # the tokenizer, as HuBERT._extract and extract_speech_tokens run it
        wave = inputs["wave16k"][:WAVE_SAMPLES].unsqueeze(0)
        pad = 640 - wave.shape[1] % 640
        wave = torch.nn.functional.pad(wave, (0, pad))
        length = wave.shape[1]
        feats = torch.nn.functional.pad(wave, (0, 3200))
        feats = torch.nn.functional.layer_norm(feats[0, : length + 3200], feats[0, : length + 3200].shape).unsqueeze(0)
        frames_ssl = math.ceil(length / 320)
        padding = torch.nn.functional.pad(torch.zeros(1, frames_ssl, dtype=torch.bool), (0, 9), value=True)
        extracted = hubert.feature_extractor(feats).transpose(1, 2)
        projected = hubert.feature_projection(extracted)
        extra = padding.size(1) % projected.size(1)
        if extra > 0:
            padding = padding[:, :-extra]
        padding = padding.view(padding.size(0), projected.size(1), -1).all(-1)
        encoded_ssl = hubert.encoder(projected, attention_mask=~padding).last_hidden_state[:, :frames_ssl]
        out["ssl"] = encoded_ssl[0]
        encodings = sast.encoder(encoded_ssl)
        out["sast_encodings"] = encodings[0]
        sast.eval()
        quant, (quants, _, indices) = sast.quantizer(encodings, torch.tensor([encodings.shape[1]]), update_codebook=False)
        results["tokens"] = indices[0][0].tolist()
        out["sast_quantized"] = quant[0]

        # one EMA step (training mode, dead codes kept)
        vq = sast.quantizer.quantizers[0]
        for cb in vq._codebook.codebooks:
            cb.threshold_ema_dead_code = 0
            cb.decay = 0.9
        vq.train()
        projected_in = vq.project_in(inputs["ema_encodings"].unsqueeze(0))
        _, commit, ema_tokens = vq._codebook(projected_in, torch.tensor([EMA_ROWS]))
        results["ema_tokens"] = ema_tokens[0].tolist()
        out["ema_commitment"] = commit.reshape(1)
        out["ema_codebook0"] = vq._codebook.codebooks[0].embed
        out["ema_codebook1"] = vq._codebook.codebooks[1].embed

    results["frames"] = int(prompt_mel.shape[0])
    if args.write:
        data = {"state": initial_state, "codebook": _fixture.flat(codebook), **results}
        data.update({key: _fixture.flat(value) for key, value in inputs.items()})
        data.update({key: [float(v) for v in value.reshape(-1).tolist()] for key, value in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {FIXTURE}: greedy {results['greedy']}, tokens {results['tokens']}")
        return
    check = _fixture.Comparison("FireRedTTS")
    for key, value in out.items():
        check.floats(key, stored[key], [float(v) for v in value.reshape(-1).tolist()], dtype="float64")
    for key, value in results.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
