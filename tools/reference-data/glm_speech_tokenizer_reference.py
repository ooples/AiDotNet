"""GLM-4-Voice's speech tokenizer from THUDM's speech_tokenizer/modeling_whisper.py, on tiny sizes, and its features.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_speech_tokenizer_reference.json, in float64 on float32
weights:
- WhisperFeatureExtractor's 128-band log-mel features of 1.3 s of audio padded to a multiple of 1280 samples, as
  extract_speech_token asks for them, computed with the extractor's filters and arithmetic in float64 (both of its paths
  compute in float32: the torch one casts, and spectrogram() keeps a complex64 buffer);
- a tiny WhisperVQForConditionalGeneration (16 wide, 4 encoder layers with the pooling and the quantizer after layer 2,
  causal convolutions, 8-frame attention blocks, 12 codes; 2 decoder layers) on features with padded frames: the codes and
  the encoder output in eval mode, and one training forward with labels (the ASR cross-entropy plus the quantize loss,
  the EMA codebook update and a dead-code restart at its first update, random.sample replaced by the first rows).
The release casts the EMA arithmetic's inputs with .float(); that cast is redirected to float64 here.

Usage (default: verify; --write: regenerate), with the GLM-4-Voice checkout (for speech_tokenizer):
  python tools/reference-data/glm_speech_tokenizer_reference.py --glm4voice <dir> [--write]
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_speech_tokenizer_reference.json"


def floats(t):
    return [float(x) for x in torch.as_tensor(t).reshape(-1).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--glm4voice", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.glm4voice)
    from speech_tokenizer.configuration_whisper import WhisperVQConfig
    from speech_tokenizer.modeling_whisper import WhisperVQForConditionalGeneration
    from transformers import WhisperFeatureExtractor
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(31)
    np.random.seed(31)

    # Features.
    samples = 20800  # 1.3 s; padded to 21760, a multiple of 1280
    audio = np.array(stored["audio"]) if stored else (0.3 * np.sin(np.arange(samples) * 0.07) * np.random.rand(samples)).astype(np.float32).astype(np.float64)
    extractor = WhisperFeatureExtractor(feature_size=128)
    length = -(-samples // 1280) * 1280
    padded = np.zeros(length)
    padded[:samples] = audio
    # _np_extract_fbank_features's arithmetic in float64: transformers' spectrogram() accumulates its STFT in a complex64
    # buffer whatever dtype it is given, so the frames are formed and transformed here (centred, reflect-padded, periodic
    # Hann, power, the extractor's Slaney filters, log10 above 1e-10, the last frame dropped).
    from transformers.audio_utils import window_function
    window = window_function(extractor.n_fft, "hann")
    reflected = np.pad(padded, (extractor.n_fft // 2, extractor.n_fft // 2), mode="reflect")
    count = length // extractor.hop_length + 1
    framed = np.stack([reflected[i * extractor.hop_length: i * extractor.hop_length + extractor.n_fft] * window for i in range(count)])
    power = np.abs(np.fft.rfft(framed, axis=1)) ** 2
    log_spec = np.log10(np.maximum(1e-10, power @ extractor.mel_filters)).T[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    features = (log_spec + 4.0) / 4.0

    # The tiny model.
    config = WhisperVQConfig(
        d_model=16, encoder_attention_heads=2, encoder_ffn_dim=32, encoder_layers=4, decoder_layers=2, decoder_attention_heads=2,
        decoder_ffn_dim=32, num_mel_bins=128, max_source_positions=64, max_target_positions=16, vocab_size=40, pad_token_id=0,
        decoder_start_token_id=1, eos_token_id=2, bos_token_id=1, pooling_kernel_size=4, pooling_position=2, pooling_type="avg",
        quantize_position=2, quantize_vocab_size=12, quantize_causal_block_size=8, quantize_ema_decay=0.99,
        quantize_commit_coefficient=0.25, quantize_loss_scale=10.0, quantize_restart_interval=1, quantize_encoder_only=False,
        encoder_causal_convolution=True, encoder_causal_attention=False, quantize_causal_encoder=False, dropout=0.0,
        attention_dropout=0.0, activation_dropout=0.0, scale_embedding=False)
    config._attn_implementation = "eager"
    model = WhisperVQForConditionalGeneration(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored:
                p.copy_(torch.tensor(stored["state"][name]).reshape(p.shape))
            elif "layer_norm" in name:
                p.copy_((1 + 0.2 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
    model = model.double()
    # The parameters as initialized (the training forward then moves the codebook).
    initial_state = {n: _fixture.flat(p) for n, p in model.named_parameters() if not n.startswith("proj_out")}
    encoder = model.model.encoder
    with torch.no_grad():
        encoder.ema_weight.copy_(encoder.codebook.weight)
        # Some codes start rare, so the restart at the first update has dead codes to replace.
        initial_counts = torch.tensor(stored["ema_count_initial"]) if stored else torch.tensor([1.0, 0.4, 2.0, 0.3, 1.5, 0.2, 1.0, 0.5, 3.0, 1.0, 0.6, 1.2])
        encoder.ema_count.copy_(initial_counts)
    frames = 64
    inputs = torch.tensor(stored["inputs"]).reshape(1, 128, frames) if stored else torch.randn(1, 128, frames).float().double()
    mask = torch.ones(1, frames, dtype=torch.long)
    mask[0, 55:] = 0
    labels = torch.tensor([[5, 9, 3, 7, 2]])

    model.eval()
    with torch.no_grad():
        out = encoder(inputs, attention_mask=mask, return_dict=True)
    eval_codes = out.quantized_token_ids[0].tolist()
    eval_hidden = out.last_hidden_state[0]
    with torch.no_grad():
        eval_ce = model(input_features=inputs, attention_mask=mask, labels=labels, return_dict=True).loss

    # One training forward: gloo for the release's all_reduce/all_gather, the first rows for random.sample, float64 for
    # the EMA's .float() casts.
    if not torch.distributed.is_initialized():
        import tempfile
        store = os.path.join(tempfile.mkdtemp(), "store").replace("\\", "/")
        torch.distributed.init_process_group("gloo", init_method="file:///" + store.lstrip("/"), rank=0, world_size=1)
    random.sample = lambda population, k: list(population)[:k]
    original_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self.double()
    model.train()
    try:
        result = model(input_features=inputs, attention_mask=mask, labels=labels, return_dict=True)
    finally:
        torch.Tensor.float = original_float
    total = result.loss
    quantize = encoder.quantize_loss
    train_codes = None
    with torch.no_grad():
        model.eval()
        train_codes = encoder(inputs, attention_mask=mask, return_dict=True).quantized_token_ids[0].tolist()

    data_out = {"eval_ce": eval_ce, "features": features, "eval_hidden": eval_hidden, "total_loss": total.detach(), "quantize_loss": quantize.detach(),
                "codebook": encoder.codebook.weight.detach(), "ema_count": encoder.ema_count, "ema_weight": encoder.ema_weight}
    if args.write:
        data = {"ema_count_initial": initial_counts.tolist(), "audio": audio.tolist(), "inputs": floats(inputs), "mask": mask[0].tolist(), "labels": labels[0].tolist(),
                "state": initial_state,
                "eval_codes": eval_codes, "codes_after_update": train_codes}
        data.update({k: floats(v) for k, v in data_out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "codes", eval_codes, "after", train_codes)
        return
    check = _fixture.Comparison("GLM-4-Voice speech tokenizer")
    for k, v in data_out.items():
        check.floats(k, stored[k], floats(v), dtype="float64")
    check.exact("eval codes", stored["eval_codes"], eval_codes)
    check.exact("codes after update", stored["codes_after_update"], train_codes)
    check.report()


if __name__ == "__main__":
    main()
