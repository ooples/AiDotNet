"""SpeechT5 as Hugging Face transformers runs it (microsoft/speecht5_tts and microsoft/speecht5_hifigan).

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_hf_reference.json, in float64 on float32 weights:
a tiny SpeechT5ForTextToSpeech's state under the Hugging Face names, its teacher-forced outputs (the frames before and
after the post-net and the stop logits) and its generate_speech output (the summed stop rule); a tiny SpeechT5HifiGan's
state with its mean and scale and its waveform of a spectrogram. The speech decoder pre-net's always-on dropout is
replaced by the identity (Hugging Face's _consistent_dropout cannot be switched off: at p = 0 it zeroes everything).
The scaled positional tables, which transformers builds in float32, are rebuilt in float64.

With --released <speecht5-hf dir> (holding tts/ and hifigan/), also records in speecht5_hf_released_reference.json the
released model's token ids of a sentence, its teacher-forced outputs and generation for a fixed x-vector, and the
released HiFi-GAN's waveform of the generated spectrogram.

Usage (default: verify; --write: regenerate), in a venv with transformers 4.57.1 and sentencepiece:
  python tools/reference-data/speecht5_hf_reference.py [--released <dir>] [--write]
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_hf_reference.json"
RELEASED = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_hf_released_reference.json"
SENTENCE = "Hello, SpeechT5 reads this sentence."


def floats(t):
    return [float(x) for x in t.detach().reshape(-1).tolist()]


def patch(modeling):
    modeling.SpeechT5SpeechDecoderPrenet._consistent_dropout = lambda self, x, p: x


def rebuild_positions(model, modeling):
    for module in model.modules():
        if isinstance(module, modeling.SpeechT5ScaledPositionalEncoding):
            max_len, dim = module.pe.shape[1], module.pe.shape[2]
            pe = torch.zeros(max_len, dim, dtype=torch.float64)
            position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
            div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float64) * -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            module.pe = pe.unsqueeze(0)


def teacher_forced(model, ids, speaker, mel):
    r = model.config.reduction_factor
    inputs = mel[:, r - 1::r]
    shifted = torch.zeros_like(inputs)
    shifted[:, 1:] = inputs[:, :-1]
    encoder = model.speecht5.encoder(input_values=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    hidden = model.speecht5.decoder(input_values=shifted, speaker_embeddings=speaker, encoder_hidden_states=encoder,
                                    encoder_attention_mask=torch.ones_like(ids)).last_hidden_state
    return model.speech_decoder_postnet(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--released")
    args = parser.parse_args()
    import transformers.models.speecht5.modeling_speecht5 as modeling
    from transformers import SpeechT5Config, SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5HifiGanConfig
    patch(modeling)
    torch.set_default_dtype(torch.float64)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(5)
    config = SpeechT5Config(vocab_size=15, hidden_size=16, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=32,
                            decoder_layers=2, decoder_attention_heads=2, decoder_ffn_dim=32, encoder_max_relative_position=4,
                            max_text_positions=40, max_speech_positions=200, num_mel_bins=10, reduction_factor=2,
                            speaker_embedding_dim=6, speech_decoder_prenet_units=8, speech_decoder_prenet_layers=2,
                            speech_decoder_postnet_units=8, speech_decoder_postnet_layers=5, speech_decoder_postnet_kernel=5,
                            hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, positional_dropout=0.0,
                            encoder_layerdrop=0.0, decoder_layerdrop=0.0, speech_decoder_prenet_dropout=0.5,
                            speech_decoder_postnet_dropout=0.0, pad_token_id=1, eos_token_id=2)
    model = SpeechT5ForTextToSpeech(config)
    state = model.state_dict()
    with torch.no_grad():
        for name, p in state.items():
            if name.endswith("num_batches_tracked") or name.endswith(".pe"):
                continue
            if stored:
                p.copy_(torch.tensor(stored["state"][name]).reshape(p.shape))
            elif "running_var" in name:
                p.copy_((1 + 0.3 * torch.rand_like(p)).float())
            elif "running_mean" in name:
                p.copy_((0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.4 * torch.randn_like(p)).float())
    model = model.double().eval()
    rebuild_positions(model, modeling)

    ids = torch.tensor([stored["tokens"]]) if stored else torch.tensor([[4, 9, 3, 12, 7, 5, 2]])
    speaker = torch.tensor([stored["speaker"]], dtype=torch.float64) if stored else torch.randn(1, 6).float().double()
    mel = torch.tensor(stored["mel"], dtype=torch.float64).reshape(1, 12, 10) if stored else torch.randn(1, 12, 10).float().double()
    with torch.no_grad():
        before, after, logits = teacher_forced(model, ids, speaker, mel)
        generated = model.generate_speech(ids, speaker, threshold=0.5, maxlenratio=3.0)

    vocoder_config = SpeechT5HifiGanConfig(model_in_dim=10, upsample_initial_channel=16, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
                                           resblock_kernel_sizes=[3, 5], resblock_dilation_sizes=[[1, 3], [1, 3]], normalize_before=True)
    vocoder = SpeechT5HifiGan(vocoder_config)
    vocoder_state = vocoder.state_dict()
    with torch.no_grad():
        for name, p in vocoder_state.items():
            if stored:
                p.copy_(torch.tensor(stored["vocoder_state"][name]).reshape(p.shape))
            elif name == "scale":
                p.copy_((0.5 + torch.rand_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
    vocoder = vocoder.double().eval()
    with torch.no_grad():
        wave = vocoder(mel[0])

    out = {"before": before[0], "after": after[0], "logits": logits[0], "generated": generated, "wave": wave}
    if args.write:
        data = {"tokens": ids[0].tolist(), "speaker": floats(speaker[0]), "mel": floats(mel),
                "state": {n: _fixture.flat(p) for n, p in model.state_dict().items() if not n.endswith("num_batches_tracked") and not n.endswith(".pe")},
                "vocoder_state": {n: _fixture.flat(p) for n, p in vocoder.state_dict().items()},
                "generated_frames": int(generated.shape[0])}
        data.update({k: floats(v) for k, v in out.items()})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, {k: list(v.shape) for k, v in out.items()})
    else:
        check = _fixture.Comparison("SpeechT5 (transformers)")
        for k, v in out.items():
            check.floats(k, stored[k], floats(v), dtype="float64")
        check.report()

    if args.released:
        released(args, modeling)


def released(args, modeling):
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Tokenizer
    stored = None if args.write else _fixture.load(RELEASED)
    tokenizer = SpeechT5Tokenizer.from_pretrained(os.path.join(args.released, "tts"))
    texts = [SENTENCE, "Numbers 1 2 3, and UPPER case?", "a", "  spaced   out  ", "café ☃☃ naïve 12:30!"]
    token_ids = [tokenizer(t)["input_ids"] for t in texts]
    model = SpeechT5ForTextToSpeech.from_pretrained(os.path.join(args.released, "tts")).double().eval()
    rebuild_positions(model, modeling)
    vocoder = SpeechT5HifiGan.from_pretrained(os.path.join(args.released, "hifigan")).double().eval()
    torch.manual_seed(7)
    speaker = torch.tensor([stored["speaker"]], dtype=torch.float64) if stored else torch.randn(1, 512).float().double()
    mel = torch.tensor(stored["mel"], dtype=torch.float64).reshape(1, 20, 80) if stored else (torch.randn(1, 20, 80) - 5).float().double()
    ids = torch.tensor([token_ids[0]])
    with torch.no_grad():
        before, after, logits = teacher_forced(model, ids, speaker, mel)
        generated = model.generate_speech(ids, speaker, threshold=0.5)
        wave = vocoder(generated)
    out = {"after": after[0], "logits": logits[0], "generated": generated, "wave": wave}
    if args.write:
        data = {"texts": texts, "token_ids": token_ids, "speaker": floats(speaker[0]), "mel": floats(mel), "generated_frames": int(generated.shape[0])}
        data.update({k: floats(v) for k, v in out.items()})
        _fixture.save(RELEASED, data, __file__)
        print("wrote", RELEASED, {k: list(v.shape) for k, v in out.items()})
        return
    check = _fixture.Comparison("SpeechT5 released (transformers)")
    check.exact("token ids", stored["token_ids"], token_ids)
    for k, v in out.items():
        check.floats(k, stored[k], floats(v), dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
