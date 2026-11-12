"""EmotiVoice reference: netease-youdao/EmotiVoice (models/prompt_tts_modified, models/hifigan, mel_process.py, feats.py),
its released code (there is no paper).

Builds a tiny randomly initialized JETSGenerator (the acoustic model and HiFi-GAN) with the release's own code (pass a
checkout with --emotivoice), stores its weights in float32 and records, in float64 and evaluation mode:
- the teacher-forced training forward: the alignment log-probabilities, Viterbi durations, binarization loss, the
  forward-sum loss, the token-averaged pitch and energy, the predictors' outputs and the prosody losses, the decoder's
  latent, HiFi-GAN's waveform of a 4-frame segment and the training mel loss against a recording's segment;
- inference: the waveform from predicted durations, pitch and energy;
- the data features: TacotronSTFT's log mel, feats.Pitch (pyworld dio + stonemask, made continuous), feats.Energy, and
  the loader's silence trim;
- a tiny SimBERT's pooled output, and SimBERT's tokenization of a few prompts (pass its vocabulary with --simbert).
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/emotivoice_reference.json. Default: verify; --write: regenerate.

The positional table, the duration loss, the energy, the mel bases and the Fourier basis run in float64
(the release builds them in float32). librosa 1.0 removed the positional arguments of librosa.filters.mel that mel_process.py passes; the call is wrapped with
keywords. Needs torch, transformers, librosa, pyworld, numba, munch and scipy.
"""
import argparse
import base64
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from munch import munchify
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/emotivoice_reference.json"
SR, FFT, HOP, MELS = 16000, 64, 16, 10
MODEL = dict(speaker_embed_dim=16, bert_embedding=8, lang_embed_dim=0,
             encoder_n_layers=2, encoder_n_heads=2, encoder_n_hidden=16, encoder_p_dropout=0.2, encoder_kernel_size_conv_mod=3,
             encoder_kernel_size_depthwise=7,
             decoder_n_layers=2, decoder_n_heads=2, decoder_n_hidden=16, decoder_p_dropout=0.2, decoder_kernel_size_conv_mod=3,
             decoder_kernel_size_depthwise=31,
             variance_n_hidden=16, variance_n_layers=3, variance_kernel_size=3, variance_p_dropout=0.1, variance_embed_kernel_size=9,
             variance_embde_p_dropout=0.0, stop_pitch_gradient=False, stop_duration_gradient=False, duration_p_dropout=0.5,
             duration_n_layers=2, duration_kernel_size=3,
             resblock="1", upsample_rates=[4, 4], upsample_kernel_sizes=[8, 8], initial_channel=MELS, upsample_initial_channel=16,
             resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], r=1)
CONFIG = dict(model=MODEL, n_mels=MELS, n_vocab=20, n_speaker=3, segment_size=4)
BERT = dict(vocab_size=30, hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16, max_position_embeddings=32)
PHONEMES = [1, 7, 3, 12, 5, 9, 2]
SPEAKER = 1
FRAMES = 24
STYLE_IDS = [2, 11, 5, 17, 3]


def load(root):
    os.chdir(root)
    sys.path.insert(0, root)
    import librosa
    import mel_process
    original = librosa.filters.mel
    mel_process.librosa_mel_fn = lambda sr, n_fft, n_mels, fmin, fmax: original(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                                                                      dtype=np.float64)
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.modules import encoder

    # Precision patch: ESPnet's positional table is built in float32 at construction and only cast up; it is rebuilt in
    # float64 for the input's length.
    def extend_pe(self, x):
        position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(x.size(1), self.d_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(dtype=x.dtype)

    encoder.PositionalEncoding.extend_pe = extend_pe
    from models.prompt_tts_modified import loss, feats, tacotron_stft
    return JETSGenerator, loss, feats, tacotron_stft, mel_process


def build(loaded, state=None):
    torch.manual_seed(2294)
    model = loaded[0](munchify(CONFIG))
    if state is not None:
        model.load_state_dict(state, strict=True)
    else:
        with torch.no_grad():
            model.am.encoder.embed[0].alpha.fill_(0.7)
            model.am.decoder.embed[0].alpha.fill_(1.3)
    return model.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    mel = torch.randn(1, MELS, FRAMES, generator=g, dtype=torch.float32) * 0.5 - 4
    pitch = torch.randn(1, FRAMES, generator=g, dtype=torch.float32)
    energy = torch.randn(1, FRAMES, generator=g, dtype=torch.float32)
    style = torch.randn(1, 8, generator=g, dtype=torch.float32)
    content = torch.randn(1, 8, generator=g, dtype=torch.float32)
    wave = torch.randn(FRAMES * HOP, generator=g, dtype=torch.float32) * 0.1
    t = np.arange(3200) / SR
    voice = (0.4 * np.sin(2 * np.pi * 180 * t) * (t > 0.02) * (t < 0.18)).astype(np.float32)
    return mel, pitch, energy, style, content, wave, voice


def flat(t):
    return [float(v) for v in torch.as_tensor(t).detach().reshape(-1).tolist()]


def run(loaded, model):
    JETSGenerator, loss, feats, tacotron_stft, mel_process = loaded
    mel, pitch, energy, style, content, wave, voice = inputs()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    out = {}
    try:
        with torch.no_grad():
            captured = {}
            handle = model.am.embed_projection1.register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))
            torch.manual_seed(11)
            o = model(inputs_ling=torch.tensor([PHONEMES]), input_lengths=torch.tensor([len(PHONEMES)]), inputs_speaker=torch.tensor([SPEAKER]),
                      inputs_style_embedding=style.double(), inputs_content_embedding=content.double(), mel_targets=mel.double(),
                      output_lengths=torch.tensor([FRAMES]), pitch_targets=pitch.double(), energy_targets=energy.double())
            handle.remove()
            out["conditioned"] = flat(captured["x"])
            out["log_p_attn"] = flat(o["log_p_attn"])
            out["durations"] = [int(v) for v in o["duration_targets"].reshape(-1).tolist()]
            out["bin_loss"] = [float(o["bin_loss"])]
            out["pitch_targets"] = flat(o["pitch_targets"])
            out["energy_targets"] = flat(o["energy_targets"])
            out["pitch_predictions"] = flat(o["pitch_predictions"])
            out["energy_predictions"] = flat(o["energy_predictions"])
            out["log_duration_predictions"] = flat(o["log_duration_predictions"])
            out["latent"] = flat(o["dec_outputs"])
            out["segment_start"] = int(o["z_start_idxs"][0])
            out["wave"] = flat(o["wav_predictions"])
            dur, pit, ene = loss.ProsodyReconLoss()(torch.tensor([len(PHONEMES)]), o["duration_targets"], o["pitch_targets"],
                                                   o["energy_targets"], o["log_duration_predictions"], o["pitch_predictions"],
                                                   o["energy_predictions"])
            # Precision patch: ProsodyReconLoss takes log(d + 1) of the durations cast to float32; the same mean runs on them
            # in float64.
            dur = torch.mean(torch.abs(torch.log(o["duration_targets"].double() + 1) - o["log_duration_predictions"]))
            out["prosody_losses"] = [float(dur), float(pit), float(ene)]
            out["forward_sum"] = [float(loss.ForwardSumLoss()(o["log_p_attn"], torch.tensor([len(PHONEMES)]), torch.tensor([FRAMES])))]
            start = out["segment_start"] * HOP
            real = wave.double()[start:start + CONFIG["segment_size"] * HOP]
            y_hat = mel_process.mel_spectrogram_torch(o["wav_predictions"].squeeze(1), FFT, MELS, SR, HOP, FFT, 0, 8000)
            y = mel_process.mel_spectrogram_torch(real[None], FFT, MELS, SR, HOP, FFT, 0, 8000)
            out["mel_loss"] = [float(F.l1_loss(y, y_hat))]
            out["training_mel"] = flat(y_hat)
            o2 = model(inputs_ling=torch.tensor([PHONEMES]), input_lengths=torch.tensor([len(PHONEMES)]), inputs_speaker=torch.tensor([SPEAKER]),
                       inputs_style_embedding=style.double(), inputs_content_embedding=content.double())
            out["inference_wave"] = flat(o2["wav_predictions"])
    finally:
        torch.set_default_dtype(previous)
    # The data features.
    stft = tacotron_stft.TacotronSTFT(FFT, HOP, FFT, MELS, SR, 0.0, 8000.0)
    voice_tensor = torch.from_numpy(voice).double()
    # Precision patch: the release builds the mel basis and the windowed Fourier basis in float32; both are rebuilt in
    # float64.
    import librosa
    from scipy.signal import get_window
    stft = stft.double()
    stft.mel_basis = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=FFT, n_mels=MELS, fmin=0.0, fmax=8000.0, dtype=np.float64))
    fourier = np.fft.fft(np.eye(FFT))
    fourier = np.vstack([np.real(fourier[:FFT // 2 + 1]), np.imag(fourier[:FFT // 2 + 1])])
    stft.stft_fn.forward_basis = torch.from_numpy(fourier[:, None, :] * get_window("hann", FFT, fftbins=True))
    out["target_mel"] = flat(stft.mel_spectrogram(voice_tensor[None]).squeeze(0).T)
    out["pitch"] = [float(v) for v in feats.Pitch(sr=SR, hop_length=HOP)._calculate_pitch(voice.astype(np.float64))]
    # Precision patch: _calculate_energy casts the recording to float32; the same formula runs on it in float64.
    energy = feats.Energy(sr=SR, n_fft=FFT, hop_length=HOP, win_length=FFT)
    power = np.abs(energy._stft(voice.astype(np.float64))) ** 2
    out["energy"] = [float(v) for v in np.sqrt(np.clip(np.sum(power, axis=0), a_min=1.0e-10, a_max=float("inf")))]
    frac = 0.005
    w = torch.from_numpy(voice)
    s = torch.where(torch.abs(w) > (torch.abs(w).max() * frac))[0][0]
    e = torch.where(torch.abs(w) > (torch.abs(w).max() * frac))[0][-1]
    out["trimmed"] = flat(torch.nn.functional.pad(w[s:e], (SR // 20, SR // 20)))
    return out


def bert_pooled(state=None):
    from transformers import BertConfig, BertModel
    torch.manual_seed(31)
    model = BertModel(BertConfig(**BERT))
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.eval().double()
    with torch.no_grad():
        return model, flat(model(torch.tensor([STYLE_IDS]))["pooler_output"])


PROMPTS = ["", "开心", "非常开心", "语速很快，音调很高！", "Happy, a bit SAD 123."]


def prompt_tokens(simbert):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast(vocab_file=os.path.join(simbert, "vocab.txt"))
    return [tokenizer(p)["input_ids"] for p in PROMPTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotivoice", required=True, help="a checkout of netease-youdao/EmotiVoice")
    parser.add_argument("--simbert", required=True, help="a directory with SimBERT's vocab.txt (WangZeJun/simbert-base-chinese)")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    simbert = os.path.abspath(args.simbert)
    loaded = load(args.emotivoice)
    mel, pitch, energy, style, content, wave, voice = inputs()
    if args.write:
        state = {k: v.float().clone().contiguous() for k, v in build(loaded).state_dict().items()}
        bert, _ = bert_pooled()
        bstate = {k: v.float().clone().contiguous() for k, v in bert.state_dict().items() if not k.endswith("position_ids")}
        _, pooled = bert_pooled(bstate)
        data = {"config": CONFIG, "bert": BERT, "phonemes": PHONEMES, "speaker": SPEAKER, "frames": FRAMES, "style_ids": STYLE_IDS,
                "sr": SR, "fft": FFT, "hop": HOP,
                "mel": flat(mel), "pitch_frames": flat(pitch), "energy_frames": flat(energy), "style": flat(style),
                "content": flat(content), "real_wave": flat(wave), "voice": [float(v) for v in voice],
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "bert_safetensors_base64": base64.b64encode(save_safetensors({"bert." + k: v for k, v in bstate.items()})).decode("ascii"),
                "pooled": pooled, "prompts": PROMPTS, "prompt_tokens": prompt_tokens(simbert), **run(loaded, build(loaded, state).double())}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    again = run(loaded, build(loaded, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double())
    check = _fixture.Comparison("EmotiVoice")
    for key, value in again.items():
        if key in ("durations", "segment_start"):
            check.exact(key, data[key], value)
        else:
            check.floats(key, data[key], value, 1e-12, float)
    bstate = {k[5:]: v for k, v in load_safetensors(base64.b64decode(data["bert_safetensors_base64"])).items()}
    check.floats("pooled", data["pooled"], bert_pooled(bstate)[1], 1e-12, float)
    check.exact("prompt_tokens", data["prompt_tokens"], prompt_tokens(simbert))
    check.report(exact=False)


if __name__ == "__main__":
    main()
