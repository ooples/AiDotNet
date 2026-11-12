"""Mega-TTS's acoustic model and prosody language model, composed from the authors' modules, on tiny sizes.

Mega-TTS (Jiang et al. 2023) has no public code. This script composes the model of its Sec. 3 from the modules of the
authors' NATSpeech code base (FastSpeechEncoder, ConvBlocks, DurationPredictor, TransformerDecoderLayer, its mel features
and its duration loss) and SyntaSpeech's multi-window discriminator, and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts_model_reference.json, in float64 on float32 weights:
- NATSpeech's log10 mel spectrogram (librosa_wav2spec: centred STFT with zero padding, Slaney mel, 10 bins from 80 to
  7600 Hz) of a two-tone recording;
- the prosody encoder (the first 6 mel bins, a linear prenet, a frame-level ConvBlocks stack, the mean over each phoneme's
  frames, a phoneme-level ConvBlocks stack), its nearest codes and the VQ loss (codebook and commitment terms, Eq. 1);
- the timbre vector (a linear prenet, a kernel-7 ConvBlocks stack, the time average) of a reference mel spectrogram;
- the decoded mel spectrogram (content plus prosody expanded by the durations, plus the timbre vector, through ConvBlocks);
- the duration predictor on content plus prosody, and NATSpeech's add_dur_loss (phoneme, word and sentence terms, with
  lambda_ph_dur 0.1);
- the L2 mel loss, and the generator's LSGAN term of the discriminator (windows 8 and 16, evaluation mode) at given starts;
- the P-LLM's logits (code embeddings plus projected content and timbre plus sinusoidal positions from 1, causal decoder
  layers, a layer norm and the head) and its cross-entropy on the codes.
Default: verify; --write: regenerate. Pass --natspeech and --syntaspeech (checkouts). Needs torch and librosa.
"""
import argparse
import importlib.util
import math
import os
import sys
import types

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts_model_reference.json"

# NatSpeechFrontEnd.EnglishPhones(): g2p_en's ARPAbet with stress, the punctuation, "|" and <BOS>.
ARPABET = ("AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 "
           "EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 "
           "V W Y Z ZH").split()
VOCABULARY = ["<pad>", "<EOS>", "<UNK>"] + sorted(set(ARPABET + list("!,.?;:") + ["|", "<BOS>"]))
PHONES = ["<BOS>", "HH", "AH0", "L", "OW1", "|", "W", "ER1", "L", "D", ",", "<EOS>", "<pad>"]
DURATIONS = [1, 2, 3, 1, 4, 0, 2, 3, 2, 1, 3, 2, 0]
SR, FFT, HOP, MELS, BINS = 22050, 1024, 256, 10, 6
H, CODES, CODE_DIM, LM_H = 16, 12, 8, 16


def is_sil(p):
    return p == "" or not p[0].isalpha()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natspeech", required=True)
    parser.add_argument("--syntaspeech", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.natspeech))
    from modules.commons.transformer import FastSpeechEncoder, TransformerDecoderLayer
    from modules.commons.nar_tts_modules import DurationPredictor
    from modules.commons.conv import ConvBlocks
    # utils/audio/align.py by path: the package's __init__ imports pyloudnorm, which the function does not need.
    align_spec = importlib.util.spec_from_file_location("natspeech_align", os.path.join(args.natspeech, "utils", "audio", "align.py"))
    align = importlib.util.module_from_spec(align_spec)
    sys.modules.setdefault("textgrid", types.SimpleNamespace(TextGrid=None))   # read only by the TextGrid parser
    align_spec.loader.exec_module(align)
    mel2token_to_dur = align.mel2token_to_dur
    import modules.commons.transformer as natt

    # Precision patch: the sinusoidal positions are built in float32 (an explicit dtype); the same in float64.
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    natt.SinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding)
    spec = importlib.util.spec_from_file_location("mwd", os.path.join(args.syntaspeech, "modules", "tts", "syntaspeech", "multi_window_disc.py"))
    mwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mwd)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(229)

    def blocks(c_in, c_out, kernel):
        return ConvBlocks(c_in, c_out, None, kernel, num_layers=2, layers_in_block=2, c_multiple=2, ln_eps=1e-5, post_net_kernel=3)

    m = torch.nn.ModuleDict({
        "content": FastSpeechEncoder(len(VOCABULARY), hidden_size=H, num_layers=2, kernel_size=5, num_heads=2),
        "prosody_prenet": torch.nn.Linear(BINS, H),
        "prosody_frames": blocks(H, H, 5),
        "prosody_phones": blocks(H, CODE_DIM, 5),
        "codebook": torch.nn.Embedding(CODES, CODE_DIM),
        "prosody_out": torch.nn.Linear(CODE_DIM, H),
        "timbre_prenet": torch.nn.Linear(MELS, H),
        "timbre": blocks(H, H, 7),
        "duration": DurationPredictor(H, n_layers=3, n_chans=H, kernel_size=3, dropout_rate=0.0),
        "decoder": blocks(H, MELS, 5),
        "code_embed": torch.nn.Embedding(CODES + 2, LM_H),
        "lm_content": torch.nn.Linear(H, LM_H),
        "lm_timbre": torch.nn.Linear(H, LM_H),
        "lm_layers": torch.nn.ModuleList([TransformerDecoderLayer(LM_H, 0.0, kernel_size=5, num_heads=2) for _ in range(2)]),
        "lm_norm": torch.nn.LayerNorm(LM_H),
        "lm_head": torch.nn.Linear(LM_H, CODES + 2),
        "disc": mwd.MultiWindowDiscriminator([8, 16], freq_length=MELS, hidden_size=6),
    })
    with torch.no_grad():
        for name, p in m.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("weight") and p.dim() == 1:
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        for name, b in m.named_buffers():
            if "running" not in name:
                continue
            if stored is not None:
                b.copy_(torch.tensor(stored["state"][name]).reshape(b.shape))
            else:
                b.copy_(((torch.rand_like(b) + 0.5) if "var" in name else 0.1 * torch.randn_like(b)).float())
        state32 = {n: v.float().clone() for n, v in m.named_parameters()}
        state32.update({n: v.float().clone() for n, v in m.named_buffers() if "running" in n})
    m.double().eval()

    g = torch.Generator().manual_seed(233)
    frames = sum(DURATIONS)
    if stored is None:
        mel = torch.randn(frames, MELS, generator=g) - 2
        reference = torch.randn(9, MELS, generator=g) - 2
        previous = [CODES] + torch.randint(0, CODES, (12,), generator=g).tolist()
        targets = previous[1:] + [int(torch.randint(0, CODES, (1,), generator=g))]
        starts = [int(torch.randint(0, frames - 8 + 1, (1,), generator=g)), int(torch.randint(0, frames - 16 + 1, (1,), generator=g))]
    else:
        mel = torch.tensor(stored["mel"]).reshape(frames, MELS)
        reference = torch.tensor(stored["reference"]).reshape(9, MELS)
        previous, targets, starts = stored["previous"], stored["targets"], stored["starts"]
    tokens = torch.tensor([[VOCABULARY.index(p) for p in PHONES]])
    L = tokens.shape[1]
    durations = torch.tensor([DURATIONS])
    mel2ph = torch.repeat_interleave(torch.arange(1, L + 1), durations[0])[None]

    # NATSpeech's librosa_wav2spec on 3000 samples of two tones.
    n = np.arange(3000)
    wav = 0.4 * np.sin(2 * np.pi * 220 * n / SR) + 0.2 * np.sin(2 * np.pi * 1330 * n / SR)
    stft = librosa.stft(wav, n_fft=FFT, hop_length=HOP, win_length=FFT, window="hann", pad_mode="constant")
    basis = librosa.filters.mel(sr=SR, n_fft=FFT, n_mels=MELS, fmin=80, fmax=7600, dtype=np.float64)
    features = np.log10(np.maximum(1e-6, basis @ np.abs(stft))).T

    with torch.no_grad():
        ones = lambda t: torch.ones(1, t, 1)
        content = m["content"](tokens)
        # The prosody encoder: the low band through the frame stack, each phoneme's mean (as weights 1/count), the phone stack.
        low = mel[None, :, :BINS]
        hf = m["prosody_frames"](m["prosody_prenet"](low), nonpadding=ones(frames))
        counts = torch.bincount(mel2ph[0], minlength=L + 1)[1:]
        pool = torch.zeros(1, L, frames)
        for f in range(frames):
            pool[0, mel2ph[0, f] - 1, f] = 1.0 / counts[mel2ph[0, f] - 1]
        phone_mask = (counts > 0).double()[None, :, None]
        hidden = m["prosody_phones"](pool @ hf, nonpadding=phone_mask)
        # VQ: the nearest codebook entry, straight through; codebook and commitment terms over the voiced phonemes.
        table = m["codebook"].weight
        distance = ((hidden[0, :, None, :] - table[None]) ** 2).sum(-1)
        codes = distance.argmin(-1)
        zq = table[codes][None]
        mask = phone_mask.expand_as(hidden)
        count = mask.sum()
        vq = ((((hidden - zq) * mask) ** 2).sum() + (((zq - hidden) * mask) ** 2).sum()) * (1.0 / count)
        quantized = hidden + (zq - hidden)
        prosody = m["prosody_out"](quantized)
        timbre = m["timbre"](m["timbre_prenet"](reference[None]), nonpadding=ones(9)).mean(1)
        # The decoder: content plus prosody expanded to the frames, plus the timbre vector.
        expand = torch.zeros(1, frames, L)
        for f in range(frames):
            expand[0, f, mel2ph[0, f] - 1] = 1.0
        x = expand @ (content + prosody) + timbre[:, None, :]
        decoded = m["decoder"](x, nonpadding=ones(frames))[0]
        dur_input = content + prosody
        dur_input = dur_input.detach() + 0.1 * (dur_input - dur_input.detach())
        dur_pred = m["duration"](dur_input, tokens == 0)
        # NATSpeech's FastSpeechTask.add_dur_loss with lambda_ph_dur 0.1, lambda_word_dur 1 and lambda_sent_dur 1.
        nonpadding = (tokens != 0).double()
        dur_gt = mel2token_to_dur(mel2ph, L).double() * nonpadding
        sil = torch.tensor([[is_sil(VOCABULARY[t]) for t in tokens[0].tolist()]]).double()
        pdur = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction="none")
        pdur = (pdur * nonpadding).sum() / nonpadding.sum() * 0.1
        word_id = (sil.cumsum(-1) * (1 - sil)).long()
        word_dur_p = dur_pred.new_zeros([1, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
        word_dur_g = dur_gt.new_zeros([1, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
        wdur = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
        word_nonpadding = (word_dur_g > 0).double()
        wdur = (wdur * word_nonpadding).sum() / word_nonpadding.sum()
        sdur = F.mse_loss((dur_pred.sum(-1) + 1).log(), (dur_gt.sum(-1) + 1).log(), reduction="mean")
        mel_loss = ((decoded - mel) ** 2).sum() * (1.0 / mel.numel())
        total = mel_loss + vq + (pdur + wdur + sdur)
        validity, _, _ = m["disc"](decoded[None, None], torch.tensor([frames]), start_frames_wins=[[starts[0]], [starts[1]]])
        adversarial = ((validity - 1) ** 2).sum() * (1.0 / validity.numel())
        # The P-LLM: teacher-forced on the codes given (one per phoneme).
        lm_content = content
        lp = len(previous)
        xl = m["code_embed"](torch.tensor(previous))[None] + m["lm_content"](lm_content[:, :lp]) + m["lm_timbre"](timbre)[:, None, :]
        xl = xl + get_embedding(lp + 1, LM_H, 0)[1:][None]
        hl = xl.transpose(0, 1)
        causal = torch.triu(torch.full((lp, lp), float("-inf")), 1)
        for layer in m["lm_layers"]:
            hl, _ = layer(hl, self_attn_mask=causal)
        logits = m["lm_head"](m["lm_norm"](hl.transpose(0, 1)))[0]
        logp = torch.log_softmax(logits, -1)
        ce = -logp[torch.arange(lp), torch.tensor(targets)].sum() * (1.0 / lp)
    torch.set_default_dtype(torch.float32)
    data = {
        "phones": PHONES, "durations": DURATIONS, "mel": mel.reshape(-1).tolist(), "reference": reference.reshape(-1).tolist(),
        "previous": previous, "targets": targets, "starts": starts, "wave": wav.tolist(),
        "features": features.reshape(-1).tolist(), "feature_frames": features.shape[0],
        "content": content.reshape(-1).tolist(), "hidden": hidden.reshape(-1).tolist(), "codes": codes.tolist(),
        "timbre": timbre.reshape(-1).tolist(), "decoded": decoded.reshape(-1).tolist(), "dur_pred": dur_pred.reshape(-1).tolist(),
        "vq": vq.item(), "pdur": pdur.item(), "wdur": wdur.item(), "sdur": sdur.item(), "mel_loss": mel_loss.item(), "total": total.item(),
        "adversarial": adversarial.item(), "logits": logits.reshape(-1).tolist(), "ce": ce.item(),
    }
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, len(state32), "tensors; codes", data["codes"], "total", data["total"])
        return
    check = _fixture.Comparison("Mega-TTS model")
    for key in ("features", "content", "hidden", "timbre", "decoded", "dur_pred", "logits"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    for key in ("vq", "pdur", "wdur", "sdur", "mel_loss", "total", "adversarial", "ce"):
        check.floats(key, [stored[key]], [data[key]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
