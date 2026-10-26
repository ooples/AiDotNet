"""MegaTTS 3's aligner through the release's own code (bytedance/MegaTTS3, tts/modules/aligner/whisper_small.py: a
Whisper whose encoder takes any length and whose logits come from a separate projection; tts/frontend_function.py's
align; tts/utils/text_utils/ph_tone_convert.py's split_ph_timestamp), on a tiny model.

Whisper.__init__ hardcodes its sizes (80 mels, a 3000-frame encoder of width 512, 6800 tokens, six layers of eight heads
both ways); the test builds the same modules at width 16 with two heads and two layers (a test-size patch). Random
weights, float64. align() casts the mel to float32 and the model computes in the input's dtype, so the test restates its
loop in float64 (precision patches: the default dtype is float64 for the STFT's window, the filter bank applies in float64,
LayerNorm's float32 cast is skipped, and scaled-dot-product attention is written out); its greedy decoding (from 798 until 799, at most 768 steps, without a cache) is the release's.
Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_aligner_reference.json:
- the log-mel of 0.4 s of 16 kHz noise cut to a multiple of 8 frames, the encoder output and the decoded tokens;
- align()'s post-processing (split_ph_timestamp, then the durations stretched or cut to the prompt's frames, then the
  length regulator) on two token sequences, one short of the frames and one over, through the release's align with the
  model's decoding replaced by the sequence.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-outetts environment (openai-whisper).
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_aligner_reference.json"
WIDTH, HEADS, LAYERS = 16, 2, 2
SEQUENCES = {
    # (40 frames) phones with timestamps: an English phone, a Chinese one (tone 1), one at the low end of the range, one with
    # a timestamp above 800.
    "short": [798, 57, 10, 207, 18, 200, 30, 330, 840, 799],
    "long": [798, 12, 20, 788, 45, 150, 60, 799],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    import whisper
    import tts.frontend_function as frontend
    from tts.modules.aligner import whisper_small as ws
    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator

    # Precision patches: LayerNorm computes in float32; the mel filter bank (float32 values) is applied in float64.
    # scaled_dot_product_attention's float64 CPU kernel returned NaN on finite inputs here (intermittently); the same
    # attention written out.
    def attention(q, k, v, is_causal=False, attn_mask=None):
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        if is_causal:
            n, m = scores.shape[-2:]
            scores = scores.masked_fill(torch.ones(n, m, dtype=torch.bool).triu(1), float("-inf"))
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v
    ws.scaled_dot_product_attention = attention
    filters = whisper.audio.mel_filters
    whisper.audio.mel_filters = lambda device, n_mels: filters(device, n_mels).double()
    ws.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)

    class TinyWhisper(ws.Whisper):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.n_vocab, self.n_text_layer, self.n_text_head, self.n_text_ctx = 6800, LAYERS, HEADS, 2048
            self.encoder = ws.AudioEncoder(n_mels=80, n_ctx=3000, n_state=WIDTH, n_head=HEADS, n_layer=LAYERS)
            self.decoder = ws.TextDecoder(n_vocab=6800, n_ctx=2048, n_state=WIDTH, n_head=HEADS, n_layer=LAYERS)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(151)
    model = TinyWhisper()
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if name == "encoder.positional_embedding":
                continue
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("ln.weight") or name.endswith("ln_post.weight"):
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        state32 = {k: v.float().clone() for k, v in model.state_dict().items() if k != "encoder.positional_embedding"}
    model = model.double().eval()

    g = torch.Generator().manual_seed(157)
    audio = (0.2 * torch.randn(6400, generator=g)).float().double().numpy() if args.write else np.array(stored["audio"], dtype=np.float64)
    fm = 8
    with torch.no_grad():
        # align(): the log-mel cut to a multiple of fm frames, then greedy decoding.
        mel = whisper.log_mel_spectrogram(audio)[None]
        prompt_max_frame = mel.size(2) // fm * fm
        mel = mel[:, :, :prompt_max_frame]
        token = torch.LongTensor([[798]])
        audio_features = model.embed_audio(mel)
        for _ in range(768):
            logits = model.logits(token, audio_features, None)
            token_pred = torch.argmax(torch.softmax(logits[:, -1], dim=-1), 1)[None]
            token = torch.cat([token, token_pred], dim=1)
            if token_pred[0] == 799:
                break
        last = logits[0, -1]

    # The post-processing, through the release's align with the decoding replaced (at the default float32).
    torch.set_default_dtype(torch.float32)
    whisper.audio.mel_filters = filters
    posts = {}
    for key, sequence in SEQUENCES.items():
        class Fixed:
            def embed_audio(self, mel):
                return None

            def logits(self, tokens, features, cache):
                out = torch.full((1, tokens.shape[1], 6800), -1.0)
                out[0, -1, sequence[tokens.shape[1]]] = 1.0
                return out
        fake = types.SimpleNamespace(sr=16000, fm=fm, device="cpu", precision=torch.float32, aligner_lm=Fixed(), length_regulator=LengthRegulator())
        wav48 = np.zeros(48 * 160 - 1, dtype=np.float32)
        ph, tone, mel2ph = frontend.align(fake, wav48)
        frames = whisper.log_mel_spectrogram(wav48).shape[1] // fm * fm
        posts[key] = {"tokens": sequence, "frames": frames, "ph": ph[0].long().tolist(), "tone": tone[0].long().tolist(), "mel2ph": mel2ph[0].tolist()}
    data = {"audio": audio.tolist(), "mel": mel.reshape(-1).tolist(), "frames": prompt_max_frame, "features": audio_features.reshape(-1).tolist(),
            "tokens": token[0].tolist(), "last_logits": last.tolist(), "posts": posts}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "frames", prompt_max_frame, "tokens", len(data["tokens"]), data["tokens"][:12], {k: (v["ph"], v["mel2ph"][-3:]) for k, v in posts.items()})
        return
    check = _fixture.Comparison("MegaTTS 3 aligner")
    for key in ("mel", "features", "last_logits"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    check.exact("tokens", stored["tokens"], data["tokens"])
    check.exact("posts", stored["posts"], data["posts"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
