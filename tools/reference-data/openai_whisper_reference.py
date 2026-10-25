"""openai-whisper 20240930 reference (MIT): the model, log-mel spectrogram, decoding, DTW word timing and transcribe(),
as OuteTTS 1.0 runs it to time a speaker's words (model.transcribe(audio, word_timestamps=True, language=None)).

A tiny randomly initialized multilingual Whisper with large-v3-turbo's layout (128 mel bins, 100 languages, a 1500-frame
audio context, a 448-token text context; widths 16, two heads, two encoder and two decoder layers) is stored in float32
and run in float64. Precision patches: the decoder's float32 dtype check on the audio features is skipped, the reference casts to float32 in LayerNorm, attention scores, logits and
log-probabilities (Tensor.float(), and transcribe's .to(float32) of each window); those casts are no-ops here, the mel filter bank (float32 data) is used in float64,
the default dtype is float64 (so the STFT's Hann window is), and scaled-dot-product attention is replaced by the reference's own explicit path (qk scaled on both sides). The DTW
keeps its float32 cost matrix, as the port does.

Records in tests/AiDotNet.Tests/Audio/ReferenceData/openai_whisper_reference.json (weights in float16 and audio in
float32, base64): the log-mel spectrogram of 1 s of audio; the encoder's output and the decoder's last-position logits for
one window; language detection; the logit filters (blank and token suppression, timestamp rules) on random scores after
several prefixes; greedy decoding (temperature 0) with timestamps; find_alignment's word timings for English text with
punctuation, for Chinese text (split per character) and for the decoded tokens; add_word_timestamps on two built segments
(punctuation merging, long-word truncation, segment bounds); and transcribe(temperature=0.0, word_timestamps=True) of
7 s of audio with the language detected. Also writes the mel filter bank the port reads,
src/Audio/Whisper/Resources/whisper_mel_filters.gz. Default: verify;
--write: regenerate.
"""
import argparse
import base64
import sys

import numpy as np
import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/openai_whisper_reference.json"
FILTERS = "src/Audio/Whisper/Resources/whisper_mel_filters.gz"
DIMS = dict(n_mels=128, n_audio_ctx=1500, n_audio_state=16, n_audio_head=2, n_audio_layer=2, n_vocab=51866, n_text_ctx=448,
            n_text_state=16, n_text_head=2, n_text_layer=2)


def patch():
    # float64 by default, so the windows and tables the reference builds at run time (torch.hann_window, arange) are too.
    torch.set_default_dtype(torch.float64)
    import whisper.audio as audio
    import whisper.model as model
    original = torch.Tensor.float

    def float_noop(self, *args, **kwargs):
        return self if self.dtype == torch.float64 else original(self, *args, **kwargs)

    torch.Tensor.float = float_noop
    original_to = torch.Tensor.to

    def to_noop(self, *args, **kwargs):
        if self.dtype == torch.float64 and (torch.float32 in args or kwargs.get("dtype") is torch.float32):
            return self
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = to_noop
    model.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)
    model.MultiHeadAttention.use_sdpa = False
    filters = audio.mel_filters

    def filters64(device, n_mels):
        return filters(device, n_mels).double()

    audio.mel_filters = filters64
    import whisper.decoding as decoding

    def features64(self, mel):
        # DecodingTask._get_audio_features without its float16/float32 dtype check.
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            return mel
        return self.model.encoder(mel)

    decoding.DecodingTask._get_audio_features = features64


def build(state=None):
    from whisper.model import ModelDimensions, Whisper
    torch.manual_seed(41)
    m = Whisper(ModelDimensions(**DIMS))
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("positional_embedding") and name.startswith("decoder"):
                p.normal_(0, 0.02)
    if state is not None:
        m.load_state_dict(state, strict=True)
    return m.double().eval()


def signal(seconds, seed):
    sr = 16000
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.3 * t)) + 0.1 * np.sin(2 * np.pi * 1700 * t)
    return (x + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def flat(t):
    return [float(v) for v in torch.as_tensor(t).detach().reshape(-1).tolist()]


def b64(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def runs(mask):
    """Indices where mask is True, as [start, end) runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append([start, i])
            start = None
    if start is not None:
        out.append([start, len(mask)])
    return out


def words(alignment):
    return [{"word": w.word, "tokens": [int(t) for t in w.tokens], "start": float(w.start), "end": float(w.end),
             "probability": float(w.probability)} for w in alignment]


def run(model):
    from whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from whisper.decoding import DecodingOptions, DecodingTask
    from whisper.timing import add_word_timestamps, find_alignment
    from whisper.tokenizer import get_tokenizer
    out = {}
    short = signal(1.0, 1)
    out["short_audio"] = b64(short, "<f4")
    out["short_mel"] = flat(log_mel_spectrogram(torch.from_numpy(short).double(), DIMS["n_mels"]))
    audio = signal(7.0, 2)
    out["audio"] = b64(audio, "<f4")
    mel = log_mel_spectrogram(torch.from_numpy(audio).double(), DIMS["n_mels"], padding=N_SAMPLES)
    segment = pad_or_trim(mel, N_FRAMES)
    en = get_tokenizer(True, num_languages=model.num_languages, language="en", task="transcribe")
    zh = get_tokenizer(True, num_languages=model.num_languages, language="zh", task="transcribe")
    with torch.no_grad():
        features = model.embed_audio(segment[None])
        out["encoder"] = flat(features[0, :40])
        out["encoder_sum"] = float(features.sum())
        tokens = torch.tensor([list(en.sot_sequence) + [en.timestamp_begin, 400, 1100, 220]])
        logits = model.logits(tokens, features)
        out["decoder_tokens"] = tokens[0].tolist()
        out["decoder_last_logits"] = b64(logits[0, -1].numpy(), "<f8")
        _, probs = model.detect_language(features)
        out["language_probs"] = probs[0] if isinstance(probs, list) else probs
    # The logit filters of a decoding task (suppress blank, suppress tokens, timestamp rules) on random scores after
    # several prefixes: the runs of tokens left allowed.
    task = DecodingTask(model, DecodingOptions(temperature=0.0, fp16=False, language="en"))
    filters = []
    ts = en.timestamp_begin
    for sampled in ([], [ts], [ts, 400], [ts, 400, ts + 50], [ts, 400, ts + 50, ts + 50], [ts + 10, 300, 301, ts + 20, ts + 20, 500]):
        prefix = torch.tensor([list(task.initial_tokens) + sampled])
        # Scores from a formula the test computes too: 4 sin(0.618034 i + case).
        scores = torch.from_numpy(4 * np.sin(np.arange(DIMS["n_vocab"]) * 0.618034 + len(filters)))[None]
        filtered = scores.clone()
        for f in task.logit_filters:
            f.apply(filtered, prefix)
        filters.append({"sampled": sampled, "allowed": runs(torch.isfinite(filtered[0]).tolist())})
    out["filters"] = filters
    out["sample_begin"] = task.sample_begin
    options = DecodingOptions(temperature=0.0, fp16=False, language="en")
    result = model.decode(segment, options)
    out["decode"] = {"tokens": result.tokens, "text": result.text, "avg_logprob": result.avg_logprob,
                     "no_speech_prob": result.no_speech_prob, "compression_ratio": result.compression_ratio}
    # Word timing of real text: English with punctuation and Chinese (split per character), then the whole
    # add_word_timestamps on built segments.
    english = en.encode(' Hello, world. "This" is a test - of (punctuated) words! Okay?')
    chinese = zh.encode("\u4f60\u597d\uff0c\u4e16\u754c\u3002\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\uff01")
    out["alignment_en_tokens"] = english
    out["alignment_en"] = words(find_alignment(model, en, english, segment, 700))
    out["alignment_zh_tokens"] = chinese
    out["alignment_zh"] = words(find_alignment(model, zh, chinese, segment, 700))
    decoded = [t for t in result.tokens if t < en.eot][:30]
    out["alignment_decoded_tokens"] = decoded
    out["alignment_decoded"] = words(find_alignment(model, en, decoded, segment, 700))
    half = len(english) // 2
    segments = [{"seek": 0, "start": 0.0, "end": 3.2, "tokens": [ts] + english[:half] + [ts + 160]},
                {"seek": 0, "start": 3.2, "end": 6.9, "tokens": [ts + 160] + english[half:] + [ts + 345]}]
    out["word_segments_in"] = [dict(s) for s in segments]
    add_word_timestamps(segments=segments, model=model, tokenizer=en, mel=segment, num_frames=700, last_speech_timestamp=0.0)
    out["word_timestamps"] = [{"start": s["start"], "end": s["end"], "words": s["words"]} for s in segments]
    transcript = model.transcribe(torch.from_numpy(audio).double(), temperature=0.0, word_timestamps=True, fp16=False, verbose=None)
    out["transcribe"] = {"text": transcript["text"], "language": transcript["language"], "segments": transcript["segments"]}
    return out


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def filters_blob():
    """openai-whisper's assets/mel_filters.npz (MIT): the 80- and 128-bin filter banks [mels, 201] in float32,
    concatenated, gzipped."""
    import gzip
    import io
    import os
    import whisper
    with np.load(os.path.join(os.path.dirname(whisper.__file__), "assets", "mel_filters.npz")) as f:
        raw = np.concatenate([f["mel_80"].astype("<f4").reshape(-1), f["mel_128"].astype("<f4").reshape(-1)]).tobytes()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(raw)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    patch()
    from safetensors.torch import load as load_safetensors
    from safetensors.torch import save as save_safetensors
    if args.write:
        # Stored in float16 (the fixture would otherwise hold 3 MB of token embeddings); both sides run these values.
        initial = {k: v.half().float().clone().contiguous() for k, v in build().state_dict().items()}
        data = clean(run(build(initial)))
        data["dims"] = DIMS
        data["safetensors_base64"] = base64.b64encode(save_safetensors({k: v.half() for k, v in initial.items()})).decode("ascii")
        _fixture.save(FIXTURE, data, __file__)
        with open(_fixture.fixture_path(FILTERS), "wb") as handle:
            handle.write(filters_blob())
        print("wrote " + FIXTURE, data["transcribe"]["language"], repr(data["transcribe"]["text"][:80]))
        return
    stored = _fixture.load(FIXTURE)
    again = clean(run(build({k: v.float() for k, v in load_safetensors(base64.b64decode(stored["safetensors_base64"])).items()})))
    check = _fixture.Comparison("openai-whisper")
    for key in ("short_mel", "encoder"):
        check.floats(key, stored[key], again[key], 1e-12, float)
    for key in ("decoder_last_logits", "decode", "filters", "alignment_en", "alignment_zh", "alignment_decoded", "word_timestamps",
                "transcribe", "language_probs"):
        check.exact(key, stored[key], again[key])
    with open(_fixture.fixture_path(FILTERS), "rb") as handle:
        check.exact("mel filters", handle.read() == filters_blob(), True)
    check.report(exact=False)


if __name__ == "__main__":
    main()
