"""GLM-4-Voice's released speech tokenizer (THUDM/glm-4-voice-tokenizer) on a recording.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_speech_tokenizer_realweights_reference.json three seconds
of 16 kHz speech (CosyVoice's zero_shot_prompt.wav, resampled by torchaudio if needed), and what the released
WhisperVQEncoder (loaded as utils.load_quantize_encoder loads it, run in float64) gives for it: the speech tokens and the
encoder output (the quantized states plus their positions). The features are WhisperFeatureExtractor's arithmetic in
float64, as in glm_speech_tokenizer_reference.py.

Usage (default: verify; --write: regenerate):
  python tools/reference-data/glm_speech_tokenizer_realweights_reference.py --glm4voice <GLM-4-Voice checkout> \\
    --tokenizer <glm-4-voice-tokenizer dir> --wav <zero_shot_prompt.wav> [--write]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_speech_tokenizer_realweights_reference.json"


def features_of(audio, extractor):
    from transformers.audio_utils import window_function
    samples = len(audio)
    length = -(-samples // 1280) * 1280
    padded = np.zeros(length)
    padded[:samples] = audio
    window = window_function(extractor.n_fft, "hann")
    reflected = np.pad(padded, (extractor.n_fft // 2, extractor.n_fft // 2), mode="reflect")
    count = length // extractor.hop_length + 1
    framed = np.stack([reflected[i * extractor.hop_length: i * extractor.hop_length + extractor.n_fft] * window for i in range(count)])
    power = np.abs(np.fft.rfft(framed, axis=1)) ** 2
    log_spec = np.log10(np.maximum(1e-10, power @ extractor.mel_filters)).T[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    mask = (np.arange(length // extractor.hop_length) * extractor.hop_length < samples).astype(np.int64)
    return (log_spec + 4.0) / 4.0, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--glm4voice", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--wav")
    args = parser.parse_args()
    sys.path.insert(0, args.glm4voice)
    import glob
    import safetensors
    from speech_tokenizer.configuration_whisper import WhisperVQConfig
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from transformers import WhisperFeatureExtractor
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    if stored:
        audio = np.array(stored["audio"])
    else:
        import torchaudio
        wave, rate = torchaudio.load(args.wav)
        if rate != 16000:
            wave = torchaudio.transforms.Resample(rate, 16000)(wave)
        audio = wave[0, :48000].numpy().astype(np.float64)

    # utils.load_quantize_encoder, on the CPU.
    config = WhisperVQConfig.from_pretrained(args.tokenizer)
    config.quantize_encoder_only = True
    config._attn_implementation = "eager"
    model = WhisperVQEncoder(config)
    state = {}
    for path in glob.glob(os.path.join(args.tokenizer, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                name = key[len("model.encoder."):] if key.startswith("model.encoder.") else key
                if name.startswith("layer_norm"):
                    continue
                if name.startswith("layers") and int(name.split(".")[1]) >= config.quantize_position:
                    continue
                state[name] = f.get_tensor(key)
    model.load_state_dict(state)
    model = model.double().eval()

    features, mask = features_of(audio, WhisperFeatureExtractor(feature_size=128))
    with torch.no_grad():
        out = model(torch.tensor(features)[None], attention_mask=torch.tensor(mask)[None], return_dict=True)
    pooled_mask = mask[::2][::4]
    codes = out.quantized_token_ids[0].tolist()
    tokens = [c for c, m in zip(codes, pooled_mask) if m]
    hidden = out.last_hidden_state[0]
    if args.write:
        _fixture.save(FIXTURE, {"audio": audio.tolist(), "tokens": tokens, "codes": codes,
                                "hidden": [float(v) for v in hidden.reshape(-1).tolist()]}, __file__)
        print("wrote", FIXTURE, len(tokens), "tokens", tokens[:12])
        return
    check = _fixture.Comparison("GLM-4-Voice released speech tokenizer")
    check.exact("tokens", stored["tokens"], tokens)
    check.floats("hidden", stored["hidden"], [float(v) for v in hidden.reshape(-1).tolist()], dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
