"""OuteTTS 1.0 reference: the release's own code (edwko/OuteTTS, outetts/version/v3, outetts/dac, outetts/utils,
outetts/models/hf_model.py) for every deterministic part around the language model, run in float64.

Pass a checkout with --outetts. Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_reference.json:
- loudness: dac.interface.process_audio_tensor (pyloudnorm's BS.1770 integrated loudness to -18 LUFS, then a -1 dBFS
  peak limit) of a few signals, short ones padded to the 400 ms block, and the measured loudness;
- word features: audio_processor.Features.extract_audio_features (RMS energy, spectral centroid and the autocorrelation
  pitch, each scaled to 0-100) of several segments, with calculate_pitch's per-frame values;
- prompts: PromptProcessor.text_normalizations (ftfy, NFKC, quotes, dashes, spacing) of a corpus, merge_speaker_text,
  get_completion_prompt with the bundled speaker (en-female-1-neutral) and get_training_prompt of it;
- chunks: utils.chunking.chunk_text of English, Japanese, Chinese and mixed text (MeCab with unidic-lite for CJK);
- sampling: the logits processors generate() applies with the release's settings (OuteTTS's 64-token windowed
  repetition penalty 1.1, temperature 0.4, top-k 40, top-p 0.9, min-p 0.05) to random scores;
- the DAC interface with a tiny randomly initialized descript DAC (weights stored in its own layout, weight_g/weight_v):
  encode (loudness normalization, 5 s windows padded to the hop) of 6 s of audio, and decode (2048-frame chunks, 15 ms
  fades, loudness normalization) of the codes.
Precision patches: pyloudnorm and the features read the audio in float64 (the release passes float32 arrays). Default:
verify; --write: regenerate. Needs the .venv-outetts environment (ftfy, pyloudnorm, mecab-python3, unidic-lite,
descript-audio-codec, transformers 4.52.3).
"""
import argparse
import base64
import json
import math
import os
import sys

import numpy as np
import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_reference.json"
DAC_CONFIG = dict(encoder_dim=4, encoder_rates=[2, 4], latent_dim=8, decoder_dim=8, decoder_rates=[4, 2], n_codebooks=2,
                  codebook_size=16, codebook_dim=4, quantizer_dropout=0.0, sample_rate=24000)
TEXTS = [
    "Hello, world!   This is   a test...",
    "“Smart quotes” – and ‘apostrophes’ — dashes―here",
    "He said: can 't , won 't and it ' s fine .",
    "What?!! Really??? Yes!!!",
    "Ｆｕｌｌ　ｗｉｄｔｈ and ﬁ ligature, Ã© mojibake",
    "Tabs\tand\nnewlines\r\nhere",
    "Soft­hyphen and zero​width and \x07 control",
    "Quotes \"inside\" text and «guillemets»",
    "こんにちは。元気ですか？",
    "",
]
CHUNKS = [
    "This is the first sentence. This is the second one, which is a bit longer than the first! Is this the third? "
    "Here comes a really long sentence that keeps going and going with many words so that it must be split into more "
    "than one part because it is longer than the maximum allowed number of words in a single chunk of text for the model. "
    "Short end.",
    "今日はいい天気ですね。明日も晴れるでしょう！吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "我来到北京清华大学。他光着双脚，脚趾头怪模怪样地向外翻着。",
    "Mixed text with 日本語 inside. And more English words follow here to fill the chunk up nicely.",
    "One two three four five six seven eight nine. Ten eleven twelve. Thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive twentysix.",
]


def load(root):
    # The package's __init__ imports every backend (llama.cpp, ExLlamaV2, vLLM); bind the package to its directory so
    # only the modules used here load.
    import types
    package = types.ModuleType("outetts")
    package.__path__ = [os.path.join(root, "outetts")]
    sys.modules["outetts"] = package
    from outetts.version.v3.prompt_processor import PromptProcessor
    from outetts.version.v3 import audio_processor
    from outetts.dac import interface as dac_interface
    from outetts.utils import chunking
    return PromptProcessor, audio_processor, dac_interface, chunking


def flat(t):
    return [float(v) for v in torch.as_tensor(t).detach().reshape(-1).tolist()]


def signals():
    sr = 24000
    t = np.arange(int(1.5 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 3100 * t)
    rng = np.random.default_rng(5)
    noise = 0.02 * rng.standard_normal(int(0.9 * sr))
    short = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(0.2 * sr)) / sr)
    spikes = np.zeros(int(1.0 * sr))
    spikes[::4000] = 0.9
    spikes += 0.001 * np.sin(2 * np.pi * 100 * np.arange(len(spikes)) / sr)
    return {"tone": tone, "noise": noise, "short": short, "spikes": spikes}


def loudness(dac_interface):
    import pyloudnorm as pyln
    out = {}
    for name, x in signals().items():
        audio = torch.from_numpy(x).double()
        padded = x if len(x) >= int(0.4 * 24000) else np.pad(x, (0, int(0.4 * 24000) - len(x)))
        measured = pyln.Meter(24000, block_size=0.4).integrated_loudness(padded)
        out[name] = {"input": [float(v) for v in x], "lufs": float(measured),
                     "output": flat(dac_interface.process_audio_tensor(audio))}
    return out


def features(audio_processor):
    sr = 24000
    f = audio_processor.Features("cpu")
    rng = np.random.default_rng(9)
    segments = {
        "voiced": 0.4 * np.sin(2 * np.pi * 180 * np.arange(6000) / sr) + 0.1 * np.sin(2 * np.pi * 360 * np.arange(6000) / sr),
        "noise": 0.05 * rng.standard_normal(4000),
        "short": 0.2 * np.sin(2 * np.pi * 300 * np.arange(300) / sr),
        "high": 0.3 * np.sin(2 * np.pi * 520 * np.arange(8000) / sr),
        "empty": np.zeros(0),
    }
    out = {}
    for name, x in segments.items():
        audio = torch.from_numpy(x).double().reshape(1, -1)
        entry = {"input": [float(v) for v in x], "features": f.extract_audio_features(audio, sr)}
        if len(x) > 0:
            entry["pitch_frames"] = flat(audio_processor.calculate_pitch(audio, sr))
            entry["pitch"] = float(audio_processor.extract_single_pitch_value(audio, sr))
        out[name] = entry
    return out


def prompts(PromptProcessor, root):
    p = PromptProcessor(None)
    from outetts.version.v3.tokens import SpecialTokens
    p.special_tokens = SpecialTokens()
    with open(os.path.join(root, "outetts/version/v3/default_speakers/json/en-female-1-neutral.json"), encoding="utf-8") as f:
        speaker = json.load(f)
    normalized = [p.text_normalizations(t) for t in TEXTS]
    merged = [list(p.merge_speaker_text(a, b)) for a, b in [("Hello there.", "The cat sat"), ("Hi", "Ends with period."),
                                                             ("こんにちは", "日本語です"), ("こんにちは", "日本語です。"),
                                                             ("Second", "First?"), ("x", "")]]
    completion = p.get_completion_prompt("Hello, how are you doing today?", json.loads(json.dumps(speaker)))
    plain = p.get_completion_prompt("No speaker here.", None)
    training = p.get_training_prompt(json.loads(json.dumps(speaker)))
    return {"speaker": speaker, "texts": TEXTS, "normalized": normalized, "merged": merged, "completion": completion,
            "plain": plain, "training": training}


def chunks(chunking):
    return [{"text": t, "chunks": chunking.chunk_text(t)} for t in CHUNKS]


def sampling():
    from transformers.generation.logits_process import (MinPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper,
                                                         TopPLogitsWarper)
    from outetts.models.hf_model import RepetitionPenaltyLogitsProcessorPatch
    g = torch.Generator().manual_seed(3)
    cases = []
    for vocab, length in ((60, 10), (60, 90), (200, 40)):
        scores = torch.randn(1, vocab, generator=g, dtype=torch.float64) * 3
        ids = torch.randint(0, vocab, (1, length), generator=g)
        x = scores.clone()
        x = RepetitionPenaltyLogitsProcessorPatch(1.1)(ids, x)
        after_penalty = x.clone()
        for warper in (TemperatureLogitsWarper(0.4), TopKLogitsWarper(40), TopPLogitsWarper(0.9), MinPLogitsWarper(0.05)):
            x = warper(ids, x)
        cases.append({"scores": flat(scores), "ids": ids.reshape(-1).tolist(), "penalized": flat(after_penalty),
                      "processed": [v if math.isfinite(v) else None for v in flat(x)]})
    return cases


def dac_part(dac_interface, state=None):
    import dac
    torch.manual_seed(17)
    model = dac.DAC(**DAC_CONFIG)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model = model.double().eval()
    codec = dac_interface.DacInterface.__new__(dac_interface.DacInterface)
    codec.device = torch.device("cpu")
    codec.model = model
    codec.sr = 24000
    sr = 24000
    t = np.arange(6 * sr) / sr
    x = 0.3 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) + 0.02 * np.sin(2 * np.pi * 2900 * t)
    audio = torch.from_numpy(x).double().reshape(1, 1, -1)
    with torch.no_grad():
        codes = codec.encode(audio)
        decoded = codec.decode(codes)
    return model, {"input": [float(v) for v in x], "codes": codes.reshape(codes.shape[1], -1).tolist(), "decoded": flat(decoded)}


def compute(root, dac_state=None):
    PromptProcessor, audio_processor, dac_interface, chunking = load(root)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        data = {"loudness": loudness(dac_interface), "features": features(audio_processor),
                "prompts": prompts(PromptProcessor, root), "chunks": chunks(chunking), "sampling": sampling()}
        model, data["dac"] = dac_part(dac_interface, dac_state)
    finally:
        torch.set_default_dtype(previous)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outetts", required=True, help="a checkout of edwko/OuteTTS")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.outetts)
    if args.write:
        import dac
        torch.manual_seed(17)
        initial = {k: v.float().clone().contiguous() for k, v in dac.DAC(**DAC_CONFIG).state_dict().items()}
        _, data = compute(root, initial)
        from safetensors.torch import save as save_safetensors
        data["dac_config"] = DAC_CONFIG
        data["dac_safetensors_base64"] = base64.b64encode(save_safetensors(initial)).decode("ascii")
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    stored = _fixture.load(FIXTURE)
    from safetensors.torch import load as load_safetensors
    _, data = compute(root, load_safetensors(base64.b64decode(stored["dac_safetensors_base64"])))
    check = _fixture.Comparison("OuteTTS")
    for key in ("prompts", "chunks"):
        check.exact(key, stored[key], data[key])
    for name, entry in data["loudness"].items():
        check.floats("loudness." + name, stored["loudness"][name]["output"], entry["output"], 1e-12, float)
    check.exact("features", {k: v["features"] for k, v in stored["features"].items()}, {k: v["features"] for k, v in data["features"].items()})
    check.exact("dac.codes", stored["dac"]["codes"], data["dac"]["codes"])
    check.floats("dac.decoded", stored["dac"]["decoded"], data["dac"]["decoded"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
