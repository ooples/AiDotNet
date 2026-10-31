"""Moshi's inner-monologue text stream (Défossez et al. 2024 §3.4.4, Eq. 5) as the release's fine-tuning code builds it
(kyutai-labs/moshi-finetune, finetune/data/interleaver.py, Interleaver.prepare_item): Helium's tokenizer (embedded in
moshi_tokenizer_reference.json), PAD 3, EPAD 0, the zero token -1, 12.5 frames a second. Cases cover a word at time 0,
words close enough to collide, a zero-duration word, unsorted input, and the TTS (+2 s) and ASR (-2 s) delays.

Usage: python tools/reference-data/moshi_interleaver_reference.py --finetune <moshi-finetune checkout> [--write]
"""
import argparse
import base64
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_interleaver_reference.json"
TOKENIZER = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_tokenizer_reference.json"
FRAME_RATE = 12.5
CASES = [
    {"frames": 40, "delay": 0.0, "words": [["Hello,", 0.0, 0.4], ["I'm", 0.5, 0.7], ["Moshi.", 0.75, 1.3]]},
    {"frames": 30, "delay": 0.0, "words": [["The", 0.32, 0.4], ["year", 0.4, 0.6], ["2024", 0.62, 1.4], ["ended.", 1.9, 2.3]]},
    {"frames": 25, "delay": 0.0, "words": [["zero", 0.5, 0.5], ["synthesis", 0.9, 1.5], ["first", 0.1, 0.3]]},
    {"frames": 60, "delay": 2.0, "words": [["Text", 2.1, 2.4], ["ahead", 2.5, 2.9], ["of", 3.0, 3.1], ["audio.", 3.2, 3.9]]},
    {"frames": 60, "delay": -2.0, "words": [["Audio", 0.2, 0.6], ["ahead", 0.8, 1.2], ["of", 1.3, 1.4], ["text.", 1.5, 2.2]]},
    {"frames": 20, "delay": 0.0, "words": [["naïve", 0.08, 0.5], ["café", 0.52, 0.9]]},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import sentencepiece as spm
    sys.modules.setdefault("moshi.conditioners", types.SimpleNamespace(ConditionAttributes=object))
    sys.path.insert(0, args.finetune)
    from finetune.data.interleaver import Interleaver
    blob = base64.b64decode(_fixture.load(TOKENIZER)["model_base64"])
    tokenizer = spm.SentencePieceProcessor(model_proto=blob)
    results = []
    for case in CASES:
        interleaver = Interleaver(tokenizer, FRAME_RATE, text_padding=3, end_of_text_padding=0, zero_padding=-1,
                                  audio_delay=case["delay"], device="cpu")
        alignments = [(w, (s, e), "SPEAKER_MAIN") for w, s, e in case["words"]]
        stream = interleaver.prepare_item(alignments, case["frames"] / FRAME_RATE).view(-1).tolist()
        results.append({**case, "stream": stream})
    if args.write:
        _fixture.save(FIXTURE, {"cases": results}, __file__)
        print("wrote", FIXTURE, [len(r["stream"]) for r in results])
        return
    check = _fixture.Comparison("Moshi interleaver")
    if _fixture.load(FIXTURE)["cases"] != results:
        check.failures.append("streams differ")
    check.report(exact=True)


if __name__ == "__main__":
    main()
