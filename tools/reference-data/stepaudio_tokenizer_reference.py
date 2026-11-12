"""Step-1's tokenizer reference (Step-Audio, stepfun-ai/Step-Audio-TTS-3B tokenizer.model): the released SentencePiece
unigram model (74,752 pieces, digits split, byte fallback, a dummy prefix, and the 9,216 audio tokens <audio_N> from id
65536 as user-defined pieces), embedded with its encodings and decodings of a few strings: Chinese, English, digits,
characters outside the vocabulary, the TTS prompt's role markers and audio tokens.

Usage: python tools/reference-data/stepaudio_tokenizer_reference.py --model <tokenizer.model> [--write]
"""
import argparse
import base64
import os
import sys

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/stepaudio_tokenizer_reference.json"
TEXTS = ["你好，世界！今天天气怎么样？", "Hello, I'm Step-Audio.", "The year 2025 had 365 days.", "(高兴1)我们去公园吧", "system\n作为一名卓越的声优演员",
         "<audio_5><audio_1027><audio_70>hello", "naïve café 😀", "  spaced   out  ", "human\n", "assistant\n"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)
    blob = base64.b64decode(stored["model_base64"]) if stored else open(args.model, "rb").read()
    processor = spm.SentencePieceProcessor(model_proto=blob)
    cases = [{"text": t, "ids": processor.encode(t), "decoded": processor.decode(processor.encode(t))} for t in TEXTS]
    if args.write:
        _fixture.save(FIXTURE, {"model_base64": base64.b64encode(blob).decode("ascii"), "cases": cases}, __file__)
        print("wrote", FIXTURE, [len(c["ids"]) for c in cases])
        return
    check = _fixture.Comparison("Step-1 tokenizer")
    if stored["cases"] != cases:
        check.failures.append("cases differ")
    check.report(exact=True)


if __name__ == "__main__":
    main()
