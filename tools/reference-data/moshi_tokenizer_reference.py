"""Helium's tokenizer reference (Moshi §3.2.1): the released SentencePiece unigram model tokenizer_spm_32k_3.model
(kyutai/moshiko-pytorch-bf16; 32,000 pieces, digits split, byte fallback), embedded with its encodings and decodings of a
few strings, among them characters outside the vocabulary that fall back to bytes.

Usage: python tools/reference-data/moshi_tokenizer_reference.py --model <tokenizer_spm_32k_3.model> [--write]
"""
import argparse
import base64
import os
import sys

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_tokenizer_reference.json"
TEXTS = ["Hello, I'm Moshi.", "The year 2024 had 366 days.", "naïve café — déjà vu", "你好，世界 😀", "  spaced   out  ",
         "Zero-shot TTS synthesis.", "e-mail: moshi@kyutai.org\ttab"]


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
    check = _fixture.Comparison("Helium tokenizer")
    if stored["cases"] != cases:
        check.failures.append("cases differ")
    check.report(exact=True)


if __name__ == "__main__":
    main()
