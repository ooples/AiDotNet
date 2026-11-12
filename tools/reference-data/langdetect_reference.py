"""Builds the embedded langdetect 1.0.9 data (src/TextToSpeech/FrontEnd/Resources/langdetect_profiles.json.gz) from the
installed package, and its reference fixture.

The resource holds the 55 language profiles (name, n_words, freq) in sorted file order and the messages.properties strings
the n-gram normalizer reads, decoded as langdetect decodes them, and the ranges of the Unicode blocks it tests. The profile order is the factory's language order;
langdetect takes it from os.listdir, which is the file system's, so the reference sorts it (as on Windows).

The fixture (tests/AiDotNet.Tests/TextToSpeech/ReferenceData/langdetect_reference.json) holds, for texts in many scripts,
the n-grams the detector extracts and get_probabilities() with seeds 0 and 7 (the detector's random.Random seeded, as
DetectorFactory.seed sets it).

Usage: python langdetect_reference.py [--write]  (default: verify the resource and fixture). Needs langdetect 1.0.9; the
megatts-ref Docker image has it:
  docker run --rm -v <repo>:/work -w /work megatts-ref python tools/reference-data/langdetect_reference.py --write
"""
import argparse
import gzip
import io
import json
import os
import sys

import langdetect
from langdetect import detector_factory
from langdetect.detector_factory import DetectorFactory
from langdetect.utils import messages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch  # noqa: F401  (_fixture records its version)
except ImportError:  # the langdetect environment has no torch
    import types
    sys.modules["torch"] = types.SimpleNamespace(__version__="not installed")
import _fixture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCE = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources", "langdetect_profiles.json.gz")
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/langdetect_reference.json"
TEXTS = [
    "Hello world, this is MegaTTS 3.",
    "你好，世界！这是一个测试。",
    "欢迎收听《TED Talks Daily》，在这里，我们每天为您带来新思想。",
    "Washington CNN When President Donald Trump declared in the House Chamber this week",
    "Ceci est une phrase en français, avec des accents: été, très, où.",
    "Dies ist ein deutscher Satz über Größe und Übermut.",
    "これは日本語のテキストです。カタカナとひらがな。",
    "한국어 텍스트입니다. 안녕하세요!",
    "Это русский текст для проверки.",
    "Đây là một câu tiếng Việt có dấu.",
    "Tiếng Việt với dấu tách: Tiếng Việt",
    "Visit https://example.com/path?x=1 or mail me at someone@example.org now please",
    "ALL CAPS WORDS SKIP NGRAMS but lowercase words count",
    "Şi România, ţară, with comma-below ș and ț letters.",
    "مرحبا بالعالم یک متن فارسی",
    "Ⓐ circled Ⓑ letters and Ⅻ numerals with mixed text here",
    "Mixed 中英文 text with 123 numbers… and “quotes”.",
    "a",
]


def build():
    directory = detector_factory.PROFILES_DIRECTORY
    profiles = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            data = json.load(f)
        profiles.append({"name": data["name"], "n_words": data["n_words"], "freq": data["freq"]})
    m = messages.Messages()
    # The blocks the n-gram normalizer tests (its table has no overlaps, so a character is in a block when in its range).
    from langdetect.utils import unicode_block as ub
    names = ["BASIC_LATIN", "LATIN_1_SUPPLEMENT", "LATIN_EXTENDED_B", "GENERAL_PUNCTUATION", "ARABIC", "LATIN_EXTENDED_ADDITIONAL",
             "HIRAGANA", "KATAKANA", "BOPOMOFO", "BOPOMOFO_EXTENDED", "CJK_UNIFIED_IDEOGRAPHS", "HANGUL_SYLLABLES"]
    table = ub._unicode_blocks
    assert all(table[i][2] < table[i + 1][1] for i in range(len(table) - 1))
    blocks = {n: [[s, e] for (b, s, e) in table if b == getattr(ub, "UNICODE_" + n)][0] for n in names}
    content = json.dumps({"profiles": profiles, "messages": m.messages, "blocks": blocks}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as g:
        g.write(content)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    assert langdetect.__file__ and os.path.isdir(detector_factory.PROFILES_DIRECTORY)
    resource = build()

    # The factory in sorted file order.
    real_listdir = os.listdir
    os.listdir = lambda p: sorted(real_listdir(p))
    factory = DetectorFactory()
    factory.load_profile(detector_factory.PROFILES_DIRECTORY)
    os.listdir = real_listdir
    cases = []
    for text in TEXTS:
        case = {"text": text}
        for seed in (0, 7):
            factory.set_seed(seed)
            d = factory.create()
            d.append(text)
            d.cleaning_text()
            case["ngrams"] = d._extract_ngrams()
            d = factory.create()
            d.append(text)
            try:
                case[f"probabilities_{seed}"] = [[l.lang, l.prob] for l in d.get_probabilities()]
            except Exception as e:  # LangDetectException: no features
                case[f"probabilities_{seed}"] = str(type(e).__name__)
        cases.append(case)
    data = {"languages": factory.get_lang_list(), "cases": cases}
    if args.write:
        os.makedirs(os.path.dirname(RESOURCE), exist_ok=True)
        with open(RESOURCE, "wb") as f:
            f.write(resource)
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", RESOURCE, len(resource), "bytes and", FIXTURE, [c["probabilities_0"][0] if isinstance(c["probabilities_0"], list) and c["probabilities_0"] else c["probabilities_0"] for c in cases])
        return
    with open(RESOURCE, "rb") as f:
        assert f.read() == resource, "the resource differs from the package's data"
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("langdetect")
    check.exact("languages", stored["languages"], data["languages"])
    for a, b in zip(stored["cases"], data["cases"]):
        check.exact("ngrams", a["ngrams"], b["ngrams"])
        for seed in (0, 7):
            pa, pb = a[f"probabilities_{seed}"], b[f"probabilities_{seed}"]
            if isinstance(pa, str) or isinstance(pb, str):
                check.exact("probabilities", pa, pb)
            else:
                check.exact("languages", [x[0] for x in pa], [x[0] for x in pb])
                check.floats("probabilities", [x[1] for x in pa], [x[1] for x in pb], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
