"""espeak-ng phonemization of the CMU ARCTIC prompts, as Pheme's front end produces it.

Pheme (PolyAI-LDN/pheme, data/semantic_dataset.py, TextTokenizer) phonemizes text with phonemizer's espeak backend
for en-us: punctuation preserved, no stress marks, no ties, language switches kept, word mismatches ignored, and the
separators phone "|" and word "_". Its to_list then splits each word into phones on "|", keeps punctuation as its own
symbols and puts "_" between words. This script does the same for every prompt of CMU ARCTIC (Kominek and Black
2003, Carnegie Mellon University, distributed under the Festvox licence; the sentences come from public-domain Project
Gutenberg texts) and writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/espeak_arctic_phonemes.json, which the C#
English G2P is measured against.

Default: verify the committed file against this machine's espeak; --write: regenerate it. Needs the ARCTIC prompt
list (cmu_us_*_arctic/etc/txt.done.data) as --prompts.
"""
import argparse
import json
import re

import espeakng_loader
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/espeak_arctic_phonemes.json"
SEPARATOR = Separator(word="_", syllable="-", phone="|")


def to_list(phonemized):
    """Pheme's TextTokenizer.to_list."""
    fields = []
    for word in phonemized.split(SEPARATOR.word):
        pieces = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
        fields.extend([p for p in pieces if p != SEPARATOR.phone] + [SEPARATOR.word])
    return fields[:-1]


def read_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            m = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)\s*$', line.strip())
            if m:
                prompts.append((m.group(1), m.group(2)))
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="the ARCTIC txt.done.data prompt list")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    backend = EspeakBackend("en-us", punctuation_marks=Punctuation.default_marks(), preserve_punctuation=True,
                            with_stress=False, tie=False, language_switch="keep-flags", words_mismatch="ignore")
    prompts = read_prompts(args.prompts)
    phonemized = backend.phonemize([text for _, text in prompts], separator=SEPARATOR, strip=True, njobs=1)
    items = [{"id": pid, "text": text, "phones": to_list(p)} for (pid, text), p in zip(prompts, phonemized)]
    data = {"espeak_ng": ".".join(str(v) for v in backend.version()), "voice": "en-us",
            "source": "CMU ARCTIC prompts (Kominek and Black 2003)", "items": items}

    if args.write:
        with open(_fixture.fixture_path(FIXTURE), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=0)
        print(f"wrote {FIXTURE}: {len(items)} prompts")
        return
    committed = _fixture.load(FIXTURE)
    check = _fixture.Comparison("espeak ARCTIC phonemes")
    check.exact("espeak_ng", committed["espeak_ng"], data["espeak_ng"])
    check.exact("items", committed["items"], items)
    check.report()


if __name__ == "__main__":
    main()
