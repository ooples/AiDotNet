"""Builds the Mandarin G2P's embedded resources from the packages Plachtaa/VALL-E-X's front end runs on.

VALL-E X's runnable reference (Plachtaa/VALL-E-X, utils/g2p/mandarin.py) converts Chinese with jieba (word
segmentation), pypinyin (BOPOMOFO readings) and cn2an (numerals). This script exports what MandarinG2P needs from
the pinned releases, so the C# port reads the same data:

- src/TextToSpeech/FrontEnd/Resources/jieba_dict.tsv.gz: jieba 0.42.1's dict.txt as "word<TAB>frequency" lines, in
  the file's order (MIT, see NOTICE).
- src/TextToSpeech/FrontEnd/Resources/jieba_hmm.json.gz: jieba's finalseg HMM (start, transition and emission log
  probabilities of the B/M/E/S states).
- src/TextToSpeech/FrontEnd/Resources/pypinyin_chars.tsv.gz and pypinyin_phrases.tsv.gz: pypinyin 0.55.0's readings in
  its BOPOMOFO style, computed by pypinyin's own converter (first reading, no heteronyms), one character or phrase per
  line: "text<TAB>bopomofo bopomofo ..." (MIT, see NOTICE).

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import gzip
import importlib.metadata
import io
import json
import os

import _fixture

PINNED = {"jieba": "0.42.1", "pypinyin": "0.55.0", "cn2an": "0.5.24"}
OUT = "src/TextToSpeech/FrontEnd/Resources/"


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def build():
    for name, version in PINNED.items():
        found = importlib.metadata.version(name)
        if found != version:
            raise SystemExit(f"{name} {found} is installed; the resources are built from {version}.")
    import jieba
    import jieba.finalseg.prob_emit as emit
    import jieba.finalseg.prob_start as start
    import jieba.finalseg.prob_trans as trans
    from pypinyin import Style
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    from pypinyin.core import _default_convert

    dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    lines = []
    with open(dict_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(" ")
            if parts and parts[0]:
                lines.append(f"{parts[0]}\t{parts[1]}")
    files = {OUT + "jieba_dict.tsv.gz": gz("\n".join(lines) + "\n")}

    hmm = {"start": start.P, "trans": trans.P, "emit": emit.P}
    files[OUT + "jieba_hmm.json.gz"] = gz(json.dumps(hmm, ensure_ascii=False, sort_keys=True))

    def reading(text):
        return " ".join(r[0] for r in _default_convert.convert(text, Style.BOPOMOFO, False, "default", strict=True))

    chars = [f"{chr(code)}\t{reading(chr(code))}" for code in sorted(PINYIN_DICT)]
    files[OUT + "pypinyin_chars.tsv.gz"] = gz("\n".join(chars) + "\n")
    phrases = [f"{phrase}\t{reading(phrase)}" for phrase in sorted(PHRASES_DICT)]
    files[OUT + "pypinyin_phrases.tsv.gz"] = gz("\n".join(phrases) + "\n")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    files = build()
    if args.write:
        for relative, data in files.items():
            with open(_fixture.fixture_path(relative), "wb") as handle:
                handle.write(data)
            print(f"wrote {relative} ({len(data)} bytes)")
        return
    bad = []
    for relative, data in files.items():
        with open(_fixture.fixture_path(relative), "rb") as handle:
            if handle.read() != data:
                bad.append(relative)
    if bad:
        raise SystemExit("MISMATCH: " + ", ".join(bad))
    print("Mandarin G2P resources: match their sources byte for byte")


if __name__ == "__main__":
    main()
