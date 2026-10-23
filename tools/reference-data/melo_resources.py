"""Builds the MeloTTS English front end's embedded resources from their pinned sources.

- melo_cmudict.tsv.gz: MeloTTS's syllabified dictionary (myshell-ai/MeloTTS melo/text/cmudict.rep at 2091453; CMU
  Pronouncing Dictionary 0.6, BSD-2-Clause), read as melo/text/english.py's read_dict does (from line 49): one
  "WORD<TAB>syllable - syllable" line per word, the syllables' phones space-separated.
- g2pen_cmudict.tsv.gz: the CMU dictionary g2p_en looks words up in (nltk's cmudict corpus, cmudict 0.7b), each word's
  first pronunciation, as g2p_en 2.1.0 takes it (cmudict.dict()[word][0]).
- g2pen_homographs.tsv.gz: g2p_en 2.1.0's homographs.en (Apache-2.0), verbatim.
- g2pen_model.f32.gz: g2p_en 2.1.0's checkpoint20.npz (Apache-2.0) as records of name, rank, dimensions and
  little-endian float32 values.
- perceptron_tagger_eng.json.gz: nltk's averaged_perceptron_tagger_eng (the weights, the tag dictionary and the
  classes), the tagger g2p_en's pos_tag uses.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import gzip
import io
import json
import os
import struct

import numpy as np

import _fixture

OUT = "src/TextToSpeech/FrontEnd/Resources/"


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def melo_cmudict(path):
    lines = []
    with open(path) as handle:                       # read_dict opens with the platform default, as here
        for index, line in enumerate(handle, start=1):
            if index < 49:
                continue
            line = line.strip()
            word, rest = line.split("  ", 1)[0], line.split("  ")[1]
            if "\t" in word or "\t" in rest:
                raise ValueError("a tab in the dictionary")
            lines.append(f"{word}\t{rest}\n")
    return gz("".join(lines).encode("utf-8")), len(lines)


def g2pen_cmudict():
    from nltk.corpus import cmudict
    entries = cmudict.dict()
    text = "".join(f"{w}\t{' '.join(p[0])}\n" for w, p in sorted(entries.items()))
    return gz(text.encode("utf-8")), len(entries)


def g2pen_model(path):
    variables = np.load(path)
    out = io.BytesIO()
    names = sorted(variables.files)
    out.write(struct.pack("<i", len(names)))
    for name in names:
        array = np.ascontiguousarray(variables[name], dtype="<f4")
        encoded = name.encode("ascii")
        out.write(struct.pack("<i", len(encoded)))
        out.write(encoded)
        out.write(struct.pack("<i", array.ndim))
        for dim in array.shape:
            out.write(struct.pack("<i", dim))
        out.write(array.tobytes())
    return gz(out.getvalue()), len(names)


def perceptron(directory):
    parts = {}
    for key in ("weights", "tagdict", "classes"):
        with open(os.path.join(directory, f"averaged_perceptron_tagger_eng.{key}.json"), encoding="utf-8") as handle:
            parts[key] = json.load(handle)
    return gz(json.dumps(parts, ensure_ascii=False, sort_keys=False).encode("utf-8")), len(parts["weights"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--melo", required=True, help="myshell-ai/MeloTTS at 2091453 (the repository root)")
    parser.add_argument("--g2pen", required=True, help="the installed g2p_en 2.1.0 package directory")
    parser.add_argument("--tagger", required=True, help="nltk_data taggers/averaged_perceptron_tagger_eng")
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    outputs = {}
    outputs["melo_cmudict.tsv.gz"], melo_words = melo_cmudict(os.path.join(args.melo, "melo", "text", "cmudict.rep"))
    outputs["g2pen_cmudict.tsv.gz"], cmu_words = g2pen_cmudict()
    with open(os.path.join(args.g2pen, "homographs.en"), "rb") as handle:
        outputs["g2pen_homographs.tsv.gz"] = gz(handle.read())
    outputs["g2pen_model.f32.gz"], arrays = g2pen_model(os.path.join(args.g2pen, "checkpoint20.npz"))
    outputs["perceptron_tagger_eng.json.gz"], features = perceptron(args.tagger)
    if args.write:
        for name, data in outputs.items():
            with open(_fixture.fixture_path(OUT + name), "wb") as handle:
                handle.write(data)
        print(f"wrote {len(outputs)} resources: {melo_words} Melo words, {cmu_words} CMU words, {arrays} arrays, {features} features")
        return
    check = _fixture.Comparison("MeloTTS English resources")
    for name, data in outputs.items():
        with open(_fixture.fixture_path(OUT + name), "rb") as handle:
            check.exact(OUT + name, handle.read(), data)
    check.report()


if __name__ == "__main__":
    main()
