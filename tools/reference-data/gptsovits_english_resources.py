"""Builds the GPT-SoVITS English front end's embedded resources.

Run in the gptsovits-ref image (g2p_en 2.1.0, wordsegment 1.3.1, nltk 3.10.4) from a clone of RVC-Boss/GPT-SoVITS at
b4570b8 (MIT), with GPT_SoVITS on the path:

- gptsovits_engdict.tsv.gz: the dictionary en_G2p reads (engdict_cache.pickle, which get_dict prefers to the .rep files
  when it exists, as it does in the release; then engdict-hot.rep; then the six deleted abbreviations), each word's
  first pronunciation: "word<TAB>phone phone ...".
- gptsovits_namedict.tsv.gz: namedict_cache.pickle's first pronunciations, the same way.
- wordsegment_unigrams.tsv.gz / wordsegment_bigrams.tsv.gz: wordsegment's counts (Norvig's count_1w / count_2w subsets,
  as the package ships them).

The homographs are g2p_en's own (g2pen_homographs.tsv.gz, already embedded); en_G2p overrides "read" and "complex" in
code. Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import gzip
import io
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def table(entries):
    lines = []
    for word in sorted(entries):
        phones = entries[word][0]
        if any("\t" in p or " " in p or "\n" in p for p in phones) or "\t" in word or "\n" in word:
            raise ValueError(f"{word!r} holds a separator")
        lines.append(f"{word}\t{' '.join(phones)}\n")
    return gz("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gptsovits", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.join(args.gptsovits, "GPT_SoVITS"))
    from text import english
    g2p = english._g2p
    if not os.path.exists(english.CACHE_PATH):
        raise FileNotFoundError("the release ships engdict_cache.pickle; without it get_dict reads the .rep files")
    import wordsegment

    def counts(path):
        with open(path, encoding="utf-8") as handle:
            return gz(handle.read())

    folder = os.path.dirname(wordsegment.__file__)
    outputs = {
        "gptsovits_engdict.tsv.gz": table(g2p.cmu),
        "gptsovits_namedict.tsv.gz": table(g2p.namedict),
        "wordsegment_unigrams.tsv.gz": counts(os.path.join(folder, "unigrams.txt")),
        "wordsegment_bigrams.tsv.gz": counts(os.path.join(folder, "bigrams.txt")),
    }
    failed = False
    for name, data in outputs.items():
        path = os.path.join(OUT, name)
        if args.write:
            with open(path, "wb") as handle:
                handle.write(data)
            print(f"wrote {name} ({len(data)} bytes)")
        else:
            with open(path, "rb") as handle:
                same = handle.read() == data
            print(f"{name}: {'identical' if same else 'DIFFERS'}")
            failed |= not same
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
