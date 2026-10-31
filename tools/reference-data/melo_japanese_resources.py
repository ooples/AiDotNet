"""Builds the MeloTTS Japanese front end's embedded resources from MeloTTS's pinned dependencies.

- pykakasi221_kanwa.tsv.gz: pykakasi 2.2.1's (GPL-3.0-or-later dictionary data from KAKASI) kanwadict4.db, the
  kanji-to-hiragana dictionary its legacy J2 converter reads: one "key<TAB>reading<US>reading..." line per entry, in
  the pickle's order (the converter takes the first longest key and its first reading, so order is data).
- pykakasi221_itaiji.tsv.gz: itaijidict4.db, the variant-kanji translation table ("code<TAB>replacement", empty when
  the character is deleted).
- bert-base-japanese-v3-vocab.txt.gz (src/Tokenization/Resources): tohoku-nlp/bert-base-japanese-v3's WordPiece
  vocabulary (Apache-2.0), verbatim.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import glob
import gzip
import io
import os
import pickle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
FRONT_END = os.path.join(ROOT, "TextToSpeech", "FrontEnd", "Resources")
TOKENIZATION = os.path.join(ROOT, "Tokenization", "Resources")


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def kanwa(path):
    table = pickle.load(open(path, "rb"))
    lines = []
    for first, entries in table.items():
        for key, readings in entries.items():
            if ord(key[0]) != first:
                raise ValueError(f"{key!r} is filed under {first}")
            if any(c in "\t\x1f\n" for c in key + "".join(readings)):
                raise ValueError(f"a separator in {key!r}")
            lines.append(f"{key}\t" + "\x1f".join(readings) + "\n")
    return gz("".join(lines).encode("utf-8")), len(lines)


def itaiji(path):
    table = pickle.load(open(path, "rb"))
    lines = [f"{code}\t{'' if value is None else value}\n" for code, value in table.items()]
    return gz("".join(lines).encode("utf-8")), len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    import pykakasi
    data = os.path.join(os.path.dirname(pykakasi.__file__), "data")
    vocab = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--tohoku-nlp--bert-base-japanese-v3/snapshots/*/vocab.txt"))[0]
    outputs = {}
    outputs[os.path.join(FRONT_END, "pykakasi221_kanwa.tsv.gz")], entries = kanwa(os.path.join(data, "kanwadict4.db"))
    outputs[os.path.join(FRONT_END, "pykakasi221_itaiji.tsv.gz")], _ = itaiji(os.path.join(data, "itaijidict4.db"))
    outputs[os.path.join(TOKENIZATION, "bert-base-japanese-v3-vocab.txt.gz")] = gz(open(vocab, "rb").read())
    if args.write:
        for path, blob in outputs.items():
            with open(path, "wb") as handle:
                handle.write(blob)
        print(f"wrote {len(outputs)} resources: {entries} kanwa entries")
        return
    failures = [os.path.basename(p) for p, blob in outputs.items() if open(p, "rb").read() != blob]
    if failures:
        raise SystemExit("differ: " + ", ".join(failures))
    print("all resources match")


if __name__ == "__main__":
    main()
