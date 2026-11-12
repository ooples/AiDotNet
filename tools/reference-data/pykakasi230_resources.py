"""Builds pykakasi 2.3.0's dictionary for AiDotNet as a delta over the committed pykakasi 2.2.1 export
(melo_japanese_resources.py): MaskGCT's Japanese front end calls 2.3.0's `kakasi().convert`, whose kanwadict4.db
differs from 2.2.1's in ~100 entries and adds a context to readings (a reading applies only after one of the listed
preceding words). Its itaijidict4.db equals 2.2.1's, which this script asserts.

pykakasi230_kanwa_delta.tsv.gz holds one line per entry that 2.3.0 changes or adds:
"key<TAB>reading<US>reading...", a reading with a context written "reading<RS>word<GS>word..."; and one line, the
key alone, per entry 2.3.0 removes. Within an entry the readings keep 2.3.0's order (the converter takes the first
reading that applies), and the order of entries does not matter (the longest key wins; two keys of one length
cannot both start the text).

Usage: python pykakasi230_resources.py PYKAKASI_2_3_0_DATA_DIR [--write]   (default: verify the committed file)
"""
import argparse
import gzip
import io
import os
import pickle

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    kanwa = pickle.load(open(os.path.join(args.data, "kanwadict4.db"), "rb"))
    itaiji = pickle.load(open(os.path.join(args.data, "itaijidict4.db"), "rb"))
    with gzip.open(os.path.join(RESOURCES, "pykakasi221_itaiji.tsv.gz"), "rt", encoding="utf-8") as handle:
        old_itaiji = {int(a): (b if b else None) for a, b in (line.rstrip("\n").split("\t") for line in handle)}
    if old_itaiji != itaiji:
        raise SystemExit("itaijidict4.db differs from 2.2.1's; export it too")
    with gzip.open(os.path.join(RESOURCES, "pykakasi221_kanwa.tsv.gz"), "rt", encoding="utf-8") as handle:
        old = {}
        for line in handle:
            key, readings = line.rstrip("\n").split("\t")
            old[key] = [(r, None) for r in readings.split("\x1f")]

    new = {}
    for first, entries in kanwa.items():
        for key, readings in entries.items():
            if ord(key[0]) != first:
                raise ValueError(f"{key!r} is filed under {first}")
            new[key] = [(y, None if c is None else list(c)) for y, c in readings]

    lines = []
    for key, readings in new.items():
        if old.get(key) != readings:
            fields = []
            for yomi, context in readings:
                text = yomi if context is None else yomi + "\x1e" + "\x1d".join(context)
                if any(ch in "\t\x1f\n" for ch in key + text):
                    raise ValueError(f"a separator in {key!r}")
                fields.append(text)
            lines.append(key + "\t" + "\x1f".join(fields) + "\n")
    for key in old:
        if key not in new:
            lines.append(key + "\n")
    blob = gz("".join(lines).encode("utf-8"))
    path = os.path.join(RESOURCES, "pykakasi230_kanwa_delta.tsv.gz")
    if args.write:
        with open(path, "wb") as handle:
            handle.write(blob)
        print(f"wrote {len(lines)} delta lines")
    elif open(path, "rb").read() != blob:
        raise SystemExit("pykakasi230_kanwa_delta.tsv.gz differs")
    else:
        print("matches")


if __name__ == "__main__":
    main()
