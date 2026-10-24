"""mecab-ko 0.996-ko-0.9.2 with mecab-ko-dic 2.1.1 (python-mecab-ko 1.3.7): its data for part-of-speech tagging, and
MeCab().pos on a test corpus.

g2pkk (MeloTTS's Korean phonemizer) annotates text with python-mecab-ko's part-of-speech tags. mecab-ko is MeCab 0.996
plus one change that matters here: Connector::cost adds dicrc's left-space-penalty-factor to a node with leading space
whose posid is listed. This script writes src/TextToSpeech/FrontEnd/Resources/mecab_ko_dic.gz in mecab_unidic_reference's
layout extended for that ("MCB2"): after each lexicon's ids and costs, each token's posid (u16) and part-of-speech tag
index (u16, the first field of its feature); after the character table, the tag names (u16 count, u8 length + bytes
each) and the penalty pairs (u16 count, then u16 posid and i32 cost each).

Run with python-mecab-ko and python-mecab-ko-dic installed (in a venv without mecab-python3: on a case-insensitive file
system their MeCab/mecab packages overwrite each other). Records MeCab().pos of a corpus in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mecab_ko_reference.json. Default: verify both; --write: regenerate.
"""
import argparse
import gzip
import io
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "_fixture" not in sys.modules:
    import types
    sys.modules["_fixture"] = types.ModuleType("_fixture")   # the binary readers below need nothing from it (or torch)
from mecab_unidic_reference import keys_of, read_dictionary, write_dictionary  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "src", "TextToSpeech", "FrontEnd", "Resources", "mecab_ko_dic.gz")
FIXTURE = os.path.join(ROOT, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "mecab_ko_reference.json")
CORPUS = [
    "나의 친구가 mp3 file 3개를 다운받고 있다",
    "전 제 일의 가치와 폰타인 대중들이 한 일의 의미를 잘 압니다.",
    "앞으로도 전 제 일에 자부심을 갖고 살아갈 겁니다",
    "안녕하세요, 만나서 반갑습니다!",
    "아버지가방에들어가신다",
    "아버지가 방에 들어가신다",
    "한국어 음성 합성은 재미있습니다.",
    "그는 1999년에 태어났고, 서울에서 25년 동안 살았다.",
    "읽고 밟고 넓은 들판을 걸었다.",
    "값이 싸서 많이 샀어요.",
    "  공백이   많은  문장  ",
    "ABC와 abc와 123",
    "😀 이모지가 들어간 문장",
    "의사의 의견은 무엇인가요?",
    "신라와 백제, 그리고 고구려",
    "",
]


def gz(blob):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(blob)
    return buffer.getvalue()


def tags_of(path, tokens):
    data = open(path, "rb").read()
    _, _, _, _, _, _, dsize, tsize, fsize, _ = struct.unpack("<10I", data[:40])
    features = data[40 + 32 + dsize + tsize:40 + 32 + dsize + tsize + fsize]
    tags = []
    for offset in tokens["feature"]:
        end = features.index(b"\0", int(offset))
        tags.append(features[int(offset):end].split(b",")[0])
    return tags


def penalties(dicdir):
    for line in open(os.path.join(dicdir, "dicrc"), encoding="utf-8"):
        line = line.split(";")[0].strip()
        if line.startswith("left-space-penalty-factor"):
            values = [int(v) for v in line.split("=", 1)[1].split(",")]
            return list(zip(values[0::2], values[1::2]))
    return []


def build(dicdir):
    out = io.BytesIO()
    out.write(b"MCB2")
    names = {}

    def tag_ids(tags):
        return np.array([names.setdefault(t, len(names)) for t in tags], dtype="<u2")

    for name in ("sys.dic", "unk.dic"):
        path = os.path.join(dicdir, name)
        units, tokens, lexsize, _, _ = read_dictionary(path)
        keys = keys_of(units)
        if sum(v & 0xFF for v in keys.values()) != lexsize:
            raise SystemExit(f"{name}: key reconstruction failed")
        write_dictionary(out, keys, tokens)
        out.write(tokens["posid"].astype("<u2").tobytes())
        out.write(tag_ids(tags_of(path, tokens)).tobytes())
    char = open(os.path.join(dicdir, "char.bin"), "rb").read()
    (csize,) = struct.unpack("<I", char[:4])
    categories = [char[4 + 32 * i:4 + 32 * (i + 1)].split(b"\0")[0] for i in range(csize)]
    out.write(struct.pack("<I", csize))
    for category in categories:
        out.write(struct.pack("<B", len(category)) + category)
    matrix = open(os.path.join(dicdir, "matrix.bin"), "rb").read()
    ml, mr = struct.unpack("<HH", matrix[:4])
    values = np.frombuffer(matrix[4:], dtype="<i2").reshape(mr, ml).astype(np.int32)
    out.write(struct.pack("<HH", ml, mr) + np.diff(values, axis=0, prepend=0).astype("<i2").tobytes())
    out.write(char[4 + 32 * csize:])
    ordered = sorted(names, key=names.get)
    out.write(struct.pack("<H", len(ordered)))
    for tag in ordered:
        out.write(struct.pack("<B", len(tag)) + tag)
    pairs = penalties(dicdir)
    out.write(struct.pack("<H", len(pairs)))
    for posid, cost in pairs:
        out.write(struct.pack("<Hi", posid, cost))
    return gz(out.getvalue())


def corpus():
    import mecab
    tagger = mecab.MeCab()
    return [{"text": t, "pos": [list(p) for p in tagger.pos(t)]} for t in CORPUS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import mecab_ko_dic
    blob = build(str(mecab_ko_dic.dictionary_path))
    data = {"corpus": corpus()}
    if args.write:
        with open(OUT, "wb") as handle:
            handle.write(blob)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        print(f"wrote {OUT} ({len(blob)} bytes) and {FIXTURE}")
        return
    same = open(OUT, "rb").read() == blob and json.load(open(FIXTURE, encoding="utf-8")) == data
    print("mecab-ko: data and corpus match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
