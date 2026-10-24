"""MeCab 0.996 with the unidic-lite 1.0.8 dictionary: its data for word segmentation, and -Owakati on a test corpus.

OuteTTS 1.0 counts and splits Japanese and Chinese text with MeCab.Tagger("-Owakati") (mecab-python3 with unidic-lite,
BSD/LGPL/GPL triple-licensed; the dictionary is BSD). Segmentation needs only part of the dictionary: the surfaces
(the keys of sys.dic's double array) with their token ranges, each token's left and right context ids and word cost,
the connection-cost matrix, the character categories (char.bin) and the unknown-word dictionary (unk.dic). The
features (158 MB of the 188 MB sys.dic) are not needed. This script reads those parts from the binary files and writes
them in a compact form, src/TextToSpeech/FrontEnd/Resources/mecab_unidic_lite.gz:

  "MCB1"; then for the system and unknown dictionaries each: key count, then per key (sorted) the shared-prefix length
  (u8) with the previous key, suffix length (u8), suffix bytes and value (i32: token index << 8 | token count); token
  count, then left ids (u16), right ids (u16) and word costs (i16); then the unknown-word category names (count, then
  u8 length + bytes each); the matrix's left and right sizes (u16) and its entries as int16 differences down each
  column (the matrix stored row-major as matrix[right_id + lsize * left_id]); and char.bin's 65535 category entries
  (u32 bit fields: type 18, default type 8, length 4, group 1, invoke 1).

The keys are reconstructed from the double array (base/check units) and checked: the token counts must sum to the
lexicon size. Records Tagger("-Owakati").parse of a corpus in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/
mecab_unidic_reference.json. Default: verify both; --write: regenerate them. Needs mecab-python3 and unidic-lite 1.0.8.
"""
import argparse
import gzip
import importlib.metadata
import io
import os
import struct
import sys

import numpy as np

import _fixture

OUT = "src/TextToSpeech/FrontEnd/Resources/mecab_unidic_lite.gz"
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mecab_unidic_reference.json"
CORPUS = [
    "すもももももももものうち",
    "今日はいい天気ですね。",
    "東京特許許可局長の許可が下りた。",
    "私は日本語を勉強しています。",
    "吾輩は猫である。名前はまだ無い。",
    "カタカナのコンピューターサイエンスとプログラミング",
    "ＡＢＣとabcと１２３と123",
    "ｶﾀｶﾅの半角とひらがな",
    "これは  空白を含む 文 です",
    "絵文字😀を含む文章🎉です",
    "Pythonで機械学習を始めよう!",
    "我来到北京清华大学",
    "他光着双脚，脚趾头怪模怪样地向外翻着。",
    "二〇二四年十月二十四日",
    "ﾃｽﾄ　テスト　test",
    "「こんにちは」と彼は言った。",
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンアイウエオ",
    "ーーー！？？",
    "東京都千代田区丸の内1-9-1",
    "",
    " ",
    "a",
]


def gz(blob):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(blob)
    return buffer.getvalue()


def read_dictionary(path):
    data = open(path, "rb").read()
    magic, version, typ, lexsize, lsize, rsize, dsize, tsize, fsize, _ = struct.unpack("<10I", data[:40])
    if version != 102:
        raise SystemExit(f"{path}: version {version}")
    offset = 40 + 32
    units = np.frombuffer(data[offset:offset + dsize], dtype=np.dtype([("base", "<i4"), ("check", "<u4")]))
    offset += dsize
    tokens = np.frombuffer(data[offset:offset + tsize], dtype=np.dtype([("lc", "<u2"), ("rc", "<u2"), ("posid", "<u2"),
                                                                       ("wcost", "<i2"), ("feature", "<u4"), ("compound", "<u4")]))
    return units, tokens, lexsize, lsize, rsize


def keys_of(units):
    """Every key of a Darts double array with its value: a node at index s with base b has children at b + c + 1
    (check == b) for byte c, and its terminal at b (check == b, base < 0, value -base - 1)."""
    base = units["base"].astype(np.int64)
    check = units["check"].astype(np.int64)
    keys = {}
    stack = [(0, b"")]
    n = len(units)
    while stack:
        s, key = stack.pop()
        b = base[s]
        if 0 <= b < n and check[b] == b and base[b] < 0:
            keys[key] = int(-base[b] - 1)
        if b < 0:
            continue
        lo, hi = b + 1, min(n, b + 257)
        if lo >= n:
            continue
        idx = np.nonzero(check[lo:hi] == b)[0]
        for i in idx:
            p = lo + int(i)
            stack.append((p, key + bytes([p - b - 1])))
    return keys


def write_dictionary(out, keys, tokens):
    items = sorted(keys.items())
    out.write(struct.pack("<I", len(items)))
    previous = b""
    for key, value in items:
        shared = 0
        while shared < min(len(key), len(previous), 255) and key[shared] == previous[shared]:
            shared += 1
        suffix = key[shared:]
        if len(suffix) > 255:
            raise SystemExit("key too long")
        out.write(struct.pack("<BB", shared, len(suffix)) + suffix + struct.pack("<i", value))
        previous = key
    out.write(struct.pack("<I", len(tokens)))
    out.write(tokens["lc"].astype("<u2").tobytes())
    out.write(tokens["rc"].astype("<u2").tobytes())
    out.write(tokens["wcost"].astype("<i2").tobytes())


def build(dicdir):
    out = io.BytesIO()
    out.write(b"MCB1")
    units, tokens, lexsize, lsize, rsize = read_dictionary(os.path.join(dicdir, "sys.dic"))
    keys = keys_of(units)
    if sum(v & 0xFF for v in keys.values()) != lexsize or len(tokens) != lexsize:
        raise SystemExit(f"key reconstruction failed: {sum(v & 0xFF for v in keys.values())} tokens of {lexsize}")
    write_dictionary(out, keys, tokens)
    unk_units, unk_tokens, unk_lexsize, _, _ = read_dictionary(os.path.join(dicdir, "unk.dic"))
    unk_keys = keys_of(unk_units)
    if sum(v & 0xFF for v in unk_keys.values()) != unk_lexsize:
        raise SystemExit("unknown-word key reconstruction failed")
    write_dictionary(out, unk_keys, unk_tokens)
    char = open(os.path.join(dicdir, "char.bin"), "rb").read()
    (csize,) = struct.unpack("<I", char[:4])
    names = [char[4 + 32 * i:4 + 32 * (i + 1)].split(b"\0")[0] for i in range(csize)]
    out.write(struct.pack("<I", csize))
    for name in names:
        out.write(struct.pack("<B", len(name)) + name)
    matrix = open(os.path.join(dicdir, "matrix.bin"), "rb").read()
    ml, mr = struct.unpack("<HH", matrix[:4])
    if (ml, mr) != (lsize, rsize):
        raise SystemExit("matrix size")
    values = np.frombuffer(matrix[4:], dtype="<i2").reshape(mr, ml).astype(np.int32)
    deltas = np.diff(values, axis=0, prepend=0).astype("<i2")
    out.write(struct.pack("<HH", ml, mr) + deltas.tobytes())
    out.write(char[4 + 32 * csize:])
    return gz(out.getvalue())


def corpus():
    import MeCab
    tagger = MeCab.Tagger("-Owakati")
    return [{"text": t, "wakati": tagger.parse(t)} for t in CORPUS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("unidic-lite") != "1.0.8":
        raise SystemExit("needs unidic-lite 1.0.8")
    import unidic_lite
    blob = build(unidic_lite.DICDIR)
    data = {"corpus": corpus()}
    if args.write:
        with open(_fixture.fixture_path(OUT), "wb") as handle:
            handle.write(blob)
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {OUT} ({len(blob)} bytes) and {FIXTURE}")
        return
    with open(_fixture.fixture_path(OUT), "rb") as handle:
        same = handle.read() == blob
    same = same and _fixture.load(FIXTURE)["corpus"] == data["corpus"]
    print("MeCab: data and corpus match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
