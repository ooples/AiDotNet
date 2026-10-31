"""Builds the GPT-SoVITS Chinese front end's embedded resources from GPT-SoVITS's pinned dependencies.

Run in the gptsovits-ref image (pypinyin 0.55.0, OpenCC 1.4.2, jieba_fast 0.53; see gptsovits_frontend_reference.py)
with a clone of RVC-Boss/GPT-SoVITS at b4570b8 (MIT) and the G2PW model GPT-SoVITS downloads (G2PWModel_1.1.zip,
Apache-2.0):

- gptsovits_pypinyin_chars.tsv.gz / gptsovits_pypinyin_phrases.tsv.gz: every character's and phrase's first reading in
  pypinyin's TONE3, INITIALS and FINALS_TONE3 styles with neutral_tone_with_five, computed by the converter pinyin and
  lazy_pinyin build (UltimateConverter): one "text<TAB>tone3,initial,final ..." line each.
- gptsovits_chinese.json.gz: GPT-SoVITS's opencpop-strict table, its tone sandhi's and erhua's word lists, its
  polyphonic correction dictionary (polyphonic.rep then polyphonic-fix.rep), G2PW's character tables and its
  bopomofo-to-pinyin map, zh_normalization's traditional-to-simplified map, both symbol tables (symbols.py, symbols2.py), and the characters Python's str.isnumeric
  accepts (the sandhi asks it).
- opencc_s2tw.tsv.gz: the dictionaries OpenCC's s2tw configuration reads, dumped from the installed package's ocd2
  files by its own opencc_dict tool; "#name" opens each dictionary, then "key<TAB>value value ..." lines.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import ast
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def pinyin_tables():
    from pypinyin import Style
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    from pypinyin.converter import UltimateConverter
    converter = UltimateConverter(neutral_tone_with_five=True)

    def reading(text):
        styles = [converter.convert(text, style, False, "default", strict=True, neutral_tone_with_five=True)
                  for style in (Style.TONE3, Style.INITIALS, Style.FINALS_TONE3)]
        if len({len(s) for s in styles}) != 1:
            raise ValueError(f"the styles of {text!r} disagree on the syllable count")
        triples = [f"{t[0]},{i[0]},{f[0]}" for t, i, f in zip(*styles)]
        if any(" " in p or "\t" in p or p.count(",") != 2 for p in triples):
            raise ValueError(f"a reading of {text!r} holds a separator")
        return " ".join(triples)

    chars = "".join(f"{chr(code)}\t{reading(chr(code))}\n" for code in sorted(PINYIN_DICT))
    phrases = "".join(f"{p}\t{reading(p)}\n" for p in sorted(PHRASES_DICT))
    return gz(chars), gz(phrases)


def chinese_tables(gptsovits, g2pw):
    text = os.path.join(gptsovits, "GPT_SoVITS", "text")
    sys.path.insert(0, os.path.join(gptsovits, "GPT_SoVITS"))
    with open(os.path.join(text, "opencpop-strict.txt"), encoding="utf-8") as handle:
        opencpop = {line.split("\t")[0]: line.strip().split("\t")[1] for line in handle.readlines()}

    # The sandhi's and erhua's word lists, read from the source (importing chinese2 would load the G2PW model).
    import importlib.util
    spec = importlib.util.spec_from_file_location("gsv_tone_sandhi", os.path.join(text, "tone_sandhi.py"))
    sandhi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandhi_module)
    sandhi = sandhi_module.ToneSandhi()
    with open(os.path.join(text, "chinese2.py"), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    erhua = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("must_erhua", "not_erhua"):
            erhua[node.targets[0].id] = sorted(ast.literal_eval(node.value))

    corrections = {}
    for name in ("polyphonic.rep", "polyphonic-fix.rep"):
        with open(os.path.join(text, "g2pw", name), encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":")
                corrections[key.strip()] = ast.literal_eval(value.strip())

    def tsv(name):
        with open(os.path.join(g2pw, name), encoding="utf-8") as handle:
            return [line.split("\t") for line in handle.read().strip().split("\n")]

    with open(os.path.join(g2pw, "bopomofo_to_pinyin_wo_tune_dict.json"), encoding="utf-8") as handle:
        bopomofo = json.load(handle)

    spec = importlib.util.spec_from_file_location("gsv_char_convert", os.path.join(text, "zh_normalization", "char_convert.py"))
    char_convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(char_convert)

    import text.symbols as symbols_v1
    import text.symbols2 as symbols_v2

    data = {
        "opencpop": opencpop,
        "must_neural_tone_words": sorted(sandhi.must_neural_tone_words),
        "must_not_neural_tone_words": sorted(sandhi.must_not_neural_tone_words),
        "must_erhua": erhua["must_erhua"],
        "not_erhua": erhua["not_erhua"],
        "polyphonic_corrections": corrections,
        "g2pw_polyphonic_chars": tsv("POLYPHONIC_CHARS.txt"),
        "g2pw_monophonic_chars": tsv("MONOPHONIC_CHARS.txt"),
        "g2pw_bopomofo_to_pinyin": bopomofo,
        "symbols_v1": symbols_v1.symbols,
        "symbols_v2": symbols_v2.symbols,
        "t2s": char_convert.t2s_dict,
        "numeric": "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isnumeric()),
    }
    return gz(json.dumps(data, ensure_ascii=False, sort_keys=True))


S2TW_DICTIONARIES = ["CJK_Compatibility_Ideographs", "STPhrases", "STPhrases_GeneratedFromRegionalPhrases", "STCharacters",
                     "TWVariantsPhrases", "TWVariants"]


def opencc_tables():
    import opencc
    clib = os.path.join(os.path.dirname(opencc.__file__), "clib")
    with open(os.path.join(clib, "share", "opencc", "s2tw.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    files = json.dumps(config)
    for name in S2TW_DICTIONARIES:
        if f'"{name}.ocd2"' not in files:
            raise ValueError(f"s2tw.json no longer reads {name}")
    parts = []
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(clib, "lib64"))
    with tempfile.TemporaryDirectory() as tmp:
        for name in S2TW_DICTIONARIES:
            out = os.path.join(tmp, name + ".txt")
            subprocess.run([os.path.join(clib, "bin", "opencc_dict"), "-i", os.path.join(clib, "share", "opencc", name + ".ocd2"),
                            "-o", out, "-f", "ocd2", "-t", "text"], check=True, env=env)
            with open(out, encoding="utf-8") as handle:
                lines = [line for line in handle.read().split("\n") if line]
            parts.append(f"#{name}\n" + "".join(line + "\n" for line in lines))
    return gz("".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gptsovits", required=True, help="clone of RVC-Boss/GPT-SoVITS at b4570b8")
    parser.add_argument("--g2pw", required=True, help="the extracted G2PWModel_1.1 directory")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    chars, phrases = pinyin_tables()
    outputs = {
        "gptsovits_pypinyin_chars.tsv.gz": chars,
        "gptsovits_pypinyin_phrases.tsv.gz": phrases,
        "gptsovits_chinese.json.gz": chinese_tables(args.gptsovits, args.g2pw),
        "opencc_s2tw.tsv.gz": opencc_tables(),
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
