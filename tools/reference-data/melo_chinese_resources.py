"""Builds the MeloTTS Chinese front end's embedded resources from MeloTTS's pinned dependencies.

Run with pypinyin 0.50.0 and jieba 0.42.1 (MeloTTS's requirements.txt; MIT) and a clone of myshell-ai/MeloTTS at
2091453 (MIT):

- pypinyin050_chars.tsv.gz / pypinyin050_phrases.tsv.gz: every character's and phrase's reading in pypinyin's
  INITIALS and FINALS_TONE3 styles with neutral_tone_with_five (the two styles MeloTTS reads), computed by pypinyin's own
  converter as lazy_pinyin builds it (first reading, strict): one "text<TAB>initial,final initial,final ..." line
  each.
- jieba_pos.tsv.gz: jieba's dictionary part-of-speech tags (dict.txt), the tags jieba.posseg reads.
- jieba_posseg.json.gz: jieba.posseg's HMM (char_state_tab, prob_start, prob_trans, prob_emit), each state "B,n".
- melo_chinese.json.gz: MeloTTS's opencpop-strict pinyin-to-phone table, its tone sandhi's word lists, and the
  characters Python's str.isnumeric accepts (the sandhi asks it).

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import ast
import gzip
import io
import json
import os

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
    # lazy_pinyin's own converter: DefaultConverter ignores neutral_tone_with_five (的 would read "e", not "e5").
    converter = UltimateConverter(neutral_tone_with_five=True)

    def reading(text):
        initials = converter.convert(text, Style.INITIALS, False, "default", strict=True, neutral_tone_with_five=True)
        finals = converter.convert(text, Style.FINALS_TONE3, False, "default", strict=True, neutral_tone_with_five=True)
        pairs = [f"{i[0]},{f[0]}" for i, f in zip(initials, finals)]
        if any(" " in p or "\t" in p for p in pairs):
            raise ValueError(f"a reading of {text!r} holds a separator")
        return " ".join(pairs)

    chars = "".join(f"{chr(code)}\t{reading(chr(code))}\n" for code in sorted(PINYIN_DICT))
    phrases = "".join(f"{p}\t{reading(p)}\n" for p in sorted(PHRASES_DICT))
    return gz(chars), gz(phrases), len(PINYIN_DICT), len(PHRASES_DICT)


def jieba_tables():
    import jieba
    from jieba.posseg.char_state_tab import P as char_state
    from jieba.posseg.prob_emit import P as emit
    from jieba.posseg.prob_start import P as start
    from jieba.posseg.prob_trans import P as trans
    path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    tags = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            word, _, tag = line.split(" ")
            tags.append(f"{word}\t{tag}\n")

    def key(state):
        return f"{state[0]},{state[1]}"

    model = {
        "char_state": {c: [key(s) for s in states] for c, states in sorted(char_state.items())},
        "start": {key(s): p for s, p in sorted(start.items())},
        "trans": {key(s): {key(t): p for t, p in sorted(row.items())} for s, row in sorted(trans.items())},
        "emit": {key(s): dict(sorted(row.items())) for s, row in sorted(emit.items())},
    }
    return gz("".join(tags)), gz(json.dumps(model, ensure_ascii=False)), len(tags)


def melo_tables(melo):
    text_dir = os.path.join(melo, "melo", "text")
    with open(os.path.join(text_dir, "opencpop-strict.txt"), encoding="utf-8") as handle:
        opencpop = {line.split("\t")[0]: line.strip().split("\t")[1] for line in handle}
    source = open(os.path.join(text_dir, "tone_sandhi.py"), encoding="utf-8").read()
    lists = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            name = node.targets[0].attr
            if name in ("must_neural_tone_words", "must_not_neural_tone_words"):
                lists[name] = sorted(ast.literal_eval(node.value))
    # Python's str.isnumeric, which the tone sandhi asks of characters: Unicode's numerics including Unihan's (一, 二, ...).
    numeric = "".join(chr(c) for c in range(0x30000) if chr(c).isnumeric())
    return gz(json.dumps({"opencpop": opencpop, "numeric": numeric, **lists}, ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--melo", required=True, help="myshell-ai/MeloTTS at 2091453 (the repository root)")
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    chars, phrases, n_chars, n_phrases = pinyin_tables()
    tags, posseg, n_tags = jieba_tables()
    outputs = {"pypinyin050_chars.tsv.gz": chars, "pypinyin050_phrases.tsv.gz": phrases, "jieba_pos.tsv.gz": tags,
               "jieba_posseg.json.gz": posseg, "melo_chinese.json.gz": melo_tables(args.melo)}
    if args.write:
        for name, data in outputs.items():
            with open(os.path.join(OUT, name), "wb") as handle:
                handle.write(data)
        print(f"wrote {len(outputs)} resources: {n_chars} characters, {n_phrases} phrases, {n_tags} tagged words")
        return
    failures = []
    for name, data in outputs.items():
        with open(os.path.join(OUT, name), "rb") as handle:
            if handle.read() != data:
                failures.append(name)
    if failures:
        raise SystemExit("differ: " + ", ".join(failures))
    print("all resources match")


if __name__ == "__main__":
    main()
