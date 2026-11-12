"""Fish Speech's text front end on a set of texts, from the releases' own code.

For each release (1.4.3 and 1.5.1): clean_text, split_text at three chunk lengths, and (1.4's API default) the Chinese
number normalizer ChnNormedText. Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech_text_reference.json.

    python fishspeech_text_reference.py --fish14 <fish-speech v1.4.3 checkout> --fish15 <fish-speech v1.5.1 checkout>

Each release is imported in its own process, since both name their package fish_speech.
"""
import argparse
import json
import os
import subprocess
import sys

TEXTS = [
    "  Hello, world!  ",
    "This is a test sentence. This is another test sentence. And a third one.",
    "a,aaaaaa3.14",
    "   ",
    "a",
    "This is a test sentence with only commas, and no dots, and no exclamation marks, and no question marks, and no newlines.",
    "This is a test sentence This is a test sentence This is a test sentence. This is a test sentence, This is a test sentence, This is a test sentence.",
    "这是一段很长的中文文本,而且没有句号,也没有感叹号,也没有问号,也没有换行符。",
    "It’s ‘quoted’ and “double”… (with brackets) [and more] 【】（x）",
    "Wait... what,,, really?! Yes!!! 3.14159 is pi, 2.5 is not.",
    "Emoji \U0001f600 here \U0001f680 and flags \U0001f1fa\U0001f1f8 and a clef \U0001d11e stays.",
    "Line one\nline two\r\nline three four",
    "Colons: semicolons; full-width：； tildes~～ dashes— dots·・ quotes「」《》",
    "  Spaces　around\u001c\u001f",
    "你好，世界！今天是2024年10月29日，温度是25.5%。",
    "电话：0595-23865596或手机+86 19859213959。金额：12块5，34.5元。",
    "我有25个苹果和3人，分数：3/4，编号：31520181154418。",
    "共有20000人参加，200多人和3.5抽，1999人。",
    "Mixed 中文 and English, with numbers 123 and 4.56, plus O2O and B2C.",
    "A very long sentence without any punctuation that keeps going and going so that the splitter has to fall back to spaces "
    "and then even further it has to break words byeverypossiblemeansbecausethereisnospaceleftinthisverylongwordthatgoesonandon",
    "日本語の文です。これはテストです！本当ですか？",
]
LENGTHS = [10, 50, 200]


def child(fish):
    sys.path.insert(0, fish)
    from fish_speech.text.clean import clean_text
    from fish_speech.text.spliter import split_text
    out = {"clean": [clean_text(t) for t in TEXTS],
           "split": {str(n): [split_text(t, n) for t in TEXTS] for n in LENGTHS}}
    try:
        from fish_speech.text.chn_text_norm.text import Text
        out["normalized"] = [Text(raw_text=t).normalize() for t in TEXTS]
    except ImportError:
        pass
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fish14", required=True)
    parser.add_argument("--fish15", required=True)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    result = {"texts": TEXTS, "lengths": LENGTHS}
    for key, path in (("v14", args.fish14), ("v15", args.fish15)):
        run = subprocess.run([sys.executable, os.path.abspath(__file__), "--fish14", "-", "--fish15", "-", "--child", os.path.abspath(path)],
                             capture_output=True, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        result[key] = json.loads(run.stdout.decode("utf-8"))
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech",
                          "ReferenceData", "fishspeech_text_reference.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("wrote", os.path.normpath(target))


if __name__ == "__main__":
    main()
