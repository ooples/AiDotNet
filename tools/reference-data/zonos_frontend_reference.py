"""Reference fixture for Zonos's text front end: normalize_numbers (from VITS2, with inflect 7.6.0) for other languages, and zonos/conditioning.py's normalize_jp_text (Zyphra/Zonos bc40d98) with
SudachiPy 0.7.0 and sudachidict_full 20261015 (its default configuration: DefaultInputTextPlugin, ProlongedSoundMarkPlugin,
IgnoreYomiganaPlugin, MeCabOovPlugin, SimpleOovPlugin, JoinNumericPlugin, JoinKatakanaOovPlugin; split mode A), and kanjize
1.6.1's number2kanji.

Records normalize_numbers for a set of English sentences; for each Japanese sentence, the morphemes' reading forms (on the NFKC'd, kanjized text) and normalize_jp_text's output; and
number2kanji for a set of numbers.

Usage: python zonos_japanese_reference.py (with ~/source/repos/.venv-zonos). Writes
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/zonos_frontend_reference.json.
"""
import json
import os
import re
import unicodedata

from kanjize import number2kanji
from sudachipy import Dictionary, SplitMode

HERE = os.path.dirname(os.path.abspath(__file__))

SENTENCES = [
    "今日はいい天気ですね。",
    "東京都に住んでいます。",
    "ＡＢＣのﾃｽﾄｶﾞｲﾄﾞです！",
    "すごーーーーい、ほんとうに〜〜すごい",
    "漢字(かんじ)を読(よ)む練習をする。",
    "1234円と56.78ドル、それに1,000,000人。",
    "第3回の会議は2026年10月28日に開かれた。",
    "ズィーマフィーンというカタカナ語とクワトロフォルマッジ。",
    "アイスクリームを食べたい。",
    "彼はEnglishとFrançaisを話す。",
    "ＡＩの研究者が新しいモデルを発表した。",
    "これはテストです。さあ、始めよう！",
    "一千二百三十四億五千万",
    "ⅰⅱⅢと①②の記号。",
    "　全角スペースと半角 スペース\t。",
    "お茶(ちゃ)と珈琲(コーヒー)",
]
NUMBERS = [0, 1, 7, 10, 11, 20, 100, 101, 110, 999, 1000, 1001, 1234, 10000, 10001, 12345678, 100000000, 2026, 3000000000000,
           10 ** 20 + 7]


ENGLISH = [
    "I paid $3.50 for 2 coffees and £1,250 for the 1st class ticket.",
    "In 1999, 2000, 2005, 2010 and 1800 we saw 3,000,000 people.",
    "The 21st, 22nd, 103rd and 1000th visitors got 0.75 points each.",
    "Room 1001, call 555 0199, $0.01, $5, $1.00 and $3.4.5 dollars.",
    "1234567 and 0 and 7 and 12.",
]


def normalize_numbers():
    # zonos.conditioning's number normalization, imported without its eSpeak, Sudachi and kanjize dependencies.
    source = open(os.path.join(HERE, "..", "..", "..", "refsrc", "Zonos", "zonos", "conditioning.py"), encoding="utf-8").read()
    start = source.index("# --- Number normalization code from")
    end = source.index("# --- Number normalization code end ---")
    namespace = {}
    exec("import re\nimport inflect\n" + source[start:end], namespace)
    return namespace["normalize_numbers"]


def main():
    normalize = normalize_numbers()
    english = [{"text": t, "normalized": normalize(t)} for t in ENGLISH]
    tokenizer = Dictionary(dict="full").create()
    cases = []
    for sentence in SENTENCES:
        text = unicodedata.normalize("NFKC", sentence)
        text = re.sub(r"\d+", lambda m: number2kanji(int(m[0])), text)
        readings = [m.reading_form() for m in tokenizer.tokenize(text, SplitMode.A)]
        cases.append({"sentence": sentence, "prepared": text, "readings": readings, "normalized": " ".join(readings)})
    result = {"english": english, "cases": cases, "kanjize": [{"number": str(n), "kanji": number2kanji(n)} for n in NUMBERS]}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "zonos_frontend_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False)
        handle.write("\n")
    for c in cases:
        print(c["normalized"])


if __name__ == "__main__":
    main()
