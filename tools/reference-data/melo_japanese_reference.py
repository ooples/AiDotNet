"""Reference fixture for MeloTTS's Japanese front end: melo/text/japanese.py's text_normalize and g2p, and the pieces
they rest on (pykakasi's converter, the number conversion, kata2phoneme).

Runs MeloTTS's own japanese.py (myshell-ai/MeloTTS at 2091453) with its pins (pykakasi 2.2.1, num2words 0.5.12,
mecab-python3 1.0.9 with unidic-lite 1.0.8, transformers' tohoku-nlp/bert-base-japanese-v3 tokenizer over fugashi),
the package's other language modules left out.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_japanese_reference.json.
"""
import importlib.util
import json
import os
import random
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc", "MeloTTS", "melo", "text"))

package = types.ModuleType("melotext")
package.__path__ = [TEXT]
sys.modules["melotext"] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symbols = load("melotext.symbols", os.path.join(TEXT, "symbols.py"))
package.symbols = symbols.symbols
japanese = load("melotext.japanese", os.path.join(TEXT, "japanese.py"))

SENTENCES = [
    "こんにちは、世界！",
    "ええ、僕はおきなと申します。こちらの小さいわらべは杏子。ご挨拶が遅れてしまいすみません。あなたの名は?",
    "あの、お前以外のみんなは、全員生きてること?",
    "今日は1789年7月14日、気温は23.5度です。",
    "この本は$5.50、あの本は¥1,200で、合計1,234,567円でした。",
    "東京都渋谷区で、スーパーマーケットへ行きました。",
    "ラーメンとコーヒーを注文した。ーーー",
    "わたしはがくせいです。かーさん、ありがとう。",
    "日本語の音声合成は面白いですね。",
    "ＡＢＣとabcと１２３、そして全角カナ。",
    "彼は「はい」と答えた――が、本当は違った。",
    "一二三四五六七八九十百千万億兆",
    "鬱蒼とした森の中で、薔薇が咲いていた。",
    "ヴァイオリンとヴィオラ、ティーカップとディズニー。",
    "髙橋さんと琦玉県の煞。",
    "おはようございます！今日もよろしくお願いします。",
    "ゝゞヽヾ〃々〆〇",
    "パーティーは3時から、0.5時間遅れて始まった。",
]

KAKASI = ["漢字", "公安委員会", "東京", "日本語", "鬱", "薔薇", "々", "〆", "かー", "がっこう", "ーかな", "カーー", "ー",
          "ぁぃぅぇぉゃゅょっゎゕゖ", "ゔ", "ゝゞ", "ヽヾ", "日本ーご", "一", "二十", "千年", "髙", "琦", "𠮷野家"]

NUMBERS = ["0", "1", "7", "4", "10", "11", "20", "100", "101", "1000", "1001", "10000", "10001", "100000000", "123456789",
           "1000000000000", "3.14", "0.5", "1,234,567", "$5.50", "¥100", "£3", "€9.99", "12.0", "007",
           "99999999999999999999"]

KATAKANA = ["コンニチハ", "ラーメン", "ヴァイオリン", "ティーカップ", "ディズニー", "ッ", "ン", "ー", "キャット", "シェフ",
            "ツァ", "ヂャ", "ヲォ", "ヶ", "ヽ", "カナ。", "フュ", "アァ"]


def kanwa_sample(count, seed):
    import pickle
    import pykakasi
    path = os.path.join(os.path.dirname(pykakasi.__file__), "data", "kanwadict4.db")
    table = pickle.load(open(path, "rb"))
    keys = sorted(k for entries in table.values() for k in entries)
    rng = random.Random(seed)
    return [keys[rng.randrange(len(keys))] for _ in range(count)]


def main():
    sentences = []
    for text in SENTENCES:
        norm = japanese.text_normalize(text)
        entry = {"text": text, "normalized": norm, "pieces": japanese.tokenizer.tokenize(norm)}
        try:
            phones, tones, word2ph = japanese.g2p(norm)
            entry.update({"phones": phones, "tones": tones, "word2ph": word2ph})
        except AssertionError:
            entry["fails"] = True   # g2p asserts every phoneme is a MeloTTS symbol
        sentences.append(entry)
    kakasi_inputs = KAKASI + kanwa_sample(400, 21)
    result = {
        "sentences": sentences,
        "kakasi": [{"text": t, "kana": japanese.conv.do(t)} for t in kakasi_inputs],
        "numbers": [{"text": t, "words": japanese.japanese_convert_numbers_to_words(t)} for t in NUMBERS],
        "katakana": [{"text": t, "phonemes": japanese.kata2phoneme(t)} for t in KATAKANA],
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "melo_japanese_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print({k: len(v) for k, v in result.items()}, sum(1 for s in sentences if s.get("fails")), "failing sentences")


if __name__ == "__main__":
    main()
