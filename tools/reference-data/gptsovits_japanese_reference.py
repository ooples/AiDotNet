"""Reference fixture for GPT-SoVITS's Japanese front end: text/japanese.py's text_normalize and g2p (with prosody).

Runs GPT-SoVITS's own text package (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (pyopenjtalk 0.4.1 with
its open_jtalk_dic_utf_8-1.11), from the GPT-SoVITS checkout, so that japanese.py compiles and loads its user dictionary
(text/ja_userdic/userdict.csv) as the release does:

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_japanese_reference.py [--write-resource]

Records, per sentence, the normalized text, the phones and clean_text's phones (v2), and pyopenjtalk's run_frontend
words of each sentence. --write-resource also writes the embedded user dictionary (gptsovits_ja_userdict.csv.gz, the
CSV byte for byte); without it the committed copy is verified.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_japanese_reference.json.
"""
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
os.environ.setdefault("version", "v2")

from text import japanese, cleaner  # noqa: E402
import pyopenjtalk  # noqa: E402

SENTENCES = [
    "こんにちは。",
    "Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！",
    "今日はいい天気ですね？",
    "私は東京に住んでいます、毎日電車で会社に行きます。",
    "ちょっと待ってください……本当ですか？！",
    "１２３円と45％の割引、2024年3月5日です。",
    "きゃりーぱみゅぱみゅはカッコいい！",
    "日本語の音声合成はとても面白いです。",
    "「はい」と彼は言った；そして、去った。",
    "ａｂｃとABCとabc。",
    "愛　してる  よ",
    "ﾊﾝｶｸｶﾀｶﾅもあります。",
    "銀河鉄道の夜、宮沢賢治。",
    "おはようございます！！！",
    "々ゝヽ、繰り返し記号。",
    "church house と corps。",
    "主殿は来たか？",
    "一、二、三、四、五。",
    "東京都庁の展望台からの景色は最高です。",
    "それは〜無理だよ…",
    # Words of the user dictionary (each reads differently without it), full width and as typed.
    "ｊｏｓｉａｈです。ｄｏｗｄｉｎｇです。ｒａｂｂｉｔです。",
    "ｕｎｂｒｉｄｌｅｄとｃａｃｈｏとａｍｂｕｌａｎｃｅｓ。",
    "trollingerとlargentとgroとtsaiとunionist。",
    "rabbitがambulancesでjosiahを運んだ。",
]


def main():
    resource = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources", "gptsovits_ja_userdict.csv.gz")
    with open(japanese.USERDIC_CSV_PATH, "rb") as handle:
        csv = handle.read()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(csv)
    if "--write-resource" in sys.argv:
        with open(resource, "wb") as handle:
            handle.write(buffer.getvalue())
    else:
        with open(resource, "rb") as handle:
            if handle.read() != buffer.getvalue():
                raise SystemExit("gptsovits_ja_userdict.csv.gz differs from text/ja_userdic/userdict.csv")

    out = []
    for text in SENTENCES:
        norm = japanese.text_normalize(text)
        entry = {"text": text, "normalized": norm, "phones": japanese.g2p(norm),
                 "clean": cleaner.clean_text(text, "ja", "v2")[0]}
        words = []
        for piece in [s for s in japanese.re.split(japanese._japanese_marks, japanese.symbols_to_japanese(norm).lower()) if s]:
            words.append([[f["string"], f["pos"], f["pron"], f["acc"], f["mora_size"], f["chain_flag"]]
                          for f in pyopenjtalk.run_frontend(piece)])
        entry["frontend"] = words
        out.append(entry)
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "gptsovits_japanese_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"sentences": out}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(out)} sentences")


if __name__ == "__main__":
    main()
