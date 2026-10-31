"""Reference fixture for MaskGCT's Japanese front end (Amphion models/tts/maskgct/g2p/g2p/japanese.py at 26f6883) and
the OpenJTalk pipeline under it (pyopenjtalk 0.3.4 with open_jtalk_dic_utf_8-1.11): for each text, pyopenjtalk's
run_frontend features (MeCab's words through text2mecab and the NJD passes), make_label's full-context labels, and
MaskGCT's g2p_ result (the phone string and the vocabulary ids).

Runs in the release's environment (refsrc/maskgct-docker) like maskgct_g2p_reference.py:

    docker run --rm -v <Amphion>:/amphion -v <repo>:/repo -w /amphion maskgct-g2p \
        python /repo/tools/reference-data/maskgct_japanese_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_japanese_reference.json.
"""
import json
import sys
import types

sys.path.insert(0, "/amphion")
sys.modules["LangSegment"] = types.SimpleNamespace(setfilters=lambda filters: None)

import pyopenjtalk  # noqa: E402
from pykakasi import kakasi  # noqa: E402
from models.tts.maskgct.g2p.g2p_generation import g2p  # noqa: E402

CASES = [
    "こんにちは、今日はいい天気ですね。",
    "私は東京に住んでいます。",
    "吾輩は猫である。名前はまだ無い。",
    "2024年10月18日に、3人で1,500円のランチを食べました！",
    "カタカナのコンピューターと、ひらがなのことば？",
    "すもももももももものうち",
    "人々は時々、色々な場所へ行く。",
    "ＡＢＣとabcと１２３",
    "東京特許許可局長の許可が下りた。",
    "「こんにちは」と彼は言った…",
    # words Open JTalk cannot read: pykakasi's reading (溶, 飼, 菴), or none (龘, 彁: unknown to both)
    "溶と飼の字、菴に住む。",
    "龘は難しい。",
    "彁という字。",
    "𠮷野家に行く。",
    "鬱々とした日々。",
    "αとβ。",
    "ｘｙｚの話",
    "魑魅魍魎の跋扈",
    "薔薇と檸檬",
]

# pykakasi 2.3.0's convert, which the front end falls back to: script changes, the kanwa dictionary, a reading
# with a context (色 after そん), unknown kanji (and the character after one skipped), variant kanji, symbols.
KAKASI_CASES = ["溶", "飼", "菴", "龘", "彁", "そん色ない", "色々", "溶液と以上", "ひらがなカタカナ漢字abc", "龘あい",
                "𠮷野家", "々", "αβγ", "ｘｙｚ", "ー長音", "東京、大阪。", "髙橋", "葛󠄀城"]


def main():
    cases = []
    for text in CASES:
        features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(features)
        phonemes, tokens = g2p(text, sentence=None, language="ja")
        cases.append({"text": text, "features": features, "labels": labels, "phonemes": phonemes, "tokens": tokens})
    converter = kakasi()
    kakasi_cases = [{"text": t, "segments": [[r["orig"], r["kana"]] for r in converter.convert(t)]} for t in KAKASI_CASES]
    path = "/repo/tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_japanese_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"pyopenjtalk": pyopenjtalk.__version__, "cases": cases, "kakasi": kakasi_cases}, handle, ensure_ascii=False)
        handle.write("\n")
    for c in cases:
        print(len(c["features"]), len(c["labels"]), len(c["tokens"]))


if __name__ == "__main__":
    main()
