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
]


def main():
    cases = []
    for text in CASES:
        features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(features)
        phonemes, tokens = g2p(text, sentence=None, language="ja")
        cases.append({"text": text, "features": features, "labels": labels, "phonemes": phonemes, "tokens": tokens})
    path = "/repo/tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_japanese_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"pyopenjtalk": pyopenjtalk.__version__, "cases": cases}, handle, ensure_ascii=False)
        handle.write("\n")
    for c in cases:
        print(len(c["features"]), len(c["labels"]), len(c["tokens"]))


if __name__ == "__main__":
    main()
