"""Reference fixture for GPT-SoVITS's LangSegmenter: text/LangSegmenter's getTexts, split-lang, BudouX and fast_langdetect.

Runs GPT-SoVITS's own LangSegmenter (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (split-lang 2.1.1,
budoux 0.9.3, fast-langdetect 1.1.0 over fasttext-predict 0.9.2.4), with fastText's lid.176.bin where GPT-SoVITS keeps
it (GPT_SoVITS/pretrained_models/fast_langdetect):

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -v <dir with lid.176.bin>:/ref/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect \
        -w /ref/GPT-SoVITS gptsovits-ref python /repo/tools/reference-data/gptsovits_langsegmenter_reference.py

Also writes the embedded BudouX models (budoux_ja.json.gz, budoux_zh_hans.json.gz, the package's JSON byte for byte)
with --write-resources, else verifies them.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_langsegmenter_reference.json.
"""
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))

from text.LangSegmenter import LangSegmenter  # noqa: E402
import fast_langdetect  # noqa: E402
import budoux  # noqa: E402

TEXTS = [
    "MyGO?,你也喜欢まいご吗？",
    "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。",
    "当时ThinkPad T60刚刚发布，一同推出的还有一款名为Advanced Dock的扩展坞配件。这款扩展坞通过连接T60底部的插槽，扩展出包括PCIe在内的一大堆接口，并且自带电源，让T60可以安装桌面显卡来提升性能。",
    "你好，世界！Hello world! こんにちは世界！안녕하세요 세계!",
    "我今天买了3个苹果，花了15.5元。",
    "The price is 100 dollars, 好贵啊！",
    "今日は2024年3月5日です。",
    "우리는 2023년에 만났어요.",
    "这是一个测试。This is a test. これはテストです。",
    "東京タワーと台北101と서울타워。",
    "Python和C++都是编程语言，Java也是。",
    "123",
    "Hello",
    "你好",
    "中文English中文English",
    "我喜欢看アニメ和漫画。",
    "這是繁體中文的句子，和简体中文不同。",
    "早上好！\n晚上好！\nGood night!",
    "AI技术，比如GPT-4和Claude，正在改变世界。",
    "我的电话是13812345678，邮箱是test@example.com。",
    "ドラえもん、のび太、しずかちゃん。",
    "他说：“OK，没问题。”",
    "1.5倍速播放，2x speed。",
    "한국어와 中文과 English를 섞어서 써요.",
]

DETECT = ["你好世界", "こんにちは", "Hello world", "안녕하세요", "Bonjour le monde", "HELLO WORLD", "衬衫", "まいご", "MyGO?,",
          "這是繁體", "Привет мир", "123", "Hola, ¿cómo estás?", "the", "a"]


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def main():
    models = os.path.join(os.path.dirname(budoux.__file__), "models")
    for source, name in (("ja.json", "budoux_ja.json.gz"), ("zh-hans.json", "budoux_zh_hans.json.gz")):
        with open(os.path.join(models, source), "rb") as handle:
            data = gz(handle.read())
        path = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources", name)
        if "--write-resources" in sys.argv:
            with open(path, "wb") as handle:
                handle.write(data)
        else:
            with open(path, "rb") as handle:
                if handle.read() != data:
                    raise SystemExit(f"{name} differs")

    segments = []
    for text in TEXTS:
        entry = {"text": text}
        for default in ("", "zh", "ja", "ko"):
            entry[f"default_{default or 'none'}"] = [[t["lang"], t["text"]] for t in LangSegmenter.getTexts(text, default)]
        segments.append(entry)
    detections = []
    for text in DETECT + TEXTS:
        one = fast_langdetect.detect(text.replace("\n", " "), model="full")
        five = fast_langdetect.detect(text.replace("\n", "").strip(), model="full", k=5, threshold=0.01)
        detections.append({"text": text, "top": [[r["lang"], r["score"]] for r in one],
                           "five": [[r["lang"], r["score"]] for r in five]})
    ja = budoux.load_default_japanese_parser()
    zh = budoux.load_default_simplified_chinese_parser()
    parses = [{"text": t, "ja": ja.parse(t), "zh": zh.parse(t)} for t in TEXTS]
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "gptsovits_langsegmenter_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"segments": segments, "detections": detections, "budoux": parses}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(segments)} texts, {len(detections)} detections")


if __name__ == "__main__":
    main()
