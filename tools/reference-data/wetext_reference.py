"""Builds the embedded WeTextProcessing 1.0.4.1 text-normalization grammars
(src/TextToSpeech/FrontEnd/Resources/wetext_{zh,en}_tn_{tagger,verbalizer}.fst.gz) and their reference fixture.

MegaTTS 3 normalizes text with WeTextProcessing's Chinese and English normalizers, which load the compiled grammars the
wheel ships (tn/zh_tn_tagger.fst and the others; the constructor's options only matter when it compiles them, which it
does not when they exist) and optimize them on load (pynini's Fst.optimize, which renumbers states and arcs). The
resources are those optimized machines in OpenFst's binary vector format, gzipped.

The fixture (tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wetext_reference.json) holds, for Chinese and English texts
covering the grammars' classes (cardinals, decimals, dates, times, money, measures, fractions, math, scores, telephone
numbers, addresses, Roman numerals, whitelisted words, plain text) and 150 seeded mixes of them each, the tagger's output and the normalized text, from the
normalizers MegaTTS 3 builds (ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False) and
EnNormalizer(overwrite_cache=False)).

Usage: python wetext_reference.py [--write]  (default: verify). Needs pynini 2.1.6 and WeTextProcessing 1.0.4.1; the
megatts-ref Docker image has them:
  docker run --rm -v <repo>:/work -w /work megatts-ref python tools/reference-data/wetext_reference.py --write
"""
import argparse
import gzip
import io
import os
import sys
import tempfile
import types

import pynini
from importlib_resources import files

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch  # noqa: F401  (_fixture records its version)
except ImportError:
    sys.modules["torch"] = types.SimpleNamespace(__version__="not installed")
import _fixture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wetext_reference.json"
NAMES = ["zh_tn_tagger", "zh_tn_verbalizer", "en_tn_tagger", "en_tn_verbalizer"]

CHINESE = [
    "你好，世界！这是一个测试。",
    "今天是2024年3月15日，气温是-3.5℃，湿度80%。",
    "会议在下午3:30开始，持续2个小时。",
    "这件衣服打八折，原价￥199.99，现价159.99元。",
    "他跑了100米，用时9.58秒，比分是3:2。",
    "电话号码是13812345678，邮编100080。",
    "1/3加上2/5等于11/15，x+y=z。",
    "我们公司有1,234,567名员工，增长了12.5%。",
    "这个房间有25平方米，重量为3.5kg，长度10cm。",
    "第1名，第二十三名，No.5，第3季度。",
    "2008年北京奥运会，共有204个国家参加。",
    "价格从100元到200元不等，约合$15～$30。",
    "现在时间是早上8点15分30秒。",
    "欢迎收听《TED Talks Daily》，我是您的主持人，Elise Hugh。",
    "他说：“我今天很开心！”然后笑了。",
    "温度在零下5度到35度之间，PM2.5指数为75。",
    "版本号v1.2.3已经发布，下载量突破10万次。",
    "一九八四年，他在3号楼502室住过。",
    "哇塞！家人们，你们太好运了。整整三十包啊！",
    "繁體中文轉換測試，這個軟件很好用。",
    "那儿有一个小孩儿在玩儿。",
    "嗯，啊，这个嘛，我觉得还行吧。",
    "５０公斤，ＡＢＣ，全角数字１２３。",
    "2024-03-15 14:30:00",
    "比例是1:3，概率是0.05。",
    "",
]
ENGLISH = [
    "Hello world, this is MegaTTS 3.",
    "The meeting is at 3:30 pm on March 15, 2024.",
    "It costs $19.99, or about £15.50 and €17.",
    "Call me at 555-123-4567 or (800) 555-0199.",
    "He ran 100 m in 9.58 s, at 37.6 km/h.",
    "About 1,234,567 people, i.e. 12.5% of them.",
    "Chapter IV, World War II, Louis XIV.",
    "Visit www.example.com or mail abc@example.com.",
    "The ratio is 3/4 and the score was 2-1.",
    "Dr. Smith lives at 221B Baker St. since 1985.",
    "On 01/02/2023 the temperature was -5 degrees.",
    "From 10 to 20 kg, 5-10 minutes, the 1990s and the '80s.",
    "Washington CNN When President Donald Trump declared in the House Chamber this week",
    "Mr. and Mrs. Jones paid $3 million for 2 acres.",
    "It is 6:45 a.m. and the 3rd, 22nd and 101st entries.",
    "",
]


def generated(language, count, seed):
    """Seeded texts mixing the grammars' classes, to exercise ties between equal-weight paths."""
    import random
    r = random.Random(seed)
    num = lambda: r.choice([str(r.randint(0, 99)), str(r.randint(100, 99999)), f"{r.randint(1, 999):,}" if language == "english" else str(r.randint(1000, 10**9)),
                            f"{r.randint(0, 999)}.{r.randint(0, 99)}", f"-{r.randint(1, 50)}", f"{r.randint(1, 9)}/{r.randint(2, 12)}"])
    if language == "chinese":
        parts = [lambda: f"{num()}元", lambda: f"{r.randint(1990, 2030)}年{r.randint(1, 12)}月{r.randint(1, 28)}日", lambda: f"{r.randint(0, 23)}:{r.randint(0, 59):02d}",
                 lambda: f"{num()}%", lambda: f"{num()}公里", lambda: f"{num()}kg", lambda: f"第{num()}名", lambda: f"{r.randint(0, 9)}:{r.randint(0, 9)}",
                 lambda: "我们", lambda: "今天", lambda: "，", lambda: "。", lambda: num(), lambda: f"{num()}℃", lambda: f"￥{num()}", lambda: "ABC"]
    else:
        parts = [lambda: f"${num()}", lambda: f"{r.choice(['Jan.', 'March', 'Oct'])} {r.randint(1, 28)}, {r.randint(1900, 2030)}", lambda: f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['am', 'pm', 'p.m.'])}",
                 lambda: f"{num()}%", lambda: f"{num()} km", lambda: f"{num()} kg", lambda: f"{r.randint(1, 120)}{r.choice(['st', 'nd', 'rd', 'th'])}", lambda: f"{r.randint(0, 9)}-{r.randint(0, 9)}",
                 lambda: "the", lambda: "Dr.", lambda: ",", lambda: ".", lambda: num(), lambda: r.choice(["II", "XIV", "IV"]), lambda: "€" + num(), lambda: "and"]
    sep = "" if language == "chinese" else " "
    return [sep.join(r.choice(parts)() for _ in range(r.randint(1, 6))) for _ in range(count)]


def build_resources():
    data = {}
    for name in NAMES:
        fst = pynini.Fst.read(str(files("tn") / f"{name}.fst")).optimize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.fst")
            fst.write(path)
            raw = open(path, "rb").read()
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as g:
            g.write(raw)
        data[name] = buffer.getvalue()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    resources = build_resources()
    zh = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
    en = EnNormalizer(overwrite_cache=False)

    def case(normalizer, text):
        try:
            tagged = normalizer.tag(text)
            return {"text": text, "tagged": tagged, "normalized": normalizer.verbalize(tagged)}
        except Exception as e:  # no path through the grammar
            return {"text": text, "error": type(e).__name__}
    data = {"chinese": [case(zh, t) for t in CHINESE + generated("chinese", 150, 181)],
            "english": [case(en, t) for t in ENGLISH + generated("english", 150, 191)]}
    if args.write:
        for name, blob in resources.items():
            with open(os.path.join(RESOURCES, f"wetext_{name}.fst.gz"), "wb") as f:
                f.write(blob)
        _fixture.save(FIXTURE, data, __file__)
        for c in data["chinese"] + data["english"]:
            print(repr(c.get("normalized", c.get("error"))))
        print("wrote", {k: len(v) for k, v in resources.items()})
        return
    for name, blob in resources.items():
        with open(os.path.join(RESOURCES, f"wetext_{name}.fst.gz"), "rb") as f:
            assert f.read() == blob, f"{name}: the resource differs from the wheel's grammar"
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("WeTextProcessing")
    check.exact("chinese", stored["chinese"], data["chinese"])
    check.exact("english", stored["english"], data["english"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
