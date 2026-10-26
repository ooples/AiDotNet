"""The NATSpeech-family Chinese text front end through its own code (MoonInTheRiver/DiffSinger,
data_gen/tts/txt_processors/zh.py with utils/text_norm.py, chn_text_norm's NSWNormalizer), with pypinyin 0.50.0.

The two files are read from a DiffSinger checkout (--diffsinger, or the two files' directory with --files: zh.py and
text_norm.py); zh.py's imports of DiffSinger's PUNCS and base processor are supplied as the same definitions. Records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/natspeech_chinese_reference.json, for Chinese texts with dates, money,
telephone numbers, fractions, percentages, measure words, digit strings, decimals, full-width punctuation, Latin letters
and neutral tones: NSWNormalizer's output, the cleaned text and the phones (process with use_tone, empty initials
dropped as encoding the space-joined phones drops them).
Default: verify; --write: regenerate. Needs pypinyin 0.50.0 (the .venv-melo312 environment).
"""
import argparse
import os
import sys
import types

try:
    import torch  # noqa: F401  (_fixture records its version)
except ImportError:
    sys.modules["torch"] = types.SimpleNamespace(__version__="not installed")
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/natspeech_chinese_reference.json"
TEXTS = [
    "你好，世界！这是一个测试。",
    "日期：1999年2月20日或09年3月15号。",
    "金钱：12块5，34.5元，20.1万元。",
    "固话：0595-23865596或23880880。",
    "手机：+86 19859213959或15659451527。",
    "分数：32477/76391。百分数：80.03%，有62％的概率。",
    "编号：31520181154418。",
    "纯数：2983.07克或12345.60米，3456万吨，2938个。",
    "今天吃了115个小笼包231个馒头",
    "特殊：O2O或B2C。",
    "我们的孩子们在玩，他说：“好的吧。”",
    "２０２３年１０月，ＡＢＣ公司（总部）——成立了！！",
    "数字一百零二和两千零二十，以及0.5和100。",
    "",
]


def load(name, path):
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), module.__dict__)
    sys.modules[name] = module
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", required=True, help="directory with DiffSinger's zh.py and text_norm.py")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for package in ("data_gen", "data_gen.tts", "data_gen.tts.txt_processors", "utils"):
        sys.modules[package] = types.ModuleType(package)
    sys.modules["data_gen.tts.data_gen_utils"] = types.SimpleNamespace(PUNCS="!,.?;:")

    class BaseTxtProcessor:
        pass
    sys.modules["data_gen.tts.txt_processors.base_text_processor"] = types.SimpleNamespace(BaseTxtProcessor=BaseTxtProcessor)
    norm = load("utils.text_norm", os.path.join(args.files, "text_norm.py"))
    zh = load("data_gen.tts.txt_processors.zh", os.path.join(args.files, "zh.py"))
    stored = None if args.write else _fixture.load(FIXTURE)
    cases = []
    for text in TEXTS:
        translated = text.translate(zh.TxtProcessor.table)
        phs, clean = zh.TxtProcessor.process(text, {"use_tone": True})
        cases.append({"text": text, "nsw": norm.NSWNormalizer(translated).normalize(remove_punc=False), "clean": clean,
                      "phones": " ".join(phs).split()})
    data = {"cases": cases}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        for c in cases:
            print(c["nsw"][:40], "|", " ".join(c["phones"])[:100])
        return
    check = _fixture.Comparison("NATSpeech Chinese front end")
    check.exact("cases", stored["cases"], data["cases"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
