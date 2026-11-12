"""Reference fixture for MeloTTS's Chinese front end: melo/text/chinese.py's text_normalize and g2p on a set of sentences.

Runs MeloTTS's own chinese.py and tone_sandhi.py (myshell-ai/MeloTTS at 2091453) with its pinned dependencies (jieba
0.42.1, pypinyin 0.50.0, cn2an 0.5.22), the package's other language modules left out. Records, per sentence, the
normalized text, the phones, the tones and the phones per character (or that g2p's alignment assert fails), and
jieba's part-of-speech cut of each sentence; and the tone sandhi's readings of a spread of jieba's dictionary words
(or that the sandhi raises).

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_chinese_reference.json.
"""
import importlib.util
import json
import os
import re
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


load("melotext.symbols", os.path.join(TEXT, "symbols.py"))
load("melotext.tone_sandhi", os.path.join(TEXT, "tone_sandhi.py"))
chinese = load("melotext.chinese", os.path.join(TEXT, "chinese.py"))
import jieba.posseg as psg  # noqa: E402

SENTENCES = [
    "你好，世界！",
    "这是一个示例文本：你好！这是一个测试....",
    "啊！但是《原神》是由,米哈游自主，[研发]的一款全新开放世界冒险游戏",
    "我有3个苹果和12.5公斤的米，还有2024年的1000000元。",
    "看一看，听一听，一天一夜，一段时间，第一名，一零零。",
    "不怕，不要，看不懂，不是不行，对不起。",
    "我很好，你也好吗？所有人都喜欢纸老虎和蒙古包。",
    "展览馆里有很多老虎，小李想买五把雨伞。",
    "奶奶和妈妈在桌上吃饭，孩子们上来下去。",
    "这个东西怎么样？我们的朋友在哪儿？",
    "老头儿在胡同儿里玩儿，一点儿也不累。",
    "嗯，呣，好吧，走吧，是呢，对啊。",
    "他说：“明天见！”她回答：‘好的’——再见～",
    "Hello你好world世界，chemistry课程。",
    "我买了两个、三个和几个西瓜；每个都很甜。",
    "试试看，想想办法，研究研究，高高兴兴。",
    "电脑软件需要更新，手机屏幕坏了。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "早上好，今天天气真不错，我们去公园散步吧。",
    "请把窗户打开，让新鲜空气进来。",
    "一二三四五六七八九十，百千万亿。",
    "Ｗｉ－Ｆｉ信号很弱(真的)。",
    "他是一位非常有名的科学家，研究过很多问题。",
    "我想起来了，你说的那本书在书架上。",
    "能不能帮我一下？好不好？行不行？",
]


def main():
    out = []
    for sentence in SENTENCES:
        norm = chinese.text_normalize(sentence)
        try:
            phones, tones, word2ph = chinese.g2p(norm)
        except AssertionError:
            # chinese.g2p asserts one reading per character ("Sometimes it will crash"); the port raises there too.
            out.append({"text": sentence, "normalized": norm, "fails": True})
            continue
        pattern = r"(?<=[{0}])\s*".format("".join(chinese.punctuation))
        cuts = [[[w, p] for w, p in psg.lcut(re.sub("[a-zA-Z]+", "", s))]
                for s in re.split(pattern, norm) if s.strip() != ""]
        out.append({"text": sentence, "normalized": norm, "phones": phones, "tones": tones, "word2ph": word2ph,
                    "cuts": cuts})
    # Words on their own: every 580th entry of jieba's dictionary (with its tag), read by pypinyin and the tone sandhi.
    import jieba
    with open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8") as handle:
        entries = [line.split() for line in handle if line.strip()]
    words = []
    for word, _, tag in entries[::580]:
        initials, finals = chinese._get_initials_finals(word)
        try:
            finals = chinese.tone_modifier.modified_tone(word, tag, finals)
        except IndexError:
            words.append({"word": word, "pos": tag, "fails": True})
            continue
        words.append({"word": word, "pos": tag, "initials": initials, "finals": finals})
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "melo_chinese_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"sentences": out, "words": words}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(out)} sentences, {len(words)} words")


if __name__ == "__main__":
    main()
