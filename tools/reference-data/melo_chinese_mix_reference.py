"""Reference fixture for MeloTTS's mixed Chinese-English front end: melo/text/chinese_mix.py's text_normalize and g2p
(its default impl='v2') on a set of sentences.

Runs MeloTTS's own chinese_mix.py, chinese.py, tone_sandhi.py and english.py (myshell-ai/MeloTTS at 2091453) with
their pinned dependencies (jieba 0.42.1, pypinyin 0.50.0, cn2an 0.5.22, inflect 7.0.0, g2p_en 2.1.0, transformers'
bert-base-uncased and bert-base-multilingual-uncased tokenizers), the package's other language modules left out
(english.py needs only japanese.distribute_phone, loaded from japanese.py itself). Records, per sentence, the normalized
text, the multilingual word pieces of the English runs, the phones, the tones and the phones per character or piece.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_chinese_mix_reference.json.
"""
import ast
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


symbols = load("melotext.symbols", os.path.join(TEXT, "symbols.py"))
package.symbols = symbols.symbols   # the package's __init__ does `from .symbols import *`, so `symbols` is the list
japanese = types.ModuleType("melotext.japanese")
source = open(os.path.join(TEXT, "japanese.py"), encoding="utf-8").read()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "distribute_phone")
exec(compile(ast.Module(body=[node], type_ignores=[]), "japanese.py", "exec"), japanese.__dict__)
sys.modules["melotext.japanese"] = japanese
load("melotext.english", os.path.join(TEXT, "english.py"))
load("melotext.tone_sandhi", os.path.join(TEXT, "tone_sandhi.py"))
load("melotext.chinese", os.path.join(TEXT, "chinese.py"))
mix = load("melotext.chinese_mix", os.path.join(TEXT, "chinese_mix.py"))

SENTENCES = [
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。",
    "今天下午，我们准备去shopping mall购物，然后晚上去看一场movie。",
    "我们现在 also 能够 help 很多公司 use some machine learning 的 algorithms 啊!",
    "NFT啊！chemistry 但是《原神》是由,米哈游自主，  [研发]的一款全.新开放世界.冒险游戏",
    "这个iPhone的价格是5999元，比Samsung贵了1000块。",
    "Hello, 世界! How are you? 我很好。",
    "他用Python和C语言写了一个program，叫做AiDotNet。",
    "请打开Wi-Fi，然后connect到network上。",
    "我不知道你说的 quixotic 是什么意思，也不懂flibbertigibbet。",
    "OK，我们 tomorrow 见，bye bye！",
    "一起去看一看那个 exhibition 吧，不要迟到。",
    "Café 和 naïve 都是外来词。",
    "老板说：“deadline 是下周一”，大家都很 stressed。",
    "春眠不觉晓，处处闻啼鸟。",
    "   multiple   spaces\tand\ttabs 中文   ",
]


def main():
    out = []
    pattern = r"(?<=[{0}])\s*".format("".join(mix.punctuation))
    for sentence in SENTENCES:
        norm = mix.text_normalize(sentence)
        phones, tones, word2ph = mix.g2p(norm)
        english = [mix.tokenizer.tokenize(run) for s in re.split(pattern, norm) if s.strip() != ""
                   for run in re.findall(r"[a-zA-Z\s]+", s)]
        out.append({"text": sentence, "normalized": norm, "pieces": english, "phones": phones, "tones": tones,
                    "word2ph": word2ph})
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "melo_chinese_mix_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"sentences": out}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(out)} sentences")


if __name__ == "__main__":
    main()
