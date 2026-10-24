"""EmotiVoice's text front end (netease-youdao/EmotiVoice frontend.py, frontend_cn.py, frontend_en.py, cn2an/): its
resources and its output on a test corpus.

EmotiVoice converts text with its own front end: Arabic numerals become Chinese numerals (its vendored cn2an), the
text splits into Chinese and other runs, jieba segments the Chinese and pypinyin (with pypinyin-dict's CC-CEDICT
phrases loaded) reads each word in TONE3 style, split into initials and finals; other runs are English, read from
LibriSpeech's lexicon or, for words it lacks, by g2p_en. The phonemes index the released token list.

Exports, from the pinned packages and an EmotiVoice checkout (--emotivoice):
- src/TextToSpeech/FrontEnd/Resources/emotivoice_pinyin_chars.tsv.gz and emotivoice_pinyin_phrases.tsv.gz: pypinyin
  0.55.0's first readings with pypinyin-dict 0.9.0's CC-CEDICT phrases loaded, in TONE3 style with the neutral tone as
  5, computed by pypinyin's own converter: "text<TAB>reading reading ..." (MIT).
- src/TextToSpeech/FrontEnd/Resources/emotivoice_lexicon.tsv.gz: lexicon/librispeech-lexicon.txt as read_lexicon reads
  it (lower-cased words, the first pronunciation of each): "word<TAB>phone phone ...".
- src/TextToSpeech/FrontEnd/Resources/emotivoice_tokens.txt.gz and emotivoice_speakers.txt.gz: the released token
  list (data/youdao/text/tokenlist) and speaker list (speaker2), one per line (Apache-2.0).
and records tn_chinese and g2p_cn_en of a corpus in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/
emotivoice_frontend_reference.json. Default: verify both; --write: regenerate them.
"""
import argparse
import gzip
import importlib.metadata
import io
import os
import sys
import types

import _fixture

PINNED = {"jieba": "0.42.1", "pypinyin": "0.55.0", "pypinyin-dict": "0.9.0", "g2p-en": "2.1.0", "nltk": "3.10.4"}
OUT = "src/TextToSpeech/FrontEnd/Resources/"
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/emotivoice_frontend_reference.json"
CORPUS = [
    "我来到北京，清华大学",
    "在昏暗狭小的房间内，我父亲躺在窗前的地板上，全身素白，显得身子特别长。",
    "他光着双脚，脚趾头怪模怪样地向外翻着，一双亲切的手平静地放在胸前，手指头也是弯曲的。",
    "第一章",
    "今天是2023年11月20日，气温是-3.5度，涨了105%。",
    "他有10001个苹果和0.25个梨，还有007号。",
    "你好呀！今天心情怎么样？我很好；谢谢。",
    "女儿和儿子一起去了公园，吃了绿色的鱼。",
    "这是一段中英文混合的文本，包含English words和Chinese characters。",
    "我喜欢用iPhone 15 Pro拍照，也喜欢AI。",
    "Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine",
    "Happy New Year! It's a wonderful day, isn't it?",
    "I'm gonna tell you what we cannot do; John's dog's bone.",
    "The quick brown fox jumps over the lazy dog.",
    "Zyxquorbal flimbertastic glorpington, said 'snarfblat' quietly.",
    "He scored 42 points in 2024.",
    "嗯，呣，哦。",
    "一九八四年，《动物农场》出版了。",
    "OK，我们开始吧。Let's go!",
    "这个问题——很难说……",
    "长城有2.1万公里长。",
    "这座城市有102000人，去年是10002000元，还有3005000个。",
    "她说：“我们明天见。”",
]


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def check_versions():
    for name, version in PINNED.items():
        found = importlib.metadata.version(name)
        if found != version:
            raise SystemExit(f"{name} {found} is installed; the resources are built from {version}.")


def load(root):
    sys.path.insert(0, root)
    # The release's cn2an/ is a namespace package (no __init__.py), which an installed cn2an would shadow; its
    # requirements do not install cn2an, so the release runs the vendored copy. Bind the package to it.
    vendored = types.ModuleType("cn2an")
    vendored.__path__ = [os.path.join(root, "cn2an")]
    sys.modules["cn2an"] = vendored
    cwd = os.getcwd()
    os.chdir(root)
    try:
        import frontend
        from frontend_en import read_lexicon, G2p
        lexicon = read_lexicon(os.path.join(root, "lexicon", "librispeech-lexicon.txt"))
        g2p = G2p()
    finally:
        os.chdir(cwd)
    return frontend, lexicon, g2p


def resources(root, lexicon):
    from pypinyin import Style
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    from pypinyin.converter import UltimateConverter

    # The converter pypinyin.pinyin(text, style=Style.TONE3, neutral_tone_with_five=True) builds.
    converter = UltimateConverter(v_to_u=False, neutral_tone_with_five=True)

    def reading(text):
        return " ".join(r[0] for r in converter.convert(text, Style.TONE3, False, "default", True))

    files = {}
    chars = [f"{chr(code)}\t{reading(chr(code))}" for code in sorted(PINYIN_DICT)]
    files[OUT + "emotivoice_pinyin_chars.tsv.gz"] = gz("\n".join(chars) + "\n")
    phrases = [f"{phrase}\t{reading(phrase)}" for phrase in sorted(PHRASES_DICT)]
    files[OUT + "emotivoice_pinyin_phrases.tsv.gz"] = gz("\n".join(phrases) + "\n")
    files[OUT + "emotivoice_lexicon.tsv.gz"] = gz("".join(f"{w}\t{' '.join(p)}\n" for w, p in lexicon.items()))
    for name, source in (("emotivoice_tokens.txt.gz", "tokenlist"), ("emotivoice_speakers.txt.gz", "speaker2")):
        with open(os.path.join(root, "data", "youdao", "text", source), encoding="utf-8") as handle:
            files[OUT + name] = gz("".join(line.strip() + "\n" for line in handle.readlines()))
    return files


def run(frontend, lexicon, g2p):
    cases = []
    for text in CORPUS:
        cases.append({"text": text, "tn": frontend.tn_chinese(text), "phonemes": frontend.g2p_cn_en(text, g2p, lexicon)})
    return {"corpus": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotivoice", required=True, help="a checkout of netease-youdao/EmotiVoice")
    parser.add_argument("--write", action="store_true", help="regenerate the resources and fixture (default: verify them)")
    args = parser.parse_args()
    check_versions()
    root = os.path.abspath(args.emotivoice)
    frontend, lexicon, g2p = load(root)
    files = resources(root, lexicon)
    data = run(frontend, lexicon, g2p)
    if args.write:
        for relative, blob in files.items():
            with open(_fixture.fixture_path(relative), "wb") as handle:
                handle.write(blob)
            print(f"wrote {relative} ({len(blob)} bytes)")
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    bad = []
    for relative, blob in files.items():
        with open(_fixture.fixture_path(relative), "rb") as handle:
            if handle.read() != blob:
                bad.append(relative)
    if _fixture.load(FIXTURE)["corpus"] != data["corpus"]:
        bad.append(FIXTURE)
    if bad:
        raise SystemExit("MISMATCH: " + ", ".join(bad))
    print("EmotiVoice front end: resources and corpus match the reference")


if __name__ == "__main__":
    main()
