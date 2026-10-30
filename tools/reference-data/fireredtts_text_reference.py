"""FireRedTTS's text front end, from the release's code (FireRedTeam/FireRedTTS, main branch: fireredtts/modules/
text_normalizer and tokenizer).

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_text_reference.json, for each text of a set:
- preprocess_text, rettt, lingua's English/Chinese detection of the cleaned text, TextNormalizer.tn (text and language);
- VoiceBpeTokenizer.encode (the language detected);
- text_split (sentencex's segmentation, Chinese windows, merging; the release drops a long last piece);
and lingua's detection of probe strings over the characters the cleaner keeps, and sentencex's segmentation of a few
paragraphs.

Usage (WeTextProcessing needs pynini, which has no Windows build; run in the firered-ref Docker image: megatts-ref with
inflect 7.4.0, lingua-language-detector 2.0.2, sentencex 0.6.1, tiktoken 0.7.0 and WeTextProcessing 1.0.3):
  docker run --rm -v <repo>:/repo -v <FireRedTTS checkout>:/fr -w /repo firered-ref \
      python tools/reference-data/fireredtts_text_reference.py --firered /fr [--write]
The script imports the release's modules without torch (stub packages), which the front end does not use.
"""
import argparse
import json
import os
import sys
import types

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fireredtts_text_reference.json"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TEXTS = [
    "小红书，是中国大陆的网络购物和社交平台，成立于二零一三年六月。",
    "今天是2024年10月17日，气温25.5℃，我花了￥128.50买了3本书。",
    "我、真是hello USA啊？谢谢你world！",
    "Hello world, this is FireRedTTS speaking.",
    "Dr. Smith met Mr. Brown at 10:30 a.m. on Jan. 5th, 2023, i.e. yesterday.",
    "The NASA and FBI teams spent $1,250 on 42 GPUs in the U.S. btw.",
    "I have 3 apples and 17 oranges @ home.",
    "  Ｆｕｌｌｗｉｄｔｈ　ｔｅｘｔ！ＡＢＣ　１２３  ",
    "Café naïve résumé — façade “quoted” ‘single’ «guillemets».",
    "他说：“你好……我很好。。。”然后走了~~~",
    "表情(＾▽＾)真开心o(*￣▽￣*)ブ哈哈",
    "Line one\nline two\tTabbed\r\nwindows.",
    "【重要】《三体》是刘慈欣的作品｛科幻｝",
    "αβγ δεζ Greek letters ΑΒΓ.",
    "〇一二三〇 and 々 marks",
    "ALL CAPS SENTENCE WITH NUMBERS 123.",
    "WHAT?! 42",
    "Mixed 中文 and English 混合 text 123 测试。",
    "zero width​space and‍zwj and nbsp and � replacement",
    "Emoji test 😀👍 with symbols ☺ ✂ ★.",
    "Ms. Johnson and Mrs. Lee have a Ph.D from St. Andrews on Sun. at 5 p.m. via Main St. and Oak Rd.",
    "Sept. 3 and Sep. 4 and May. 5, Tues. and Thur. meetings, e.g. reviews & b.t.w. notes.",
    "A.D. 100 and 300 B.C. and the ad. disclaimer",
    "[laugh] I can't believe it [breath] really",
    "就是 [rep] 就是 我 [hic] 我 [elong] 好",
    "Numbers: 1000000, 2,500, 3.14159 and 007.",
    "电话号码是13800138000，邮箱是test@example.com。",
    "",
    "   ",
    "!!!",
    "第1名和第2名，1/2的人，50%的概率，-5度。",
    "The quick brown fox jumps over the lazy dog. It was a sunny day in the park. Children were playing football happily. Then it started raining heavily over the whole town.",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。床前明月光，疑是地上霜。举头望明月，低头思故乡。白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "这是一个非常非常长的句子没有任何标点符号一直写下去直到超过三十个字符为止看看会怎么切分这段文本内容呢真的很长很长很长很长",
    "Short one. Another short. A third one that is somewhat longer than the others here.",
    "He said \"Stop. Now.\" and left. (See fig. 3.) Then e.g. more. Visit U.S. soon!",
    "First paragraph here.\n\nSecond paragraph after a blank line. With two sentences.",
    "Yahoo! is a company. What!? Really?! Yes!!! OK.",
    "第一句话。第二句话！第三句话？第四句话；第五句话：结束",
    "Price is 3.5 dollars. Version 2.0 released. Call 911. Done.",
    "hello",
    "UPPER lower UPPER",
]

PROBES = [
    "hello world", "123 456", "...!?", "", " ", "αβγ", "αβγ hello", "〇 abc", "〇〇 aa",
    "〇 a", "a 〇", "〇 a α", "ab 〇〇", "abc 〇〇〇 αβγ", "〆", "〆 a",
    "〱〱 a", "〇 α", "〮〮〮 a", "〡〢 hello", "hello 々々々 x",
    "αβγ δεζ ηθι κλμ x", "x", "X Y Z 〇〇",
]

PARAGRAPHS = [
    "Mr. Smith went to Washington. He arrived at 5 p.m. and left.",
    "The value is 3.14. It is pi. Use it wisely!",
    "She said \"Hello. How are you?\" Then she left.",
    "See ref.[1] for details. Next sentence [2] here. Final.",
    "Email me at john.doe@example.com. Thanks.",
    "Is it e.g. this? Or i.e. that? I.B.M. is a company.",
    "One!Two?Three. four five. Six.",
    "Ends without terminator",
    "Multiple...dots... here. And here!!",
    "A sentence.  Double spaced.   Triple.",
]


def install_stubs(root):
    """The release's packages without fireredtts/modules/__init__.py (which imports torch and the networks)."""
    for name, path in [("fireredtts", "fireredtts"), ("fireredtts.modules", "fireredtts/modules")]:
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(root, path)]
        sys.modules[name] = module
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firered", required=True, help="the FireRedTTS checkout (main branch)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.firered)
    install_stubs(root)
    from fireredtts.modules.text_normalizer import normalize
    from fireredtts.modules.text_normalizer.utils import text_split
    from fireredtts.modules.tokenizer.tokenizer import VoiceBpeTokenizer
    from lingua import Language
    from sentencex import segment

    tokenizer = VoiceBpeTokenizer()
    tn = tokenizer.tn_engine
    names = {Language.ENGLISH: "en", Language.CHINESE: "zh"}

    def detect(text):
        language = tn.language_detector.detect_language_of(text)
        return names.get(language) if language is not None else None

    cases = []
    for text in TEXTS:
        cleaned = normalize.preprocess_text(text)
        abbreviated = normalize.rettt(cleaned)
        normalized, language = tn.tn(text.strip())
        cases.append({
            "text": text, "preprocessed": cleaned, "rettt": abbreviated, "detected": detect(abbreviated),
            "normalized": normalized, "language": language, "ids": tokenizer.encode(text),
            "split": text_split(text),
        })
    data = {
        "cases": cases,
        "probes": [{"text": t, "detected": detect(t)} for t in PROBES],
        "segments": [{"text": t, "sentences": list(segment("en", t))} for t in PARAGRAPHS],
        "zh_segments": [{"text": t, "sentences": list(segment("zh", t))} for t in TEXTS if any("一" <= c <= "龥" for c in t)],
    }
    path = os.path.join(REPO, *FIXTURE.split("/"))
    if args.write:
        data["generator"] = "tools/reference-data/" + os.path.basename(__file__)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)
        print(f"wrote {FIXTURE}: {len(cases)} texts")
        return
    stored = json.load(open(path, encoding="utf-8"))
    failures = [k for k in ("cases", "probes", "segments", "zh_segments") if stored[k] != data[k]]
    if failures:
        raise SystemExit(f"FireRedTTS text: mismatches in {failures}")
    print("FireRedTTS text: all match")


if __name__ == "__main__":
    main()
