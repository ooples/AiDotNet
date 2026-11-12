"""Reference fixture for MeloTTS's Korean front end: melo/text/korean.py's text_normalize and g2p, and g2pkk on its own.

Runs MeloTTS's own korean.py (myshell-ai/MeloTTS at 2091453) with g2pkk 0.1.2 on its Linux branch (python-mecab-ko
1.3.7 with mecab-ko-dic 2.1.1; on Windows g2pkk would load eunjeon instead), jamo 0.4.1, nltk's cmudict and
transformers' kykim/bert-kor-base tokenizer. Run in a venv without mecab-python3 (the two MeCab packages collide on a
case-insensitive file system).

g2pkk replaces English words and numbers by iterating a Python set, so an input where one such match contains another
reads differently under different hash seeds; the fixture rejects those inputs.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_korean_reference.json.
"""
import importlib.util
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc", "MeloTTS", "melo", "text"))

melo = types.ModuleType("melo")
melo.__path__ = [os.path.dirname(TEXT)]
sys.modules["melo"] = melo
package = types.ModuleType("melo.text")
package.__path__ = [TEXT]
sys.modules["melo.text"] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symbols = load("melo.text.symbols", os.path.join(TEXT, "symbols.py"))
package.symbols = symbols.symbols
package.punctuation = symbols.punctuation
load("melo.text.ko_dictionary", os.path.join(TEXT, "ko_dictionary.py"))
korean = load("melo.text.korean", os.path.join(TEXT, "korean.py"))

import g2pkk.g2pkk as g2pkk_module  # noqa: E402

g2pkk_module.platform = types.SimpleNamespace(system=lambda: "Linux")   # MeloTTS's platform: python-mecab-ko

SENTENCES = [
    "전 제 일의 가치와 폰타인 대중들이 한 일의 의미를 잘 압니다.",
    "앞으로도 전 제 일에 자부심을 갖고 살아갈 겁니다",
    "나의 친구가 mp3 file 3개를 다운받고 있다",
    "안녕하세요, 만나서 반갑습니다!",
    "그는 1999년에 태어났고, 서울에서 25년 동안 살았다.",
    "사과 2개와 배 10개, 그리고 책 3권을 샀다.",
    "읽고 밟고 넓은 들판을 걸었다.",
    "값이 싸서 많이 샀어요.",
    "KOREA의 IT 산업과 AI 기술, CCTV와 SNS.",
    "의사의 의견은 무엇인가요?",
    "신라와 백제, 그리고 고구려와 결단력.",
    "1+1 행사와 2+1 행사가 있습니다.",
    "오늘 학교에서 밥을 먹고 집에 와서 game을 했다",
    "그 사람 좀 old school이야",
    "漢字가 섞인 文章입니다.",
    "할 수 있을 걸 알았지만, 갈 데가 없었다.",
    "닭이 울고 흙을 밟았다.",
    "같이 굳이 해돋이를 보러 가자.",
]

G2PKK = ["나의 친구가 mp3 file 3개를 다운받고 있다", "의견란", "결단력", "생산량", "우리 3시 10분에 만나자.",
         "123456789번", "20살", "0개", "1000000원", "같이", "굳이", "읽다", "넓다", "밟고", "핥다", "젊은", "앉고",
         "할 것을", "갈 데가", "의사의", "희망", "computer", "school", "strength", "Kim", "xylophone",
         "1234567890123456개"]

NUMBERS = [("123,456,789", True), ("123,456,789", False), ("0", True), ("20", False), ("20", True), ("10", True),
           ("10000", True), ("100000000", True), ("1000000000000", True), ("1001", True), ("31", False),
           ("99", False), ("1234567890123456", True), ("100010001", True)]


def guard(text):
    words = set(re.findall("[A-Za-z']+", text))
    numbers = set(re.findall(r"([\d][\d,]*)", text))
    for group in (words, numbers):
        for a in group:
            for b in group:
                if a != b and a in b:
                    raise SystemExit(f"{text!r}: {a!r} is inside {b!r}; g2pkk's set order would decide the output")


def main():
    g2p = None
    sentences = []
    for text in SENTENCES:
        norm = korean.text_normalize(text)
        for token in norm.split():
            guard(token)
        phones, tones, word2ph = korean.g2p(norm)
        sentences.append({"text": text, "normalized": norm, "pieces": korean.tokenizer.tokenize(norm), "phones": phones,
                          "tones": tones, "word2ph": word2ph})
    g2p = korean.g2p_kr
    results = []
    for text in G2PKK:
        guard(text)
        results.append({"text": text, "output": g2p(text)})
    from g2pkk.numerals import process_num
    numbers = [{"num": n, "sino": s, "output": process_num(n, sino=s)} for n, s in NUMBERS]
    result = {"sentences": sentences, "g2pkk": results, "numbers": numbers}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "melo_korean_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print({k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main()
