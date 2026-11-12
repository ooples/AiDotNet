"""Chatterbox's multilingual text front end reference (resemble-ai/chatterbox 5de7a54: mtl_tts.punc_norm and
models/tokenizers/tokenizer.MTLTokenizer over grapheme_mtl_merged_expanded_v1.json and Cangjie5_TC.json), with
spacy-pkuseg 1.0.2's spacy_ontonotes segmenter and pykakasi 2.3.0. Records pkuseg's cut of Chinese passages, punc_norm of
assorted strings, and the token ids MTLTokenizer.encode gives each sentence in its language (and with no language).
Usage: python tools/reference-data/chatterbox_multilingual_text_reference.py --chatterbox <checkout>/src
       --model-dir <ResembleAI/chatterbox snapshot> --pkuseg-home <dir holding spacy_ontonotes/> [--write]   (in .venv-chatterbox)
"""
import argparse
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chatterbox_multilingual_text_reference.json"

CHINESE = [
    "我爱北京天安门，天安门上太阳升。",
    "今天是2025年10月3日，气温是23.5摄氏度。",
    "他说：“人工智能（AI）正在改变世界”，我们要认真对待。",
    "中华人民共和国成立于一九四九年十月一日。",
    "小明在ＡＢＣ公司工作了三年半，月薪一万二千元。",
    "研究生命的起源是一个很有意思的课题",
    "南京市长江大桥 欢迎您 www.example.com/路径_测试",
    "这个东西，那个东西，都是好东西！你觉得呢？",
]

SENTENCES = {
    "ar": "مرحبا بكم في عالم الذكاء الاصطناعي.",
    "da": "Det er en dejlig dag i København.",
    "de": "Die Straße ist heute sehr ruhig.",
    "el": "Καλημέρα σας, ΟΔΥΣΣΕΑΣ και ΣΟΦΙΑΣ.",
    "en": "Hello there; this is a test... of “quotes” — and dashes.",
    "es": "¿Dónde está la biblioteca? ¡Aquí!",
    "fi": "Hyvää huomenta, mitä kuuluu?",
    "fr": "C'est l'été, ça va très bien.",
    "hi": "नमस्ते, आप कैसे हैं? मैं ठीक हूँ।",
    "it": "Perché la città è così bella?",
    "ja": "今日は東京へ行きます。カタカナとひらがなも、漢字も混ざっています",
    "ko": "안녕하세요, 반갑습니다. 오늘 날씨가 좋네요",
    "ms": "Selamat pagi, apa khabar?",
    "nl": "Het is een mooie dag in Amsterdam.",
    "no": "Jeg bor i Bergen, og det regner ofte.",
    "pl": "Zażółć gęślą jaźń, proszę.",
    "pt": "Ação e coração não são iguais.",
    "sv": "Det är en vacker dag i Göteborg.",
    "sw": "Habari za asubuhi, rafiki yangu.",
    "tr": "İstanbul'da IŞIK ve ılık hava var.",
    "zh": "我爱北京天安门，今天是2025年。你好吗",
}

PUNC = ["", "hello world", "  multiple   spaces here ", "Done…", "a: b; c — d", "你好。", "質問？", "ok,", "hi、"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chatterbox", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pkuseg-home", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    os.environ["PKUSEG_HOME"] = args.pkuseg_home
    sys.path.insert(0, args.chatterbox)
    from spacy_pkuseg import pkuseg
    import chatterbox.models.tokenizers.tokenizer as tk
    from chatterbox.mtl_tts import punc_norm

    segmenter = pkuseg()
    cangjie = os.path.join(args.model_dir, "Cangjie5_TC.json")
    with mock.patch.object(tk, "hf_hub_download", lambda **kw: cangjie):
        tokenizer = tk.MTLTokenizer(os.path.join(args.model_dir, "grapheme_mtl_merged_expanded_v1.json"))
    data = {
        "pkuseg": [{"text": t, "words": segmenter.cut(t)} for t in CHINESE],
        "punc_norm": [{"text": t, "normalized": punc_norm(t)} for t in PUNC],
        "encode": [{"language": lang, "text": text, "ids": tokenizer.encode(punc_norm(text), language_id=lang),
                    "no_language": tokenizer.encode(punc_norm(text))} for lang, text in SENTENCES.items()],
    }
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Chatterbox multilingual text")
    for key in ("pkuseg", "punc_norm", "encode"):
        check.exact(key, stored[key], data[key])
    check.report()


if __name__ == "__main__":
    main()
