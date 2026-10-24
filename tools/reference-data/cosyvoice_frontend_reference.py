"""CosyVoice text front-end reference: FunAudioLLM/CosyVoice's CosyVoiceFrontEnd.text_normalize (cosyvoice/cli/frontend.py,
with cosyvoice/utils/frontend_utils.py) as it runs without the wetext / ttsfrd normalizers (text_frontend == ''), with
CosyVoice-300M's text tokenizer (openai-whisper 20250625's get_tokenizer(multilingual, num_languages=100)) and
inflect 7.3.1, CosyVoice's pin. Pass a checkout of FunAudioLLM/CosyVoice with --cosyvoice.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_frontend_reference.json, for each text, the
sentences of text_normalize(text, split=True) and the string of text_normalize(text, split=False): Chinese rules
(blanks, ², '.', ' - ', brackets, a trailing comma), English numbers spelled out, splitting at 60-80 units with a short
last sentence merged, closing quotes kept with their sentence, punctuation-only sentences dropped, and special-token
text passed through. Default: verify; --write: regenerate.
"""
import argparse
import importlib.machinery
import sys
import types
from unittest import mock

import inflect
import whisper.tokenizer

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_frontend_reference.json"
TEXTS = [
    "",
    "Hello world",
    "  Hello there, how are you today?  ",
    "I have 3 apples and 12 oranges, and 2024 was a good year; 1000000 people agree.",
    "The value is 3.14 and the code is 007!",
    "收到漂流瓶的人看到后也许会回复你。 这是第二句话！第三句话？",
    "我们在 2024 年 3 月 去了 北京 和 Shanghai city 。",
    "面积是 50m² 和 20m³，速度很快 - 非常快。",
    "（这是括号）【这是方括号】——破折号`反引号`",
    "这句话以逗号结尾，，",
    "他说：“你好。”然后离开了。她回答：“再见！”",
    "第一行\n第二行\n第三行",
    "这是一段很长的中文文本，用来测试分句的逻辑。" * 6,
    "This is a long English paragraph that keeps going. " * 12,
    "Short one. Another short one! A third? Fourth; fifth: done",
    "...!!!",
    "<|en|>Hello<|endoftext|> special tokens skip the front end 123",
    "x² and ³ superscripts",
    "Quote at the end.\" Then more text.",
]


def frontend():
    sys.path.insert(0, ARGS.cosyvoice)
    for name in ["onnxruntime", "torchaudio", "torchaudio.compliance", "torchaudio.compliance.kaldi"]:
        stub = mock.MagicMock()
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules.setdefault(name, stub)
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, num_languages=100, language="en", task="transcribe")
    state = types.SimpleNamespace(tokenizer=tokenizer, allowed_special="all", inflect_parser=inflect.engine(), text_frontend="")
    return lambda text, split: CosyVoiceFrontEnd.text_normalize(state, text, split=split)


def record(normalize):
    cases = []
    for text in TEXTS:
        try:
            sentences, joined = normalize(text, True), normalize(text, False)
            cases.append({"text": text, "sentences": sentences, "unsplit": joined})
        except Exception as error:  # the reference raises on some inputs; record that it does
            cases.append({"text": text, "error": type(error).__name__})
    return {"cases": cases, "inflect": inflect.__version__ if hasattr(inflect, "__version__") else "7.3.1"}


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--write", action="store_true")
    ARGS = parser.parse_args()
    data = record(frontend())
    if ARGS.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    check = _fixture.Comparison("CosyVoice front end")
    check.exact("cases", _fixture.load(FIXTURE)["cases"], data["cases"])
    check.report()


if __name__ == "__main__":
    main()
