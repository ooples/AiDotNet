"""Step-Audio's prompts as the release builds them, under transformers 4.48.3 (its pin), whose LlamaTokenizerFast prepends
<s> to every encode() and matches the control tokens (<|BOT|>, <|EOT|>) as tokens:

- the TTS prompt: StepAudioTTS.tokenize (tts.py) imported from the Step-Audio checkout (its model-loading imports stubbed),
  its system prompts read from tts.py's source, for plain, speaker, RAP and humming text with a dual-codebook prompt code;
- the chat prompt: StepAudio.apply_chat_template (stepaudio.py) for text and audio turns, then encode().

Usage: PYTHONPATH=<transformers 4.48.3> python tools/reference-data/stepaudio_prompt_reference.py --stepaudio <Step-Audio
checkout> --tokenizer <dir with tokenizer.model and tokenizer_config.json> [--write]
"""
import argparse
import ast
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/stepaudio_prompt_reference.json"
PROMPT_CODE = [65536 + 3, 65536 + 900, 65536 + 1024 + 7, 65536 + 1024 + 4000, 65536 + 1024 + 12]
TTS_CASES = [
    {"text": "今天天气真好，我们去公园散步吧。", "prompt_text": "你好，我是婷婷。", "speaker": ""},
    {"text": "(高兴1)Hello there, nice to meet you!", "prompt_text": "Hi.", "speaker": "Tingting"},
    {"text": "(RAP)我们一起唱起来", "prompt_text": "嗯。", "speaker": "Tingting"},
    {"text": "（哼唱）啦啦啦", "prompt_text": "嗯。", "speaker": "Tingting"},
]
CHAT_CASES = [
    [{"role": "user", "content": "你好，我叫小明，你叫什么名字？"}],
    [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": {"type": "audio", "audio": "AUDIO"}},
     {"role": "assistant", "content": "好的"}, {"role": "user", "content": "再说一遍"}],
]
ENCODE_TEXTS = ["你好，世界！", "Hello there", "<audio_5>hello", "hi<|EOT|>there", "<|BOT|>human\n问题<|EOT|>",
                "  leading spaces", "(高兴1)今天", "1234 apples", "naïve 😀"]
AUDIO_TOKENS = "".join(f"<audio_{x}>" for x in (1, 2, 1024 + 5, 1024 + 6, 1024 + 7))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stepaudio")
    parser.add_argument("--tokenizer")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import transformers
    assert transformers.__version__ == "4.48.3", transformers.__version__
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for name in ("cosyvoice", "cosyvoice.cli", "cosyvoice.cli.cosyvoice"):
        module = types.ModuleType(name)
        module.CosyVoice = object
        sys.modules[name] = module
    utils = types.ModuleType("utils")
    utils.load_optimus_ths_lib = lambda *a: None
    for n in ("load_audio", "speech_adjust", "volumn_adjust"):
        setattr(utils, n, lambda *a, **k: None)
    sys.modules["utils"] = utils
    sys.path.insert(0, args.stepaudio)
    import tts as tts_module

    source = open(os.path.join(args.stepaudio, "tts.py"), encoding="utf-8").read()
    prompts = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == "sys_prompt_dict":
            prompts = ast.literal_eval(node.value)
    tts = tts_module.StepAudioTTS.__new__(tts_module.StepAudioTTS)
    tts.tokenizer = tokenizer
    tts.sys_prompt_dict = prompts
    tts_ids = [tts.tokenize(c["text"], c["prompt_text"], c["speaker"], PROMPT_CODE) for c in TTS_CASES]

    sys.modules.setdefault("tokenizer", types.ModuleType("tokenizer")).StepAudioTokenizer = object
    import stepaudio as stepaudio_module
    chat = stepaudio_module.StepAudio.__new__(stepaudio_module.StepAudio)
    chat.encode_audio = lambda path: AUDIO_TOKENS
    chat_ids = [tokenizer.encode(chat.apply_chat_template(case)) for case in CHAT_CASES]

    encodings = [{"text": t, "ids": tokenizer.encode(t)} for t in ENCODE_TEXTS]
    data = {"encodings": encodings, "prompt_code": PROMPT_CODE, "audio_tokens": AUDIO_TOKENS, "system_prompts": prompts,
            "tts": [{**c, "ids": ids} for c, ids in zip(TTS_CASES, tts_ids)],
            "chat": [{"messages": c, "ids": ids} for c, ids in zip(CHAT_CASES, chat_ids)]}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, [len(x) for x in tts_ids], [len(x) for x in chat_ids])
        return
    check = _fixture.Comparison("Step-Audio prompts")
    stored = _fixture.load(FIXTURE)
    if stored["tts"] != data["tts"] or stored["chat"] != data["chat"] or stored["encodings"] != data["encodings"]:
        check.failures.append("prompts differ")
    check.report(exact=True)


if __name__ == "__main__":
    main()
