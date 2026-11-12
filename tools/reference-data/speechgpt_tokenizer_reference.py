"""SpeechGPT's tokenizer as the release runs it: transformers 4.29's slow LlamaTokenizer over fnlp/SpeechGPT-7B-cm's
tokenizer.model and added_tokens.json.

Run with transformers 4.29.2 (the release's 4.29.0.dev0 line) and sentencepiece; without the tokenizers package that
version misreads tokenizer_config.json's AddedToken entries and appends <s> and </s> as new ids, so the directory is
read through a copy holding only tokenizer.model, added_tokens.json and a plain config, and the ids are <s> (1) followed
by tokenize()'s tokens: the splitting at the added tokens, their white-space stripping and the per-piece SentencePiece
encoding that matter.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechgpt_tokenizer_reference.json the ids of the release's
prompts and some edge cases. Usage (default: verify; --write: regenerate):
  python tools/reference-data/speechgpt_tokenizer_reference.py --model <SpeechGPT-7B-cm dir> [--write]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import LlamaTokenizer  # noqa: E402  (before any torch stand-in: transformers probes for torch)
try:
    import torch  # noqa: F401
except ImportError:
    # The transformers 4.29 environment needs no torch; _fixture imports it for tensor helpers this script does not use.
    import types
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["torch"].__version__ = "not installed"
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speechgpt_tokenizer_reference.json"
META = ("You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is "
        "developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by "
        "the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n")
TEXTS = [
    META + "[Human]: Can you please read this sentence out loud? This is input: Today is a sunny day. <eoh>. [SpeechGPT]: ",
    META + "[Human]: Can you transcribe the speech into a written format? This is input: <sosp><12><5><991><0><eosp> <eoh>. [SpeechGPT]: ",
    "[Human]: This is the speech instruction: <sosp><7><7><8><eosp>. You can do it step by step. <eoh>. [SpeechGPT]: [tq] hi; [ta] hello; [ua] <sosp><3><eosp><eoa>",
    "Hello  world.\nNew line <12> <13>x and 100 dollars, café ☃ <eos>",
    " leading space and trailing <eosp>  ",
    "<1000> is not a unit but <999> is",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    import transformers
    assert transformers.__version__.startswith("4.29"), transformers.__version__
    work = tempfile.mkdtemp()
    for name in ("tokenizer.model", "added_tokens.json"):
        shutil.copy(os.path.join(args.model, name), work)
    with open(os.path.join(work, "tokenizer_config.json"), "w") as handle:
        json.dump({"tokenizer_class": "LlamaTokenizer", "add_bos_token": True, "add_eos_token": False, "clean_up_tokenization_spaces": False}, handle)
    tokenizer = LlamaTokenizer.from_pretrained(work)
    ids = [[1] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in TEXTS]
    if args.write:
        _fixture.save(FIXTURE, {"texts": TEXTS, "ids": ids}, __file__)
        print("wrote", FIXTURE, [len(i) for i in ids])
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("SpeechGPT tokenizer")
    for n, (expected, actual) in enumerate(zip(stored["ids"], ids)):
        check.exact(f"text {n}", expected, actual)
    check.report()


if __name__ == "__main__":
    main()
