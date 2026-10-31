"""Fish Speech's text tokenizers on a set of texts, from the releases' own code.

Fish Speech 1.5: fish_speech/tokenizer.py's FishTokenizer (tiktoken over tokenizer.tiktoken with FISH_TIKTOKEN_PATTERN and
the release's special tokens). Fish Speech 1.4: Hugging Face tokenizers over tokenizer.json (byte-level BPE behind the
Digits and Punctuation pre-tokenizers). Records the ids of each text in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech_tokenizer_reference.json.

Default: verify; --write: regenerate. Usage: python fishspeech_tokenizer_reference.py --fish15 <fish-speech v1.5.1 checkout>
--model15 <fishaudio/fish-speech-1.5 dir> --model14 <fishaudio/fish-speech-1.4 dir>. Needs tiktoken and tokenizers.
"""
import argparse
import os
import sys

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fishspeech_tokenizer_reference.json"

TEXTS = [
    "Hello, world! It's a beautiful day, isn't it? We'll see.",
    "The year 2024 had 366 days; pi is 3.14159...",
    "你好，世界！今天是2024年3月5日。",
    "こんにちは、世界。カタカナとひらがな！",
    "Spaces   in    a row, tabs\tand\nnew\r\nlines\n\n\nend",
    "A literal quirk:   ?!x and  ?! y and ?!",
    "Symbols $100 + 5% = <tag> ^_^ ~tilde~ `code` |pipe| @user #hash",
    "Emoji 😀 and supplementary 𠀀 letters, café naïve Ünïcödé.",
    "<|im_start|>system\nSpeak out the provided text.<|im_end|><|im_start|>user\nHi there<|im_end|><|im_start|>assistant\n<|voice|>",
    "  leading spaces and trailing   ",
    "Quotes “curly” ‘single’ «guillemets» — dash – en-dash … ellipsis",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fish15", required=True)
    parser.add_argument("--model15", required=True)
    parser.add_argument("--model14", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.fish15))
    from fish_speech.tokenizer import FishTokenizer
    from tokenizers import Tokenizer

    fish15 = FishTokenizer(os.path.join(args.model15, "tokenizer.tiktoken"))
    fish14 = Tokenizer.from_file(os.path.join(args.model14, "tokenizer.json"))
    out = {
        "texts": TEXTS,
        "ids15": [fish15.encode(t) for t in TEXTS],
        "ids14": [fish14.encode(t, add_special_tokens=False).ids for t in TEXTS],
        "semantic_begin15": fish15.semantic_begin_id,
        "im_end15": fish15.get_token_id("<|im_end|>"),
        "semantic14": fish14.token_to_id("<|semantic|>"),
        "im_end14": fish14.token_to_id("<|im_end|>"),
    }
    if args.write:
        _fixture.save(FIXTURE, out, __file__)
        print(f"wrote {FIXTURE}")
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Fish Speech tokenizers")
    for key, value in out.items():
        check.exact(key, stored[key], value)
    check.report()


if __name__ == "__main__":
    main()
