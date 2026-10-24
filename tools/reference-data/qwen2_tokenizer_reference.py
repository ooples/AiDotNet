"""Qwen2 text tokenizer reference, CosyVoice 2's: transformers 5.18.0's AutoTokenizer over Qwen/Qwen2.5-0.5B's
tokenizer files (Apache-2.0) with the special tokens CosyVoice2Tokenizer adds (cosyvoice/tokenizer/tokenizer.py), whose
encode is tokenizer([text])["input_ids"][0]. Pass the directory holding the tokenizer files with --qwen.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/qwen2_tokenizer_reference.json the ids of texts that exercise
NFC, every branch of the pre-tokenizer pattern (case-insensitive contractions, letters with one leading non-letter,
single digits, symbol runs with trailing newlines, whitespace runs ending in newlines, the trailing-space lookahead),
CJK, emoji, supplementary-plane letters, non-ASCII whitespace and special tokens written inline; the vocabulary size;
and the SHA-256 of the vocabulary and merges files the library ships. Default: verify; --write: regenerate.
"""
import argparse
import hashlib
import os

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/qwen2_tokenizer_reference.json"
SPECIALS = ['<|im_start|>', '<|im_end|>', '<|endofprompt|>', '[breath]', '<strong>', '</strong>', '[noise]', '[laughter]',
            '[cough]', '[clucking]', '[accent]', '[quick_breath]', "<laughter>", "</laughter>", "[hissing]", "[sigh]",
            "[vocalized-noise]", "[lipsmack]", "[mn]"]
TEXTS = [
    "Hello world",
    "I'm here, IT'S done; they'VE gone, we'Ll see.",
    "Numbers 12345 and 3.14159, plus 1,000,000!",
    "  leading spaces,   inner runs   and trailing   ",
    "tabs\tand\nnew\n\nlines\r\n end\n\n  \n x",
    "(parenthesised) [bracketed] {braced} \"quoted\" -dash ...dots!!!\n\n",
    "收到漂流瓶的人看到后也许会回复你。这是第二句话！",
    "日本語のテキストと한국어 텍스트",
    "emoji 😀🎉 and symbols ©®™ → ∑",
    "rare 𠀀𠀁 letters x𠀀y 𠀀's",
    "café naïve combining marks",
    "no break　ideographic em space",
    "<|im_start|>Hello<|im_end|> [laughter]haha[laughter] <|endofprompt|>你好[breath]",
    "<strong>bold</strong> and [quick_breath] then [vocalized-noise]",
    "a" * 300,
    "",
    " ",
    "\n",
    "mixed123abc 4x4 x² ①②",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen)
    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
                                  "additional_special_tokens": SPECIALS})
    data = {"texts": TEXTS, "ids": [tokenizer([t])["input_ids"][0] for t in TEXTS], "vocabulary": len(tokenizer),
            "added": {t: tokenizer.convert_tokens_to_ids(t) for t in ["<|endoftext|>"] + SPECIALS}}
    for name in ("vocab.json", "merges.txt"):
        with open(os.path.join(args.qwen, name), "rb") as handle:
            data[name + "_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    committed = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Qwen2 tokenizer")
    for key in ("texts", "ids", "vocabulary", "added", "vocab.json_sha256", "merges.txt_sha256"):
        check.exact(key, committed[key], data[key])
    check.report()


if __name__ == "__main__":
    main()
