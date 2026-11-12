"""Whisper's multilingual text tokenizer reference: openai-whisper 20250625's whisper.tokenizer.get_encoding (tiktoken
0.9.0 over assets/multilingual.tiktoken, MIT), the text tokenizer CosyVoice-300M builds with get_tokenizer(multilingual,
num_languages=100, language='en', task='transcribe') and calls with allowed_special='all'.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/whisper_tiktoken_reference.json, for 99 languages
(Whisper large-v2 and earlier) and 100 (large-v3, CosyVoice): the vocabulary size, the special tokens' ids, and the
encodings of texts that exercise each branch of the pre-tokenizer pattern (contractions, letters, digits, other
symbols, whitespace runs with and without the trailing-space lookahead), CJK, emoji, a supplementary-plane letter,
combining marks, and special tokens written inline. Also the SHA-256 of the vocabulary file the library ships.
Default: verify; --write: regenerate.
"""
import hashlib
import os

import tiktoken
import whisper
import whisper.tokenizer as whisper_tokenizer

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/whisper_tiktoken_reference.json"
TEXTS = [
    "Hello world",
    "I'm here, it's done; they've gone, we'll see, I'd say, you're right.",
    "'S 'T contractions are case-sensitive: DON'T",
    "Numbers 12345 and 3.14159, plus 1,000,000!",
    "  leading spaces,   inner runs   and trailing   ",
    "tabs\tand\nnew\n\nlines\r\n end",
    "你好，世界！CosyVoice 是一个语音合成模型。",
    "日本語のテキストと한국어 텍스트",
    "emoji 😀🎉 and symbols ©®™ → ∑",
    "rare 𠀀𠀁 letters",
    "café naïve combining marks",
    "<|en|>Hello<|endoftext|> inline <|zh|>你好<|0.00|>",
    "a" * 300,
    "",
    " ",
    "!!!'s?",
    "mixed123abc 4x4 x²",
    # Supplementary-plane letters and digits next to other classes: a UTF-16 regex would read each half of the
    # surrogate pair as a symbol and cut these differently.
    "x𠀀y 𠀀!? 𝐀𝐁c x𝟙𝟚3",
    # A supplementary letter before a contraction: read as a symbol it would absorb the apostrophe ("𠀀'", "s"), where
    # the pattern gives "𠀀" then "'s".
    "𠀀's 𝐀'll x𝟙'd",
    # Odd runs of one repeated pair: equal-rank pairs, where the leftmost must merge first.
    "aaa zzzzz !!!!!!! ababa xxxxxxx",
]


def specials(encoding):
    return {name: encoding.encode_single_token(name) for name in sorted(encoding.special_tokens_set)}


def record():
    data = {"texts": TEXTS}
    for languages in (99, 100):
        encoding = whisper_tokenizer.get_encoding("multilingual", num_languages=languages)
        data[str(languages)] = {
            "n_vocab": encoding.n_vocab,
            "specials": specials(encoding),
            "tokens": [encoding.encode(text, allowed_special="all") for text in TEXTS],
        }
    path = os.path.join(os.path.dirname(whisper.__file__), "assets", "multilingual.tiktoken")
    with open(path, "rb") as handle:
        data["vocabulary_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    data["tiktoken"] = tiktoken.__version__
    return data


def main():
    args = _fixture.parse_args(__doc__)
    data = record()
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    committed = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Whisper tiktoken")
    for key in ("texts", "99", "100", "vocabulary_sha256"):
        check.exact(key, committed[key], data[key])
    check.report()


if __name__ == "__main__":
    main()
