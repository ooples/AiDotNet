"""GLM-4's tokenizer as GLM-4-Voice-9B's remote code builds it (tokenization_chatglm.py, tiktoken 0.14).

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_tokenizer_reference.json the ids AutoTokenizer gives
(trust_remote_code, special tokens added) for texts that exercise cl100k's pattern (case-insensitive contractions,
three-digit number groups, newline runs, punctuation before letters, CJK, emoji), the added tokens (chat roles,
audio tokens) and GLM-4-Voice's prompt template.

Usage (default: verify; --write: regenerate), with glm-4-voice-9b's tokenizer files in <dir>:
  python tools/reference-data/glm_tokenizer_reference.py --model <dir> [--write]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_tokenizer_reference.json"
TEXTS = [
    "Hello world! I'M sure it'S fine, they'RE here.",
    "Numbers: 1234567 and 3.14159, 2024-10-29",
    "line one\r\n\r\n  line two\n\n\tindented   spaced  ",
    "!hello ?world ...dots ,comma",
    "你好，世界！GLM-4-Voice 支持中文和英文。",
    "emoji \U0001F600 and \U00020000 rare",
    "<|begin_of_audio|><|audio_5|><|audio_16382|><|end_of_audio|>",
    "<|system|>\nUser will provide you with a speech instruction. Do it step by step. First, think about the instruction "
    "and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. <|user|>\n"
    "<|begin_of_audio|><|audio_1|><|audio_2|><|end_of_audio|><|assistant|>streaming_transcription\n",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    import regex
    ids = [tokenizer(text)["input_ids"] for text in TEXTS]
    decoded = [tokenizer.decode(i[2:], spaces_between_special_tokens=False) for i in ids]  # as the release decodes
    # cl100k's pre-tokens of the ordinary text (added tokens removed), as tiktoken's regex splits it.
    pattern = regex.compile(tokenizer.pat_str)
    plain = [regex.sub(r"<\|[^|]*\|>", "", text) for text in TEXTS]
    splits = [pattern.findall(text) for text in plain]
    if args.write:
        _fixture.save(FIXTURE, {"texts": TEXTS, "ids": ids, "decoded": decoded, "plain": plain, "splits": splits}, __file__)
        print("wrote", FIXTURE, [len(i) for i in ids])
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("GLM-4 tokenizer")
    check.exact("ids", stored["ids"], ids)
    check.exact("decoded", stored["decoded"], decoded)
    check.exact("splits", stored["splits"], splits)
    check.report()


if __name__ == "__main__":
    main()
