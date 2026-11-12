"""BERT tokenizer reference: Hugging Face transformers' BertTokenizer (the reference BasicTokenizer + WordPiece) with
the bert-base-uncased vocabulary that ships with AiDotNet (src/Tokenization/Resources/bert-base-uncased-vocab.txt.gz,
from google-research/bert).

Records the token ids, with [CLS] and [SEP], of texts that exercise each rule: Grounding DINO's period-separated
prompts, punctuation, accents, upper case, control and CJK characters, and words that WordPiece splits or cannot
match.
Default: verify; --write: regenerate tests/AiDotNet.Tests/Tokenization/ReferenceData/bert_tokenizer_reference.json.
"""
import argparse
import gzip
import json
import os
import sys
import tempfile

FIXTURE = "tests/AiDotNet.Tests/Tokenization/ReferenceData/bert_tokenizer_reference.json"
VOCAB = "src/Tokenization/Resources/bert-base-uncased-vocab.txt.gz"

TEXTS = [
    "a cat . a remote control .",
    "person. dog. traffic light.",
    "Héllo, WORLD! It's 3:15pm — naïve café?",
    "中文 mixed with English",
    "unbelievably supercalifragilisticexpialidocious tokenization",
    "tab\tseparated\nlines\u0007with control",
    "e-mail: someone@example.com (twice), [brackets] {braces} ^caret$ `tick`",
]


def compute():
    from transformers import BertTokenizer
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vocab.txt")
        with gzip.open(VOCAB, "rb") as f, open(path, "wb") as out:
            out.write(f.read())
        tokenizer = BertTokenizer.from_pretrained(d, do_lower_case=True)
        assert tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", "."]) == [101, 102, 1012], "the vocabulary did not load"
        return {"cases": [{"text": t, "ids": tokenizer(t)["input_ids"]} for t in TEXTS]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("wrote", len(result["cases"]))
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    print("fixture matches" if stored == result else "fixture differs")
    sys.exit(0 if stored == result else 1)


if __name__ == "__main__":
    main()
