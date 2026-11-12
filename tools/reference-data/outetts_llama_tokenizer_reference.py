"""Reproduce OuteTTSTokenizerTests' actual Llama release token ids.

Pass OuteAI/Llama-OuteTTS-1.0-1B/tokenizer.json, revision 6cb62a5.
Requires Hugging Face tokenizers. The model tokenizer itself is not redistributed.
The native implementation reuses the existing Llama 3 resources and generates
the release's contiguous audio/prompt token mapping.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json", type=Path)
    args = parser.parse_args()
    raw = args.tokenizer_json.read_bytes()
    expected_sha = "ec54a55e3c6fc7318ea02cbd1a6eb1fb180bff1b58acbf068504a25f1b407b7b"
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SystemExit("The tokenizer is not the pinned release.")
    data = json.loads(raw)
    resources = Path(__file__).resolve().parents[2] / "src/Tokenization/Resources"
    vocabulary = json.loads(gzip.decompress((resources / "llama3-vocab.json.gz").read_bytes()))
    merges = gzip.decompress((resources / "llama3-merges.txt.gz").read_bytes()).decode().splitlines()
    specials = json.loads(gzip.decompress((resources / "llama3-special-tokens.json.gz").read_bytes()))
    if vocabulary != data["model"]["vocab"]:
        raise SystemExit("The embedded Llama vocabulary differs from the release.")
    release_merges = [" ".join(row) if isinstance(row, list) else row for row in data["model"]["merges"]]
    if merges != release_merges:
        raise SystemExit("The embedded Llama merges differ from the release.")
    if specials != [row["content"] for row in data["added_tokens"] if row["id"] < 128256]:
        raise SystemExit("The embedded Llama special tokens differ from the release.")
    texts = [
        "Hello, world!",
        "cafe\u0301 / caf\u00e9 / \u4f60\u597d / \U0001f642",
        "1234567890",
        "<|im_start|>user\n<|text_start|>Hello<|text_end|><|audio_start|>",
        "<|c1_0|><|c1_1024|><|c2_0|><|c2_1024|><|t_0.00|><|t_10.00|>",
        "<|energy_0|><|energy_1000|><|spectral_centroid_1000|><|pitch_1000|>",
    ]
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    cases = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise SystemExit("The release tokenizer did not round-trip the reference text.")
        cases.append({"text": text, "ids": ids, "decoded": decoded})
    print(json.dumps(cases, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
