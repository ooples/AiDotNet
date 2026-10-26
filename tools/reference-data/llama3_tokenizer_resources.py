"""Builds the embedded Llama 3 tokenizer (src/Tokenization/Resources/llama3-*.gz) from a Llama 3.2 tokenizer.json, and its
reference fixture.

Llama 3, 3.1 and 3.2 share one byte-level BPE (128000 tokens, 280147 merges, ignore_merges) and 256 special tokens
(128000-128255). Orpheus's tokenizer is that one plus its added tokens (<custom_token_0..28682> and <|audio|>); this
script checks the two agree before writing anything.

Usage: python llama3_tokenizer_resources.py LLAMA_3_2_TOKENIZER_JSON ORPHEUS_TOKENIZER_JSON
(the ungated copies unsloth/Llama-3.2-3B-Instruct and unsloth/orpheus-3b-0.1-ft serve). Needs the tokenizers package.
Writes the three resources and tests/AiDotNet.Tests/TextToSpeech/ReferenceData/llama3_tokenizer_reference.json, the
ids Orpheus's tokenizer gives a set of strings (encode with add_special_tokens, so the <|begin_of_text|> prefix too).
"""
import gzip
import json
import os
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
RESOURCES = os.path.join(REPO, "src", "Tokenization", "Resources")

TEXTS = [
    "tara: Hello there, how are you doing today?",
    "leo: Man, the way social media has, um, completely changed how we interact is just wild, right? <laugh>",
    "Numbers 1234567 and 3.14159, dates 2025-03-18, and emails a.b@example.com.",
    "  leading spaces\tand tabs\nnew lines\r\n\r\nand trailing   ",
    "Ünïcödé façade, naïve café — “quotes” and ‘apostrophes’; I'm, you're, it's, WE'LL, they'D.",
    "日本語のテキストと中文文本, 한국어 텍스트, emoji 🎉👍🏽 and 𝔘𝔫𝔦𝔠𝔬𝔡𝔢.",
    "<custom_token_5>raw custom tokens<custom_token_28682> and <|eot_id|> specials inline",
    "zac: I went to the store <sigh> and they were out of everything <chuckle>.",
]


def write_gz(name, text):
    with open(os.path.join(RESOURCES, name), "wb") as handle:
        handle.write(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def main():
    llama_path, orpheus_path = sys.argv[1], sys.argv[2]
    with open(llama_path, encoding="utf-8") as handle:
        llama = json.load(handle)
    with open(orpheus_path, encoding="utf-8") as handle:
        orpheus = json.load(handle)
    model = llama["model"]
    assert model["type"] == "BPE" and model["ignore_merges"] and llama["normalizer"] is None
    assert model["vocab"] == orpheus["model"]["vocab"] and model["merges"] == orpheus["model"]["merges"]
    assert llama["pre_tokenizer"] == orpheus["pre_tokenizer"]
    specials = [token["content"] for token in llama["added_tokens"]]
    assert [token["id"] for token in llama["added_tokens"]] == list(range(128000, 128256))
    assert specials == [token["content"] for token in orpheus["added_tokens"][:256]]
    extra = orpheus["added_tokens"][256:]
    assert [t["content"] for t in extra] == [f"<custom_token_{i}>" for i in range(28683)] + ["<|audio|>"]
    assert [t["id"] for t in extra] == list(range(128256, 156940))
    assert all(not t["special"] for t in extra[:-1]) and extra[-1]["special"]

    vocab = sorted(model["vocab"].items(), key=lambda pair: pair[1])
    assert [i for _, i in vocab] == list(range(128000))
    assert all(" " not in left and " " not in right for left, right in model["merges"])
    write_gz("llama3-vocab.json.gz", json.dumps(dict(vocab), ensure_ascii=False))
    write_gz("llama3-merges.txt.gz", "\n".join(f"{left} {right}" for left, right in model["merges"]))
    write_gz("llama3-special-tokens.json.gz", json.dumps(specials, ensure_ascii=False))

    tokenizer = Tokenizer.from_file(orpheus_path)
    cases = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=True).ids} for text in TEXTS]
    out = os.path.join(REPO, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "llama3_tokenizer_reference.json")
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"cases": cases}, handle, ensure_ascii=False)
        handle.write("\n")
    print("vocab", len(vocab), "merges", len(model["merges"]), "specials", len(specials), "cases", len(cases))


if __name__ == "__main__":
    main()
