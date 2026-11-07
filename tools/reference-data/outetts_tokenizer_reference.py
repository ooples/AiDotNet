"""OuteTTS 1.0's tokenizers: Hugging Face tokenizers' byte-level BPE, as the release encodes prompts
(tokenizer.encode(prompt, add_special_tokens=False)).

- OuteTTS-1.0-0.6B (Apache-2.0) is Qwen3's tokenizer: Qwen2's vocabulary and merges (embedded in the library already)
  with the release's added tokens. This exports those added tokens to
  src/TextToSpeech/Latest/Resources/outetts_qwen3_added_tokens.json.gz (content, id, special) after checking that the
  vocabulary, merges, normalizer and pre-tokenizer are Qwen2's, and records the ids of a corpus of prompts and texts.
  Pass the model's tokenizer.json with --qwen3.
- Llama-OuteTTS-1.0-1B's tokenizer (CC-BY-NC-SA, read from the user's model directory at run time) cannot be stored
  here, so a small tokenizer of the same form is trained with the tokenizers library (Llama 3's Split pattern, ByteLevel
  without its own regex, ignore_merges, the release's special tokens) and stored with the ids it gives the corpus.
  A GPT-2-form tokenizer (ByteLevel with its own regex) is added for coverage.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_tokenizer_reference.json. Default: verify; --write:
regenerate.
"""
import argparse
import gzip
import io
import json
import os
import sys

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/outetts_tokenizer_reference.json"
RESOURCE = "src/TextToSpeech/Latest/Resources/outetts_qwen3_added_tokens.json.gz"
LLAMA3_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
CORPUS = [
    "Hello, how are you doing today?",
    "<|im_start|>\n<|text_start|>The cat sat.<|text_end|>\n<|audio_start|>\n<|word_start|>The<|features|><|t_0.20|>"
    "<|energy_10|><|spectral_centroid_7|><|pitch_31|><|code|><|c1_12|><|c2_1023|><|c1_0|><|c2_7|><|word_end|>\n<|word_start|>",
    "Numbers 1234567 and 3.14159, prices $12.50 — dashes – and ellipses...",
    "I'm sure they've said: \"it's fine\", you'll SEE, we'd agree.",
    "日本語のテキストと中文文本，还有한국어。",
    "Émojis 😀🎉 and accents: café, naïve, Ångström, ﬁ ligature",
    "  leading spaces\n\nnewlines\r\n\ttabs   trailing  ",
    "<|im_end|><|im_start|>text<|unknown_token|>",
    "",
]


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def corpus(root):
    with open(os.path.join(root, "outetts/version/v3/default_speakers/json/en-female-1-neutral.json"), encoding="utf-8") as f:
        speaker = json.load(f)
    texts = list(CORPUS)
    words = speaker["words"]
    texts.append("\n".join(w["word"] for w in words))
    return texts


def qwen3(path, texts):
    from tokenizers import Tokenizer
    data = json.load(open(path, encoding="utf-8"))
    if data["normalizer"] != {"type": "NFC"}:
        raise SystemExit("unexpected normalizer")
    pattern = data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    if pattern != r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+":
        raise SystemExit("unexpected pre-tokenizer")
    added = [[a["content"], a["id"], a["special"]] for a in data["added_tokens"]]
    tokenizer = Tokenizer.from_file(path)
    return added, [tokenizer.encode(t, add_special_tokens=False).ids for t in texts]


def train(texts, llama):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, Regex
    tokenizer = Tokenizer(models.BPE(ignore_merges=llama))
    if llama:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(LLAMA3_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    special = ["<|im_start|>", "<|im_end|>", "<|text_start|>", "<|text_end|>", "<|audio_start|>", "<|audio_end|>",
               "<|word_start|>", "<|word_end|>", "<|features|>", "<|code|>", "<|t_0.20|>", "<|energy_10|>",
               "<|spectral_centroid_7|>", "<|pitch_31|>", "<|c1_12|>", "<|c2_1023|>", "<|c1_0|>", "<|c2_7|>", "<|c1_1|>"]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    training = [t for t in texts for _ in range(3)] + ["the quick brown fox jumps over the lazy dog " * 5,
                                                         "speech synthesis with a language model and a codec"]
    tokenizer.train_from_iterator(training, trainer)
    blob = tokenizer.to_str()
    return blob, [tokenizer.encode(t, add_special_tokens=False).ids for t in texts]


def compute(root, qwen3_path):
    texts = corpus(root)
    added, qwen3_ids = qwen3(qwen3_path, texts)
    llama_json, llama_ids = train(texts, True)
    gpt2_json, gpt2_ids = train(texts, False)
    return added, {"texts": texts, "qwen3_ids": qwen3_ids, "llama3_form": {"tokenizer": llama_json, "ids": llama_ids},
                   "gpt2_form": {"tokenizer": gpt2_json, "ids": gpt2_ids}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outetts", required=True, help="a checkout of edwko/OuteTTS")
    parser.add_argument("--qwen3", required=True, help="OuteAI/OuteTTS-1.0-0.6B's tokenizer.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    added, data = compute(os.path.abspath(args.outetts), args.qwen3)
    blob = gz(json.dumps(added, ensure_ascii=False))
    if args.write:
        with open(_fixture.fixture_path(RESOURCE), "wb") as handle:
            handle.write(blob)
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {RESOURCE} ({len(blob)} bytes) and {FIXTURE}")
        return
    stored = _fixture.load(FIXTURE)
    with open(_fixture.fixture_path(RESOURCE), "rb") as handle:
        same = handle.read() == blob
    same = same and stored["qwen3_ids"] == data["qwen3_ids"]
    # The trained tokenizers are stored; their ids are recomputed from the stored files.
    from tokenizers import Tokenizer
    for form in ("llama3_form", "gpt2_form"):
        tokenizer = Tokenizer.from_str(stored[form]["tokenizer"])
        same = same and [tokenizer.encode(t, add_special_tokens=False).ids for t in stored["texts"]] == stored[form]["ids"]
    print("OuteTTS tokenizers: match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
