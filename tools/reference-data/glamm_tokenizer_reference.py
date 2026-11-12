"""GLaMM tokenizer reference: the sentencepiece library's BPE encoding, wrapped as the slow LLaMA tokenizer of the
transformers revision GLaMM pins (cae78c46) wraps it, run on a small BPE model trained here with LLaMA's trainer settings.

Vicuna's tokenizer.model is distributed under the Llama 2 license and is not shipped; this script trains a 400-piece BPE
model with LLaMA's settings (byte fallback, identity normalization, a dummy prefix, extra whitespace kept, digits split)
on a small corpus and stores it (base64) in the fixture, so both sides read the same model.

The wrapping is transcribed from that revision (tokenization_utils.py PreTrainedTokenizer.tokenize/_decode and
tokenization_llama.py LlamaTokenizer): the text is split on GLaMM's added tokens (added as strings, so whitespace on both
sides of one is stripped) and on <s>, </s>, <unk> (AddedTokens without stripping); each other part is encoded by
SentencePiece on its own; <s> is prepended. LLaVA's tokenizer_image_token splits on <image>. Decoding joins SentencePiece
runs (each with a leading space: the revision's _maybe_add_prefix_space compares tokens with a set of ids, so it always
adds one) and added tokens with spaces, then applies clean_up_tokenization.
Default: verify; --write: regenerate tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_tokenizer_reference.json.
"""
import argparse
import base64
import json
import os
import sys
import tempfile

FIXTURE = "tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_tokenizer_reference.json"
ADDED = ["<im_start>", "<im_end>", "<bbox>", "<point>", "[SEG]", "<p>", "</p>"]
CORPUS = [
    "A chat between a curious human and an artificial intelligence assistant.",
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    "Could you please give me a detailed description of the image?",
    "Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    "Please segment the person riding the horse in this image.",
    "Can you provide me with a detailed description of the region in the picture marked by region1?",
    "The image shows a dog sitting on a red couch next to a window, while a cat sleeps on the floor.",
    "There are 2 people, 3 bicycles and 12 birds in the park near the old stone bridge.",
    "Sure, it is the man in the blue shirt standing beside the car.",
    "A small boat floats on the calm lake under a cloudy sky at sunset.",
] * 20
CASES = [
    "A chat between a curious human and an artificial intelligence assistant. USER: <im_start><image><im_end>\n"
    "Please segment the dog in this image. ASSISTANT: Sure, it is [SEG].</s>",
    "Could you please give me a detailed description of the image? ASSISTANT: <p> A dog </p> [SEG] sits on <p> a couch </p> [SEG].",
    "  leading spaces and   repeated   spaces\tand tabs\nand newlines",
    "Emoji 🐶 and CJK 犬 fall back to bytes, digits 2024 split.",
    "region1 <bbox> and region2 <bbox> with a <point>",
    "<s>already has a bos</s><unk>",
]


def train(directory):
    import sentencepiece as spm
    corpus = os.path.join(directory, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        f.write("\n".join(CORPUS))
    prefix = os.path.join(directory, "glamm_test")
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, model_type="bpe", vocab_size=400, byte_fallback=True,
                                   normalization_rule_name="identity", add_dummy_prefix=True, remove_extra_whitespaces=False,
                                   split_digits=True, allow_whitespace_only_pieces=True, character_coverage=1.0,
                                   unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2)
    with open(prefix + ".model", "rb") as f:
        return f.read()


def wrap(model_bytes):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_proto=model_bytes)
    special = [sp.id_to_piece(sp.unk_id()), sp.id_to_piece(sp.bos_id()), sp.id_to_piece(sp.eos_id())]
    base = sp.get_piece_size()

    def split(text):  # leftmost, longest match over the added and special tokens (none is a prefix of another)
        parts, start, i = [], 0, 0
        tokens = ADDED + special
        while i < len(text):
            match = max((t for t in tokens if text.startswith(t, i)), key=len, default=None)
            if match is None:
                i += 1
                continue
            if i > start:
                parts.append(text[start:i])
            parts.append(match)
            i += len(match)
            start = i
        if start < len(text):
            parts.append(text[start:])
        return parts

    def tokenize(text):
        tokens = split(text)
        for i, token in enumerate(tokens):
            if token in ADDED:  # strings added with add_tokens: strip both neighbours
                if i + 1 < len(tokens):
                    tokens[i + 1] = tokens[i + 1].lstrip()
                if i > 0:
                    tokens[i - 1] = tokens[i - 1].rstrip()
        out = []
        for token in tokens:
            if not token:
                continue
            if token in ADDED:
                out.append(base + ADDED.index(token))
            elif token in special:
                out.append(sp.piece_to_id(token))
            else:
                out.extend(sp.encode(token))
        return out

    def encode(text):
        return [sp.bos_id()] + tokenize(text)

    def image_token(prompt):
        chunks = [encode(c) for c in prompt.split("<image>")]
        ids, offset = [], 0
        if chunks and chunks[0] and chunks[0][0] == sp.bos_id():
            offset = 1
            ids.append(chunks[0][0])
        separated = [e for pair in zip(chunks, [[-200] * (offset + 1)] * len(chunks)) for e in pair][:-1]
        for x in separated:
            ids.extend(x[offset:])
        return ids

    def convert_tokens_to_string(pieces):
        out, current, prev_special = "", [], False
        for p in pieces:
            if p in special:
                if not prev_special:
                    out += " "
                out += sp.decode_pieces(current) + p
                prev_special, current = True, []
            else:
                current.append(p)
                prev_special = False
        out += sp.decode_pieces(current)
        return " " + out

    def decode(ids):
        texts, run = [], []
        for i in ids:
            if i >= base:
                if run:
                    texts.append(convert_tokens_to_string([sp.id_to_piece(r) for r in run]))
                    run = []
                texts.append(ADDED[i - base])
            else:
                run.append(i)
        if run:
            texts.append(convert_tokens_to_string([sp.id_to_piece(r) for r in run]))
        text = " ".join(texts)
        for a, b in [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                     (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]:
            text = text.replace(a, b)
        return text

    return sp, encode, image_token, decode


def compute():
    with tempfile.TemporaryDirectory() as d:
        model = train(d)
    sp, encode, image_token, decode = wrap(model)
    cases = []
    for text in CASES:
        ids = encode(text)
        cases.append({"text": text, "pieces": sp.encode(text, out_type=str), "spm_ids": sp.encode(text), "ids": ids,
                      "image_ids": image_token(text), "decoded": decode([i for i in ids if i >= 0])})
    return {"model": base64.b64encode(model).decode("ascii"), "pieces": sp.get_piece_size(), "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("wrote", len(result["cases"]), "cases;", result["pieces"], "pieces")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    # The trainer is deterministic for a given sentencepiece version; compare the wrapped results on the stored model.
    sp, encode, image_token, decode = wrap(base64.b64decode(stored["model"]))
    same = all(encode(c["text"]) == c["ids"] and image_token(c["text"]) == c["image_ids"] for c in stored["cases"])
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
