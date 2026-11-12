"""T5 tokenizer reference: google-t5/t5-base's SentencePiece model (sentencepiece) on a test corpus.

This is what audiocraft's T5Conditioner feeds MusicGen's text encoder: transformers 4.x's slow T5Tokenizer, which
splits on the special strings, segments each piece of text with sentencepiece and appends </s>. That is reproduced
here with sentencepiece itself, because transformers 5 replaced the slow tokenizer with the Rust `tokenizers`
backend, whose normalizer differs (it drops a combining mark after a no-break space, where sentencepiece emits
<unk>). The corpus has:
- hand-written music descriptions and edge cases (whitespace runs, tabs and newlines, leading and trailing spaces,
  full-width and accented letters, characters the vocabulary lacks, special-token strings);
- 2000 seeded strings mixing ASCII, Latin-1, Greek, Cyrillic, CJK, full-width forms, combining marks, emoji and
  whitespace, which exercise the NFKC character map, unknown-run merging and the Viterbi segmentation.
It records the ids (sentencepiece ids plus the appended </s>) in
tests/AiDotNet.Tests/Tokenization/ReferenceData/t5_tokenizer_reference.json.
Needs sentencepiece and huggingface_hub. Default: verify; --write: regenerate.
"""
import argparse
import json
import os
import random
import re
import sys

FIXTURE = "tests/AiDotNet.Tests/Tokenization/ReferenceData/t5_tokenizer_reference.json"

HAND_WRITTEN = [
    "An 80s driving pop song with heavy drums and synth pads in the background",
    "lofi slow bpm electro chill with organic samples",
    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
    "Pop dance track with catchy melodies, tropical percussions, and upbeat rhythms, perfect for the beach",
    "classic reggae track with an electronic guitar solo",
    "genre: rock. moods: energetic, dark. key: E minor. bpm: 140",
    "", " ", "   leading and trailing   ", "tabs\tand\nnewlines\r\nhere", "multiple     spaces",
    "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "café naïve résumé Ångström", "日本語のテキスト", "emoji 🎵🎶🎸 song",
    "ﬁ ligature ½ fraction ① circled", "Ⅻ roman ㎏ unit", "é combining acute", "non breaking spaces",
    "What's up? It's 3:45pm -- let's go!", "a</s>b", "<extra_id_0> masked <extra_id_99>", "<pad><unk>",
]

ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    ".,;:!?'\"()[]{}-_/\\@#$%^&*+=<>|~`",
    "     \t\n  ",
    "àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿßÀÉÖÜ",
    "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔ",
    "абвгдежзийклмнопрстуфхцчшщыэюяАБВГ",
    "音楽的歌曲节奏鼓吉他钢琴和弦旋律",
    "ＡＢＣａｂｃ０１２！？",
    "̧̀́̂̈",
    "🎵🎶🎸🥁🎹🎻😀🔥",
    "ﬁﬂ½¼①②Ⅻ㎏™©®",
]


def corpus():
    rng = random.Random(20261011)
    texts = list(HAND_WRITTEN)
    for _ in range(2000):
        n = rng.randint(1, 40)
        chars = []
        for _ in range(n):
            alphabet = ALPHABETS[rng.choices(range(len(ALPHABETS)), weights=[30, 6, 6, 12, 4, 2, 2, 2, 2, 1, 2, 1])[0]]
            chars.append(rng.choice(alphabet))
        texts.append("".join(chars))
    return texts


SPECIAL = re.compile(r"</s>|<pad>|<unk>|<extra_id_(\d{1,2})>")


def compute():
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download
    sp = spm.SentencePieceProcessor(model_file=hf_hub_download("google-t5/t5-base", "spiece.model"))
    special = {"</s>": 1, "<pad>": 0, "<unk>": sp.unk_id()}

    def encode(text):
        ids, position = [], 0
        for match in SPECIAL.finditer(text):
            ids += sp.encode(text[position:match.start()])
            ids.append(special.get(match.group(0), 32099 - int(match.group(1) or 0)))
            position = match.end()
        ids += sp.encode(text[position:])
        return ids if ids and ids[-1] == 1 else ids + [1]

    return [{"text": t, "ids": encode(t)} for t in corpus()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cases = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": "google-t5/t5-base spiece.model (sentencepiece), </s> appended", "cases": cases}, f, ensure_ascii=False)
        print(f"wrote {len(cases)} cases")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)["cases"]
    bad = [c for c, s in zip(cases, stored) if c != s]
    print(f"{len(cases) - len(bad)}/{len(cases)} match")
    sys.exit(1 if bad or len(cases) != len(stored) else 0)


if __name__ == "__main__":
    main()
