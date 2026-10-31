"""NATSpeech's English text front end through its own code (NATSpeech/NATSpeech, data_gen/tts/txt_processors/en.py:
preprocess_text, EnG2p over g2p_en 2.1.0 with NLTK's tweet tokenizer and tagger, process and the base postprocess with
word separators and <BOS>/<EOS>; utils/text/text_encoder.py's TokenTextEncoder). Records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/natspeech_frontend_reference.json, for English texts (numbers, currency,
ordinals, contractions, hyphens, quotes, accents, homographs, out-of-vocabulary words, repeated punctuation, "i.e.",
"etc."), the cleaned text, the phones and their ids over the processor's full phone set.
Default: verify; --write: regenerate. Pass --natspeech (checkout). Needs g2p_en and nltk (the .venv-melo312 environment).
"""
import argparse
import os
import sys

try:
    import torch  # noqa: F401  (_fixture records its version)
except ImportError:
    import types
    sys.modules["torch"] = types.SimpleNamespace(__version__="not installed")
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/natspeech_frontend_reference.json"
TEXTS = [
    "Hello world, this is Mega-TTS.",
    "The quick brown fox jumps over the lazy dog!",
    "I read the book yesterday; today I will read another.",
    "Lead the way, but don't touch the lead pipe.",
    "It costs $19.99, or about £15 — roughly 20% off.",
    "On March 3rd, 1999, 42 people met at 10:30.",
    "\"Quoted\" words (and brackets) shouldn't matter...",
    "Naïve café owners!!! Really??",
    "That is, i.e. the end, etc.",
    "Supercalifragilisticexpialidocious and zxqwerty are odd words.",
    "We'll, they've, it's, can't, won't.",
    "The 2nd and 23rd entries were live; the live show was great.",
    "   leading and trailing spaces   ",
    "Mr. Smith-Jones lives at 221B Baker Street.",
    "",
    "!!!",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natspeech", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.natspeech))
    from data_gen.tts.txt_processors.en import TxtProcessor
    from utils.text.text_encoder import TokenTextEncoder
    from g2p_en import G2p
    phones = sorted(set([p for p in G2p().phonemes if p not in ("<pad>", "<unk>", "<s>", "</s>")] + list("!,.?;:") + ["|", "<BOS>", "<EOS>"]))
    encoder = TokenTextEncoder(None, vocab_list=phones, replace_oov="<UNK>")
    stored = None if args.write else _fixture.load(FIXTURE)
    cases = []
    for text in TEXTS:
        struct, clean = TxtProcessor.process(text, {"with_phsep": True, "add_eos_bos": True})
        ph = [p for w in struct for p in w[1]]
        cases.append({"text": text, "clean": clean, "words": [w[0] for w in struct], "phones": ph, "ids": encoder.encode(" ".join(ph))})
    data = {"vocabulary": [encoder.id_to_token[i] for i in range(len(encoder))], "cases": cases}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        for c in cases:
            print(repr(c["text"][:40]), " ".join(c["phones"])[:120])
        return
    check = _fixture.Comparison("NATSpeech front end")
    check.exact("vocabulary", stored["vocabulary"], data["vocabulary"])
    check.exact("cases", stored["cases"], data["cases"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
