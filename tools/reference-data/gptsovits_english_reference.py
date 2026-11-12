"""Reference fixture for GPT-SoVITS's English front end: text/english.py's text_normalize and g2p, and clean_text.

Runs GPT-SoVITS's own text package (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (g2p_en 2.1.0,
wordsegment 1.3.1, nltk 3.10.4, inflect 7.6.0), from the GPT-SoVITS checkout:

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_english_reference.py

Records, per sentence, the normalized text, the TweetTokenizer tokens and their tags, the phones and clean_text's
phones; expend.normalize on number cases; wordsegment on runs of letters; and the phones of a spread of words from
wordsegment's unigrams (dictionary words, names, letters read out, possessives, compounds and predictions).

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_english_reference.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
os.environ.setdefault("version", "v2")

from text import english, cleaner  # noqa: E402
from text.en_normalization.expend import normalize  # noqa: E402
import wordsegment  # noqa: E402
from nltk import pos_tag  # noqa: E402

SENTENCES = [
    "hello",
    "Hello world, this is a test.",
    "e.g. I used openai's AI tool to draw a picture.",
    "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.",
    "I read the book yesterday, and I will read it again tomorrow.",
    "The complex is a complex problem.",
    "Mr. Smith paid $6.24 for 3 apples and 32.3$ for 1.1£ of tea at 13:30.",
    "It's 4:00 now; the meeting ends at 13:00 and the bus leaves at 9:05.",
    "We ran 5km in 20min, then 1.2s faster, drank 1L and 0.1km more.",
    "The temperature was 4°F, 23°C, and -19°C at 1st, 2nd and 22nd place.",
    "3/23, 1/2, 3/2, 1/3, 6/1 of the 12,000 people agreed 100%.",
    "1. test ordinal number 1st.  2. second item.",
    "In 1999, 2000, 2005, 2010 and 1066 things happened; also 3000 and 999.",
    "x + y = z and 3 × 4 = 12, 10 - 3 = 7, 8 ÷ 2 = 4.",
    "Temperatures of -5 and -12 degrees.",
    "Café naïve résumé coöperate.",
    "The quick brown fox jumps over the lazy dog!",
    "What?! No way... really??? Yes!!!",
    "NASA and the FBI met at the UN with IBM and the CEO.",
    "I'm sure you'll love it; they've said we'd go, but I can't.",
    "Charles's dog and James's cat met the Joneses' kids at Bob's house.",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "thisisatest of wordsegment and somemorewordsjoined together.",
    "Visit www.example.com or end.Next sentence starts here.",
    "Ozymandias, Xenophon and Quetzalcoatl visited Llanfairpwllgwyngyll.",
    "AE AI AR IOS HUD OS are abbreviations.",
    "Use the iPhone, the McDonald's app and the YouTube channel.",
    "A is the first letter, and a is an article. I am here.",
    "state-of-the-art, well-known, mother-in-law, twenty-one.",
    "He said \"hello\" and she said ’goodbye’。Really！Why？",
    "hmm... ok -- fine - sure",
    "The 3rd, 11th, 12th, 13th, 21st, 101st and 111th entries.",
    "£7.14 and $0.99 and $1 and £1.01 and $100.",
    "1,234,567 is a big number; 0.5 is half; 3.14159 is pi.",
    "The   spaces    between    words.",
    "i.e. this is that, E.G. for example.",
    "tbsp tsp ft h 5tbsp 2tsp 6ft 3h",
]

NUMBER_CASES = [
    "1. ", "21. x", "3 + 4", "-5", " -5", "a -7 b", "12,345", "9:30", "23:59", "0:00", "12:00", "5m", "1m", "1.0m", "2km/h",
    "10ft", "1L", "3tbsp", "1tsp", "2h", "1min", "45s", "30°C", "1°F", "£5", "£5.5", "5£", "£0.01", "$1.2.3", "$12", "12$",
    "(12$", "3.14", "3. 14", "1/4", "2/2", "5/1", "7/8", "1st", "2nd", "3rd", "4th", "11th", "21st", "1001", "1900", "2000",
    "2001", "2010", "2999", "3000", "1000", "100", "0", "007", "50%", "x²", "AbcDef", "i.e.", "e.g.", "A",
]


def main():
    sentences = []
    for text in SENTENCES:
        norm = english.text_normalize(text)
        tokens = english.word_tokenize(norm)
        entry = {"text": text, "normalized": norm, "tokens": tokens, "tags": [t for _, t in pos_tag(tokens)],
                 "phones": english.g2p(norm), "clean": cleaner.clean_text(text, "en", "v2")[0]}
        sentences.append(entry)
    numbers = []
    for t in NUMBER_CASES:
        try:
            numbers.append({"text": t, "normalized": normalize(t)})
        except ValueError:
            # "(12$": _dollars_re_end's class admits "(", and int() then rejects it.
            numbers.append({"text": t, "fails": True})
    segments = [{"text": t, "words": wordsegment.segment(t)} for t in [
        "thisisatest", "wheninthecourseofhumaneventsitbecomesnecessary", "choosespain", "itwasabrightcolddayinapril",
        "a" * 300, "the" * 120, "superman", "abc123def",
    ]]
    unigrams = list(wordsegment.UNIGRAMS)
    words = []
    for word in unigrams[3::997] + ["Smith's", "Charles's", "Xyzzyqx", "Mcgonagall", "Zxqw", "aaa", "bbb", "a-b", "it's"]:
        words.append({"word": word, "phones": english.g2p(word)})
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "gptsovits_english_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"sentences": sentences, "numbers": numbers, "segments": segments, "words": words}, handle,
                  ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(sentences)} sentences, {len(numbers)} number cases, {len(segments)} segments, {len(words)} words")


if __name__ == "__main__":
    main()
