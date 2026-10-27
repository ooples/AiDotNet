"""Reference fixture for MaskGCT's text front end (Amphion models/tts/maskgct/g2p at 26f6883) for the languages it reads
through eSpeak NG: English (including the Chinese-English segmenter's English segments), French, German and Korean.

Runs inside the release's own environment (Amphion's Dockerfile: Ubuntu 18.04 with apt's espeak-ng 1.49.2) with each G2P
package at its latest release before MaskGCT's (phonemizer 3.3.0, inflect 7.4.0, Unidecode 1.3.8); see
refsrc/maskgct-docker/Dockerfile. Build and run:

    docker build -t maskgct-g2p refsrc/maskgct-docker
    docker run --rm -v <Amphion>:/amphion -v <repo>:/repo -w /amphion maskgct-g2p \
        python /repo/tools/reference-data/maskgct_g2p_reference.py

For each text it records MaskGCT's g2p_ result (the phone string and the vocabulary ids), and every call phonemizer makes
to espeak_TextToPhonemes (the voice, the text chunk and eSpeak's output), so the port's phonemizer layer can be checked
without a native library and its native wrapper against the same calls where espeak-ng 1.49.2 is installed.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_g2p_reference.json.
"""
import json
import os
import sys
import types

sys.path.insert(0, "/amphion")
# LangSegment's only PyPI release (0.2.0, a 2025 re-upload) fails to import. MaskGCT calls LangSegment.setfilters at import
# and LangSegment.getTexts only for language "auto", which its pipeline never passes, so a module with the first is enough.
sys.modules["LangSegment"] = types.SimpleNamespace(setfilters=lambda filters: None)
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

calls = []
_text_to_phonemes = EspeakWrapper.text_to_phonemes


def recording(self, text, tie=False):
    out = _text_to_phonemes(self, text, tie)
    calls.append({"voice": self.voice.language, "text": text, "phonemes": out})
    return out


EspeakWrapper.text_to_phonemes = recording

from models.tts.maskgct.g2p.g2p_generation import chn_eng_g2p, g2p  # noqa: E402


def g2p_(text, language):
    # maskgct_utils.g2p_, verbatim (importing maskgct_utils would load the whole model stack).
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)

CASES = [
    ("en", "Hello, world! It's 3:30 pm on Jan. 5th, 2024 - Mr. Smith paid $12.50 for 2 books."),
    ("en", "We do not break. We never give in. We never back down."),
    ("en", "In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model; 50% faster? Yes!"),
    ("en", "The quick brown fox jumps over the lazy dog's 1st, 22nd and 103rd tails... (really)"),
    ("en", "Dr. Jones, Ltd. etc. trained 1/2 of the 3.14 models at St. Louis for $3."),
    ("fr", "Bonjour, M. Dupont! Comment allez-vous aujourd'hui ? J'ai 3 chats; c'est-à-dire beaucoup."),
    ("fr", "Mme Martin habite au 12, boul. Saint-Michel, etc."),
    ("de", "Guten Tag! Wie geht es Ihnen heute? Ich habe 3 Katzen; das ist schön."),
    ("de", "Straße, Mädchen und Fräulein: „Zürich“ liegt in der Schweiz."),
    ("ko", "안녕하세요! 오늘 날씨가 정말 좋네요. AI와 CEO는 IT 회사에서 일해요."),
    ("ko", "한국어 텍스트, 숫자 123과 漢字 섞임."),
]


def main():
    cases = []
    for language, text in CASES:
        start = len(calls)
        phonemes, tokens = g2p_(text, language)
        cases.append({"language": language, "text": text, "phonemes": phonemes, "tokens": tokens, "calls": calls[start:]})
    result = {"espeak": os.popen("espeak-ng --version").read().strip(), "cases": cases}
    path = "/repo/tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_g2p_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False)
        handle.write("\n")
    for c in cases:
        print(c["language"], len(c["tokens"]), c["phonemes"][:80])


if __name__ == "__main__":
    main()
