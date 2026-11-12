"""Reference fixture for MeloTTS's Spanish and French front ends: melo/text/spanish.py and french.py (text_normalize,
g2p, and es2ipa/fr2ipa on single words).

Runs MeloTTS's own modules (myshell-ai/MeloTTS at 2091453) with its pins (gruut[es,fr] 2.2.3, num2words 0.5.12,
transformers' dccuchile/bert-base-spanish-wwm-uncased and dbmdz/bert-base-french-europeana-cased tokenizers), the
package's other language modules left out. Two conventions the C# port follows (decided in PR #2294):

- LANG=C.UTF-8, the locale MeloTTS runs under on Linux; gruut's currency symbols come from Babel's default locale.
- gruut's processor is reset before every es2ipa/fr2ipa call, as in a fresh process: French liaison otherwise
  lengthens the lexicon cache's phoneme lists for the rest of the process.

Words dateparser would read relative to the current time (e.g. "12h") are rejected here: the port treats them as
words, and their reference reading changes daily.

Records per sentence the normalized text, the word pieces, the phones, tones and phones per piece; per word the
phonemizer output (or that MeloTTS raises).

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_romance_reference.json.
"""
import os

os.environ["LANG"] = "C.UTF-8"
for variable in ("LC_ALL", "LC_MONETARY", "LC_NUMERIC", "LANGUAGE"):
    os.environ.pop(variable, None)

import datetime  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import dateparser  # noqa: E402
import gruut  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc", "MeloTTS", "melo", "text"))

package = types.ModuleType("melotext")
package.__path__ = [TEXT]
sys.modules["melotext"] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symbols = load("melotext.symbols", os.path.join(TEXT, "symbols.py"))
package.symbols = symbols.symbols
for sub in ("es_phonemizer", "fr_phonemizer"):
    sub_package = types.ModuleType(f"melotext.{sub}")
    sub_package.__path__ = [os.path.join(TEXT, sub)]
    sys.modules[f"melotext.{sub}"] = sub_package
spanish = load("melotext.spanish", os.path.join(TEXT, "spanish.py"))
french = load("melotext.french", os.path.join(TEXT, "french.py"))

BASES = (datetime.datetime(2001, 2, 3, 4, 5, 6), datetime.datetime(2011, 7, 8, 9, 10, 11))


def guarded(module, name, lang):
    original = getattr(module, name)

    def phonemize(word):
        for token in word.split():
            if not any("0" <= c <= "9" for c in token):
                continue  # gruut asks dateparser only about words with a digit
            parsed = [dateparser.parse(token, settings={"STRICT_PARSING": True, "RELATIVE_BASE": b}, languages=[lang])
                      for b in BASES]
            if any(p is not None for p in parsed) and parsed[0] != parsed[1]:
                raise SystemExit(f"{token!r} is a date relative to now in {lang}; leave it out of the fixture")
        gruut._LOCAL.processors = {}
        return original(word)

    setattr(module, name, phonemize)
    return phonemize


es2ipa = guarded(spanish.es_to_ipa, "es2ipa", "es")
fr2ipa = guarded(french.fr_to_ipa, "fr2ipa", "fr")

SPANISH = [
    "Hola, ¿cómo estás? ¡Muy bien!",
    "En nuestros tiempos estos dos pueblos ilustres empiezan a curarse, gracias sólo a la sana y vigorosa higiene de 1789.",
    "El niño comió 21 manzanas y 1000000 de uvas en 2024.",
    "La señora Pérez llegó el 12mayo2020 con €5 y £3; costó 100 euros.",
    "¿Y a quién echaría de menos, en el mundo si no fuese a vos?",
    "Pingüino, cigüeña, acción, jamón y desvío.",
    "El perro de San Roque no tiene rabo porque Ramón Rodríguez se lo ha robado.",
    "Ella dijo: \"Ven aquí\" (rápido) — y se fue...",
    "Zxqv flibbertigibbet anticonstitucionalmente.",
    "Hay 3 gatos, 15 perros y 99 pájaros en el año 1999.",
    "Wi-Fi, e-mail y rock & roll.",
    "Él tenía 1e5 razones y ¥100.",
]

FRENCH = [
    "Bonjour, comment allez-vous ? Très bien, merci !",
    "Les enfants sont allés à l'école en 1789 avec 21 amis.",
    "M. Dupont et Mme Martin habitent au no. 81 de l'avenue.",
    "Il y a 121121 étoiles et 2000 planètes, etc.",
    "C'est un petit appartement, les grands arbres et des amis.",
    "Nous avons vu 1 chat, 80 chiens et 71 oiseaux le 12mai2020.",
    "Qu'est-ce que c'est ? « Un livre », dit-il.",
    "Ils ont payé €5, £3 et 100 dollars.",
    "Aujourd'hui, l'hôpital a reçu 1er et 2e prix.",
    "Zxqv grobulateur anticonstitutionnellement.",
    "Vingt et un, quatre-vingt-dix-neuf, cent un.",
    "Mlle Dubois & M. Leroy sont arrivés à midi.",
]

WORDS_ES = ["hola", "1789", "1", "21", "2000", "100000", "1e5", "2e3", "abc123", "zxqv", "¿", ",", "€5", "£3", "¥100",
            "$7", "₹12", "€nan1", "12mayo2020", "1ene2020", "2020mayo12", "29feb2019", "31abr2020", "veintiún", "méxico",
            "pingüino", "1000000", "1000001", "999999999", "123456789012", "0", "00", "007", "15", "1er", "x", "ll", "ñ",
            "wi-fi", "rock_and", "a’b", "«hola»", "(hola)", "¿hola", "hola!", "1999", "3000", "2999", "1001"]
WORDS_FR = ["bonjour", "1789", "1", "21", "71", "80", "81", "91", "99", "100", "101", "121121", "2000", "1e5", "€5",
            "£3", "$7", "¥100", "12mai2020", "1janv2020", "1er", "16e", "les", "des", "vingt", "Paris", "PARIS",
            "aujourd'hui", "l'école", "qu'est", "allez-vous", "quatre-vingt-dix", "M", "zxqv", "abc123", "«", "»",
            "(bonjour)", "¿bonjour", "1000000", "1000001", "200", "300", "1980", "11", "17", "très", "et", "un"]


def sample_lexicon(lang, step, seed):
    import gruut_lang_es
    import gruut_lang_fr
    directory = os.path.dirname((gruut_lang_es if lang == "es" else gruut_lang_fr).__file__)
    connection = sqlite3.connect(os.path.join(directory, "espeak", "lexicon.db"))
    words = [w for (w,) in connection.execute("SELECT DISTINCT word FROM word_phonemes ORDER BY word")]
    picked = words[::step]
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyzáéíóúñü" if lang == "es" else "abcdefghijklmnopqrstuvwxyzàâçéèêëîïôûùüÿœ"
    unknown = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 10))) for _ in range(150)]
    return picked, unknown


def phonemize_words(function, words):
    out = []
    for word in words:
        try:
            out.append({"word": word, "ipa": function(word)})
        except SystemExit:
            raise
        except Exception as error:  # MeloTTS raises: the port must raise too
            out.append({"word": word, "fails": type(error).__name__})
    return out


def sentences(module, texts):
    out = []
    for text in texts:
        norm = module.text_normalize(text)
        phones, tones, word2ph = module.g2p(norm)
        out.append({"text": text, "normalized": norm, "pieces": module.tokenizer.tokenize(norm), "phones": phones,
                    "tones": tones, "word2ph": word2ph})
    return out


def main():
    es_dictionary, es_unknown = sample_lexicon("es", 2000, 11)
    fr_dictionary, fr_unknown = sample_lexicon("fr", 300, 12)
    result = {
        "es": {"sentences": sentences(spanish, SPANISH), "words": phonemize_words(es2ipa, WORDS_ES),
               "dictionary": phonemize_words(es2ipa, es_dictionary), "unknown": phonemize_words(es2ipa, es_unknown)},
        "fr": {"sentences": sentences(french, FRENCH), "words": phonemize_words(fr2ipa, WORDS_FR),
               "dictionary": phonemize_words(fr2ipa, fr_dictionary), "unknown": phonemize_words(fr2ipa, fr_unknown)},
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "melo_romance_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    for lang in ("es", "fr"):
        counts = {k: len(v) for k, v in result[lang].items()}
        print(lang, counts)


if __name__ == "__main__":
    main()
