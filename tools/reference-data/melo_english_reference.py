"""Reference fixture for MeloTTS's English front end: melo/text/english.py's text_normalize and g2p on a set of sentences.

Runs MeloTTS's own english.py (myshell-ai/MeloTTS at 2091453) with its pinned dependencies (inflect 7.0.0, g2p_en
2.1.0, transformers' bert-base-uncased tokenizer), the package's other language modules left out: english.py needs only
japanese.distribute_phone, which is loaded from japanese.py itself. Records, per sentence, the normalized text, the
phones, the tones and the phones per word piece, and the ids MeloTTS feeds its model after add_blank.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_english_reference.json.
"""
import ast
import importlib.util
import json
import os
import sys
import types

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
package.symbols = symbols.symbols   # the package's __init__ does `from .symbols import *`, so `symbols` is the list
japanese = types.ModuleType("melotext.japanese")
source = open(os.path.join(TEXT, "japanese.py"), encoding="utf-8").read()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "distribute_phone")
exec(compile(ast.Module(body=[node], type_ignores=[]), "japanese.py", "exec"), japanese.__dict__)
sys.modules["melotext.japanese"] = japanese
english = load("melotext.english", os.path.join(TEXT, "english.py"))

SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog!",
    "I refuse to collect the refuse around here.",
    "Please read the book you read yesterday.",
    "The wind will wind the clock; a lead pipe will lead the way.",
    "Dr. Smith lives on St. James St. with Mrs. Jones and Mr. Brown.",
    "It costs $5.50, or £3, or ¥100, or $1,234,567.89 in total.",
    "Meet me at 10:30 pm, 7:05 am or 23:15.",
    "In 1999, 2005, 2000, 1800 and 2017 the numbers changed: 3.14, -5, 42.",
    "She finished 1st, he was 22nd and they were 103rd.",
    "Don't worry, it's fine; we'll see what's next.",
    "A state-of-the-art, well-known co-worker.",
    "Zyxwvut flibbertigibbet quixotically snarfled the grobulator.",
    "Café naïve résumé façade.",
    "What? Really! Yes... maybe.",
    "Supercalifragilisticexpialidocious is long.",
    "Bob's 3 cats & 2 dogs (and a parrot) live at No. 9.",
    "The FBI and NASA met the CEO of IBM.",
    "Tokenization and preprocessing of transformers.",
    "Present the present to the presenter at present.",
    "Object to the object; a minute minute.",
    "We live in a live broadcast; close the close door.",
    "There are 1000000 stars and 1,001 nights.",
    "Call 911 or 555-1234.",
    "AI researchers' models' outputs.",
]


def main():
    out = []
    for sentence in SENTENCES:
        norm = english.text_normalize(sentence)
        phones, tones, word2ph = english.g2p(norm)
        out.append({"text": sentence, "normalized": norm, "phones": phones, "tones": tones, "word2ph": word2ph,
                    "pieces": english.tokenizer.tokenize(norm)})
    # g2p_en on its own: every homograph, and 200 letter strings drawn from a fixed generator (unknown to the
    # dictionary, so its GRU predicts them).
    import random
    g2p = english._g2p
    rng = random.Random(1234)
    words = sorted(g2p.homograph2features)
    unknown = []
    while len(unknown) < 200:
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 12)))
        if w not in g2p.cmu and w not in g2p.homograph2features and w.upper() not in english.eng_dict:
            unknown.append(w)
    g2pen = [{"word": w, "phones": g2p(w)} for w in words + unknown]
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "melo_english_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"melo_commit": "209145371cff8fc3bd60d7be902ea69cbdb7965a", "sentences": out, "g2p_en": g2pen}, f, ensure_ascii=False)
    print("wrote", os.path.normpath(path), len(out), "sentences")


if __name__ == "__main__":
    main()
