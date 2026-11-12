"""The Russian stress front end the multilingual Chatterbox calls (russian_text_stresser 0.6.0 from
Vuizur/add-stress-to-epub 9e064c8: RussianTextStresser().stress_text over its bundled ru_core_news_sm-3.6.0 spaCy pipeline,
russian_dict.db and simple_cases.pkl from its v1.0.1 release), under spaCy 3.6.1 / thinc 8.1.12. Records, per sentence:
spaCy's tokens (text, trailing whitespace, POS, morphology), the six hashed lexical features tok2vec reads, the first
sentence's tok2vec output, and stress_text of the sentence and of the lower-cased NFKD form Chatterbox passes.
Usage: python tools/reference-data/russian_text_stresser_reference.py --dict-dir <dir with russian_dict.db, simple_cases.pkl>
       [--write]   (in .venv-rustress: python 3.11, spacy 3.6.1, numpy<2, the stresser installed --no-deps)
"""
import argparse
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/russian_text_stresser_reference.json"
SENTENCES = [
    "Твои слова ничего не значат.",
    "Мама мыла раму, а папа читал газету в кресле.",
    "Кто-то постучал в дверь; это был почтальон!",
    "Ёжик пошёл в лес и нашёл там большой гриб.",
    "В 2024 году мы купили 3 новых стула за 1500 рублей.",
    "«Привет», — сказал он, т.е. просто поздоровался.",
    "Москва — столица России, а Санкт-Петербург — культурная столица.",
    "Я не знаю, что делать: идти или остаться?",
    "Учёные изучают замок на берегу реки, а мы закрыли дверь на замок.",
    "Посмотрите на сайт www.example.ru или напишите на test@example.com",
    "Его друзья приехали из-за границы вчера утром.",
    "Это  двойной   пробел\tи табуляция.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dict-dir", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from russian_text_stresser.text_stresser import RussianTextStresser
    stresser = RussianTextStresser(db_file=os.path.join(args.dict_dir, "russian_dict.db"),
                                   simple_cases_file=os.path.join(args.dict_dir, "simple_cases.pkl"))
    nlp = stresser._nlp
    from spacy.attrs import NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE
    rows = []
    for i, sentence in enumerate(SENTENCES):
        chatterbox = unicodedata.normalize("NFKD", sentence.lower())
        row = {"text": sentence, "chatterbox_text": chatterbox, "stressed": stresser.stress_text(sentence),
               "chatterbox_stressed": stresser.stress_text(chatterbox)}
        for key, text in (("tokens", sentence), ("chatterbox_tokens", chatterbox)):
            doc = nlp(text)
            row[key] = [{"text": t.text, "ws": t.whitespace_, "pos": t.pos_, "morph": str(t.morph)} for t in doc]
            row[key + "_features"] = [[int(v) for v in r] for r in doc.to_array([NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE]).tolist()]
            if i == 0 and key == "tokens":
                row["tok2vec"] = [[float(x) for x in r] for r in doc.tensor.tolist()]
        rows.append(row)
    data = {"sentences": rows}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Russian stresser")
    for a, b in zip(stored["sentences"], data["sentences"]):
        for key in ("stressed", "chatterbox_stressed", "tokens", "chatterbox_tokens", "tokens_features", "chatterbox_tokens_features"):
            check.exact(key, a[key], b[key])
    check.report()


if __name__ == "__main__":
    main()
