"""Builds the MeloTTS Spanish and French front ends' embedded resources from gruut 2.2.3 (MIT), the phonemizer MeloTTS's
es_phonemizer/fr_phonemizer call with espeak=True.

Run in an environment with MeloTTS's pins (gruut[es,fr]==2.2.3, Babel 2.x) and LANG=C.UTF-8, the locale MeloTTS runs
under on Linux (gruut reads its currency symbols from Babel's default locale, en_US_POSIX there):

- gruut_{es,fr}_lexicon.tsv.gz: gruut_lang_{es,fr}/espeak/lexicon.db's word_phonemes table, "word<TAB>role<TAB>phonemes"
  ordered by word and pron_order, as gruut's SqlitePhonemizer reads it.
- gruut_es_g2p.json.gz, gruut_fr_g2p.json.gz, gruut_fr_pos.json.gz: the CRFsuite models (espeak/g2p/model.crf and
  pos/model.crf) as labels in id order, each attribute's state features in the model's attribute-reference order, and
  the dense label-to-label transition matrix. Weights are read from the model file's feature chunk (exact doubles; the
  dump pycrfsuite prints rounds them to six decimals); the label and attribute strings come from pycrfsuite.
- gruut_locales.json.gz: per locale (es_ES, fr_FR), Babel's decimal and group symbols, the stand-alone month names
  gruut verbalizes dates with (format_date "MMMM"), and gruut's currency symbols in the order it tries them with the
  currency each names.
- num2words_currency.json.gz: num2words 0.5.12's (LGPL-2.1) Spanish and French currency forms (CURRENCY_FORMS) and the
  Spanish feminine-currency lists (CURRENCIES_UNA, CENTS_UNA), which gruut's currency verbalization reads.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import datetime
import gzip
import io
import json
import operator
import os
import sqlite3
import struct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def lexicon(db_path):
    connection = sqlite3.connect(db_path)
    lines = []
    for word, role, phonemes in connection.execute(
            "SELECT word, role, phonemes FROM word_phonemes ORDER BY word, pron_order, id"):
        if any("\t" in s or "\n" in s for s in (word, role, phonemes)):
            raise ValueError(f"a separator in the entry for {word!r}")
        lines.append(f"{word}\t{role}\t{phonemes}\n")
    return gz("".join(lines)), len(lines)


def crf(model_path):
    import pycrfsuite
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    info = tagger.info()
    labels = [None] * len(info.labels)
    for label, index in info.labels.items():
        labels[int(index)] = label
    attributes = [None] * len(info.attributes)
    for attribute, index in info.attributes.items():
        attributes[int(index)] = attribute

    data = open(model_path, "rb").read()
    header = struct.unpack("<4sI4sIIIIIIIII", data[:48])
    off_features, off_attrrefs = header[7], header[11]
    if data[off_features:off_features + 4] != b"FEAT" or data[off_attrrefs:off_attrrefs + 4] != b"AFRF":
        raise ValueError("unexpected CRFsuite model layout")
    num_features = struct.unpack("<I", data[off_features + 8:off_features + 12])[0]
    features = [struct.unpack("<IIId", data[off_features + 12 + 20 * k:off_features + 32 + 20 * k])
                for k in range(num_features)]

    transitions = [[0.0] * len(labels) for _ in labels]
    for kind, source, target, weight in features:
        if kind == 1:
            transitions[source][target] = weight

    num_attrs = struct.unpack("<I", data[off_attrrefs + 8:off_attrrefs + 12])[0]
    state = {}
    for a in range(num_attrs):
        offset = struct.unpack("<I", data[off_attrrefs + 12 + 4 * a:off_attrrefs + 16 + 4 * a])[0]
        count = struct.unpack("<I", data[offset:offset + 4])[0]
        refs = struct.unpack(f"<{count}I", data[offset + 4:offset + 4 + 4 * count])
        entries = []
        for fid in refs:
            kind, source, target, weight = features[fid]
            if kind != 0 or source != a:
                raise ValueError("attribute reference to a feature of another attribute")
            entries.append([target, weight])
        state[attributes[a]] = entries

    # Cross-check against pycrfsuite's (rounded) dump.
    for (attribute, label), weight in info.state_features.items():
        match = [w for t, w in state[attribute] if labels[t] == label]
        if len(match) != 1 or abs(match[0] - weight) > 1e-6:
            raise ValueError(f"state feature {attribute}/{label} does not match the dump")
    model = {"labels": labels, "attributes": state, "transitions": transitions}
    return gz(json.dumps(model, ensure_ascii=False)), len(labels), len(state)


def locales():
    import babel
    import babel.dates
    import babel.numbers
    if babel.default_locale("LC_MONETARY") != "en_US_POSIX":
        raise SystemExit("run with LANG=C.UTF-8 (Babel's default locale must be en_US_POSIX, as on Linux)")
    result = {}
    for locale in ("es_ES", "fr_FR"):
        # TextProcessorSettings.__post_init__: {symbol in the default locale: code} over the locale's currencies,
        # then sorted by length, longest first (a stable sort).
        currencies = {babel.numbers.get_currency_symbol(cn): cn for cn in babel.Locale(locale).currency_symbols}
        symbols = sorted(currencies, key=operator.length_hint, reverse=True)
        result[locale] = {
            "decimal": babel.numbers.get_decimal_symbol(locale),
            "group": babel.numbers.get_group_symbol(locale),
            "months": [babel.dates.format_date(datetime.date(2001, m, 1), "MMMM", locale=locale) for m in range(1, 13)],
            "currencies": [[s, currencies[s]] for s in symbols],
        }
    return gz(json.dumps(result, ensure_ascii=False, sort_keys=True))


def currency_forms():
    from num2words import lang_ES, lang_FR
    result = {
        "es": {code: [list(forms[0]), list(forms[1])] for code, forms in lang_ES.Num2Word_ES.CURRENCY_FORMS.items()},
        "fr": {code: [list(forms[0]), list(forms[1])] for code, forms in lang_FR.Num2Word_FR.CURRENCY_FORMS.items()},
        "es_currencies_una": list(lang_ES.CURRENCIES_UNA),
        "es_cents_una": list(lang_ES.CENTS_UNA),
    }
    return gz(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    import gruut_lang_es
    import gruut_lang_fr
    es_dir = os.path.dirname(gruut_lang_es.__file__)
    fr_dir = os.path.dirname(gruut_lang_fr.__file__)
    outputs = {}
    outputs["gruut_es_lexicon.tsv.gz"], n_es = lexicon(os.path.join(es_dir, "espeak", "lexicon.db"))
    outputs["gruut_fr_lexicon.tsv.gz"], n_fr = lexicon(os.path.join(fr_dir, "espeak", "lexicon.db"))
    outputs["gruut_es_g2p.json.gz"], *_ = crf(os.path.join(es_dir, "espeak", "g2p", "model.crf"))
    outputs["gruut_fr_g2p.json.gz"], *_ = crf(os.path.join(fr_dir, "espeak", "g2p", "model.crf"))
    outputs["gruut_fr_pos.json.gz"], *_ = crf(os.path.join(fr_dir, "pos", "model.crf"))
    outputs["gruut_locales.json.gz"] = locales()
    outputs["num2words_currency.json.gz"] = currency_forms()
    if args.write:
        for name, data in outputs.items():
            with open(os.path.join(OUT, name), "wb") as handle:
                handle.write(data)
        print(f"wrote {len(outputs)} resources: {n_es} Spanish and {n_fr} French lexicon entries")
        return
    failures = [name for name, data in outputs.items() if open(os.path.join(OUT, name), "rb").read() != data]
    if failures:
        raise SystemExit("differ: " + ", ".join(failures))
    print("all resources match")


if __name__ == "__main__":
    main()
