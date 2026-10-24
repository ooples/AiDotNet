"""ftfy 6.3.1 (rspeer/python-ftfy, Apache-2.0): its data tables, and fix_text on a test corpus.

OuteTTS 1.0's prompt builder runs text through ftfy.fix_text. The C# port (Ftfy.cs) reads the tables ftfy builds at
import time, exported here so both sides use the same data:
- the charmap codecs fix_encoding tries (latin-1, the sloppy Windows code pages, iso-8859-2, macroman, cp437): each as
  the 256 characters its bytes decode to;
- the HTML entities ftfy unescapes (html5 entities with a semicolon, plus upper-cased forms), and html.unescape's tables
  for numeric references (the invalid code points it drops and the C1 references it remaps);
- the full-width character map;
- the badness and UTF-8-detector regular expressions, with Python's \\s and \\w spelled out (its \\s also matches U+001C
  to U+001F, and its \\w is str.isalnum() or '_', neither as .NET's classes are; astral letters, which a .NET class
  cannot hold, are left out of \\w).
in src/TextToSpeech/FrontEnd/Resources/ftfy_tables.json.gz, and records fix_text of a corpus (clean text, HTML
entities, ligatures, full-width and curly characters, control and C1 characters, terminal escapes, line breaks, and
mojibake made through each codec, nested and partial) in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/
ftfy_reference.json. Default: verify both; --write: regenerate them. Needs ftfy==6.3.1.
"""
import argparse
import gzip
import html
import importlib.metadata
import io
import json
import sys

import _fixture

TABLES = "src/TextToSpeech/FrontEnd/Resources/ftfy_tables.json.gz"
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/ftfy_reference.json"
SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Café crème brûlée, naïve façade, résumé — 10 € please.",
    "Привет, как дела? Всё хорошо.",
    "Ελληνικά γράμματα και λέξεις.",
    "Żółć gęślą jaźń, Łódź.",
    "日本語のテキストです。",
    "“Smart quotes” and ‘single ones’ … ellipsis",
    "Emoji 😀 and symbols ™ © ® ° ±",
    "Ağaç, şeker, İstanbul, ılık.",
]


def gz(text):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def python_whitespace():
    return "".join(chr(c) for c in range(0x110000) if chr(c).isspace())


def ranges(codes):
    codes = sorted(codes)
    out, start, prev = [], None, None
    for c in codes:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append((start, prev))
            start = prev = c
    if start is not None:
        out.append((start, prev))
    return "".join("\\u%04x" % a if a == b else "\\u%04x-\\u%04x" % (a, b) for a, b in out)


def tables():
    import ftfy  # noqa: F401  (registers the sloppy codecs)
    from ftfy import badness, chardata
    codecs = {e: bytes(range(256)).decode(e, errors="replace") for e in chardata.CHARMAP_ENCODINGS}
    space = "[" + "".join("\\u%04x" % ord(c) for c in python_whitespace()) + "]"
    word = "[" + ranges(c for c in range(0x10000) if chr(c).isalnum() or chr(c) == "_") + "]"

    def dotnet(pattern):
        rest = pattern.replace("\\s", "").replace("\\w", "").replace("\\xa0", "").replace("\\xad", "")
        if "\\" in rest:
            raise SystemExit("unexpected escape in a pattern")
        return pattern.replace("\\s", space).replace("\\w", word)

    return {
        "charmap_encodings": chardata.CHARMAP_ENCODINGS,
        "codecs": codecs,
        "html_entities": chardata.HTML_ENTITIES,
        "invalid_charrefs": {str(k): v for k, v in html._invalid_charrefs.items()},
        "invalid_codepoints": sorted(html._invalid_codepoints),
        "width_map": {str(k): v for k, v in chardata.WIDTH_MAP.items()},
        "badness": dotnet(badness.BADNESS_RE.pattern),
        "utf8_detector": dotnet(chardata.UTF8_DETECTOR_RE.pattern),
    }


def corpus():
    cases = list(SAMPLES)
    cases += [
        "Fish &amp; chips &lt;3 &eacute;l&egrave;ve &EACUTE; &#233; &#x41; &#128; &#0; &#59; &bogus; &amp",
        "<b>bold &amp; tags</b> stay &amp; escaped",
        "line one\n<i>html</i>\nline &amp; three",
        "ﬁne ﬂour, ﬀ ligatures and Ĳssel",
        "ＦＵＬＬ　ｗｉｄｔｈ　１２３！",
        "tab\tand\x00null\x07bell\x1bescape \x1b[31mred\x1b[0m done",
        "C1 \x80 \x93quoted\x94 \x85 next",
        "windows\r\nmac\rline sep para",
        "﻿bom and ​ zero-width ￹ annotations",
        "lone \ud83d surrogate",
        "Â´ and Ã and Ã  and Ã¡ plain",
        "single Ã© in clean text, and a Â nbsp",
    ]
    for text in SAMPLES:
        for encoding in ("windows-1252", "latin-1", "windows-1251", "macroman", "cp437", "iso-8859-2", "windows-1250"):
            try:
                cases.append(text.encode("utf-8").decode(encoding))
            except UnicodeDecodeError:
                pass
        try:
            cases.append(text.encode("utf-8").decode("windows-1252").encode("utf-8").decode("windows-1252"))
        except UnicodeDecodeError:
            pass
        mixed = text.encode("utf-8").decode("latin-1")
        cases.append(text + " | " + mixed)
    cases.append("Ã¡rbol and árbol")
    cases.append("I like Ã\xa0 la carte")
    cases.append("Ã\xa0 Ã©t Ã¨ C\xc3\xa0")
    cases.append("â€œquotedâ€\x9d and â€™s")
    cases.append("â€œquotedâ€ and â€™s")
    cases.append("\xc3\xa9\xed\xa0\xbd\xed\xb8\x80 cesu")
    return [{"text": t, "fixed": __import__("ftfy").fix_text(t)} for t in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("ftfy") != "6.3.1":
        raise SystemExit("needs ftfy 6.3.1")
    blob = gz(json.dumps(tables(), ensure_ascii=False, sort_keys=True))
    data = {"corpus": corpus()}
    if args.write:
        with open(_fixture.fixture_path(TABLES), "wb") as handle:
            handle.write(blob)
        _fixture.save(FIXTURE, data, __file__)
        print(f"wrote {TABLES} ({len(blob)} bytes) and {FIXTURE} ({len(data['corpus'])} cases)")
        return
    with open(_fixture.fixture_path(TABLES), "rb") as handle:
        same = handle.read() == blob
    same = same and _fixture.load(FIXTURE)["corpus"] == data["corpus"]
    print("ftfy: tables and corpus match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
