"""Exports Open JTalk 1.11's UTF-8 rule tables (the *_rule_utf_8.h headers of text2mecab, the njd_set_* passes,
njd2jpcommon and jpcommon, as pyopenjtalk 0.3.4 vendors them; Modified BSD) for AiDotNet's port: every
`#define NAME "string"` and every `static const char *name[] = { ... }` string list, NULL entries kept as null so
the port's loops stop where the C loops do. The C sources are read, not retyped.

Usage: python openjtalk_rules.py OPEN_JTALK_SRC   (pyopenjtalk-0.3.4/lib/open_jtalk/src)
Writes src/TextToSpeech/FrontEnd/Resources/openjtalk_rules.json.gz.
"""
import glob
import gzip
import json
import os
import re
import sys

ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "0": "\0"}


def tokens(source):
    # C tokens that matter here: string literals (decoded as UTF-8), identifiers, punctuation; comments dropped.
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if source.startswith("/*", i):
            i = source.index("*/", i + 2) + 2
        elif source.startswith("//", i):
            i = source.index("\n", i)
        elif c == '"':
            j, out = i + 1, bytearray()
            while source[j] != '"':
                if source[j] == "\\":
                    out += ESCAPES[source[j + 1]].encode("utf-8")
                    j += 2
                else:
                    out += source[j].encode("utf-8")
                    j += 1
            yield ("str", out.decode("utf-8"))
            i = j + 1
        elif c.isalnum() or c == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            yield ("id", source[i:j])
            i = j
        elif c.isspace():
            i += 1
        else:
            yield ("punct", c)
            i += 1


def parse(path):
    source = open(path, encoding="utf-8").read()
    defines, lists = {}, {}
    for match in re.finditer(r'^#define\s+(\w+)\s+(".*?")\s*(?:/\*.*\*/)?\s*$', source, re.MULTILINE):
        defines[match.group(1)] = next(tokens(match.group(2)))[1]
    stream = list(tokens(re.sub(r"^#.*$", "", source, flags=re.MULTILINE)))
    for k in range(len(stream) - 6):
        if stream[k] == ("id", "char") and stream[k + 1] == ("punct", "*") and stream[k + 2][0] == "id" \
                and stream[k + 3] == ("punct", "[") and stream[k + 4] == ("punct", "]") and stream[k + 5] == ("punct", "="):
            name, values, j = stream[k + 2][1], [], k + 7
            while stream[j] != ("punct", "}"):
                kind, value = stream[j]
                if kind == "str":
                    values.append(value)
                elif stream[j] == ("id", "NULL"):
                    values.append(None)
                elif stream[j] != ("punct", ","):
                    raise ValueError(f"{path}: unexpected {stream[j]} in {name}")
                j += 1
            lists[name] = values
    return defines, lists


def main():
    root = sys.argv[1]
    defines, lists = {}, {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*_rule_utf_8.h"))):
        if "_ascii_for_" in path:
            continue
        d, l = parse(path)
        defines.update(d)
        lists.update(l)
    data = json.dumps({"defines": defines, "lists": lists}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources", "openjtalk_rules.json.gz")
    with open(out, "wb") as handle:
        handle.write(gzip.compress(data, mtime=0))
    print(len(defines), "defines;", {k: len(v) for k, v in lists.items()})


if __name__ == "__main__":
    main()
