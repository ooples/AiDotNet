"""Builds the English G2P's embedded resources from their pinned sources.

- src/TextToSpeech/FrontEnd/Resources/cmudict.tsv.gz: the CMU Pronouncing Dictionary (cmusphinx/cmudict, cmudict.dict
  at commit 74790861f6; BSD-2-Clause, see NOTICE), one "word<TAB>ARPABET" line per word with its first pronunciation,
  sorted, gzipped deterministically.
- src/TextToSpeech/FrontEnd/Resources/nrl_rules.tsv: the letter-to-sound rules of NRL Report 7948 (Elovitz, Johnson,
  McHugh and Shore 1976, a US Government work), extracted from the report's SNOBOL4 TRANS program as typed in
  Lord-Nightmare/NRL_TextToPhonemes (snobol/TRANS.SNO, commit e998cf7a34): one "group<TAB>left<TAB>match<TAB>right<TAB>
  phones" line per rule, in the program's order.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import gzip
import io
import re

import _fixture

CMUDICT_OUT = "src/TextToSpeech/FrontEnd/Resources/cmudict.tsv.gz"
NRL_OUT = "src/TextToSpeech/FrontEnd/Resources/nrl_rules.tsv"


def build_cmudict(path):
    entries = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if not line:
                continue
            head, *phones = line.split()
            word = re.sub(r"\(\d+\)$", "", head)
            entries.setdefault(word, " ".join(phones))      # the first pronunciation
    text = "".join(f"{w}\t{p}\n" for w, p in sorted(entries.items())).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as gz:
        gz.write(text)
    return buffer.getvalue(), len(entries)


def build_nrl(path):
    text = open(path, encoding="latin-1").read()
    blocks = re.findall(r"^\s+(\w+)RULE\.ENG\s*=\s*\n((?:\+.*\n)+)", text, re.M)
    lines = []
    for group, body in blocks:
        for line in body.splitlines():
            quoted = re.match(r"""^\+\s*(['"])(.*)\1\s*$""", line)
            if not quoted:
                raise ValueError(f"unparsed rule line in {group}: {line!r}")
            parts = re.match(r"^(.*?)\[(.*?)\](.*)=/(.*)/\\?$", quoted.group(2))
            if not parts:
                raise ValueError(f"unparsed rule in {group}: {quoted.group(2)!r}")
            left, match, right, phones = parts.groups()
            if "\t" in "".join(parts.groups()):
                raise ValueError("a rule contains a tab")
            lines.append("\t".join((group, left, match, right, phones)))
    return ("\n".join(lines) + "\n").encode("utf-8"), len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmudict", required=True, help="cmusphinx/cmudict cmudict.dict at commit 74790861f6")
    parser.add_argument("--trans", required=True, help="Lord-Nightmare/NRL_TextToPhonemes snobol/TRANS.SNO at e998cf7a34")
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    cmudict, words = build_cmudict(args.cmudict)
    nrl, rules = build_nrl(args.trans)
    if args.write:
        for relative, data in ((CMUDICT_OUT, cmudict), (NRL_OUT, nrl)):
            with open(_fixture.fixture_path(relative), "wb") as handle:
                handle.write(data)
        print(f"wrote {CMUDICT_OUT} ({words} words) and {NRL_OUT} ({rules} rules)")
        return
    check = _fixture.Comparison("G2P resources")
    for relative, data in ((CMUDICT_OUT, cmudict), (NRL_OUT, nrl)):
        with open(_fixture.fixture_path(relative), "rb") as handle:
            check.exact(relative, handle.read(), data)
    check.report()


if __name__ == "__main__":
    main()
