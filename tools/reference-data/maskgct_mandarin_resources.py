"""Exports the data of MaskGCT's Mandarin front end (Amphion models/tts/maskgct/g2p at 26f6883, MIT) for AiDotNet:
mandarin.py's Latin-to-bopomofo and bopomofo-to-IPA tables (taken from the module itself, not retyped), its lexicon
(sources/chinese_lexicon.txt), the pinyin-to-bopomofo table (sources/pinyin_2_bpmf.txt), and the polyphone model's
character list, label dictionary and BERT vocabulary (sources/g2p_chinese_model: polychar.txt, polydict.json,
vocab.txt), and the code points Python's str.isnumeric accepts (mandarin.py's 一 sandhi tests it) as ranges. The polyphone
model's weights (poly_bert_model.onnx) are not exported; the model loads them from the release's file.

Run it with the release's Python (3.9, in refsrc/maskgct-docker) so isnumeric follows its Unicode tables:
    docker run --rm -v <Amphion>:/amphion -v <repo>:/repo maskgct-g2p python /repo/tools/reference-data/maskgct_mandarin_resources.py /amphion

Usage: python maskgct_mandarin_resources.py AMPHION_ROOT
Writes src/TextToSpeech/FrontEnd/Resources/maskgct_mandarin.json.gz.
"""
import ast
import gzip
import json
import os
import sys


def table(source, name):
    # The list literal a module-level assignment builds its regexes from: [(pattern, replacement), ...].
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            comprehension = node.value
            if isinstance(comprehension, ast.ListComp):
                return [list(pair) for pair in ast.literal_eval(comprehension.generators[0].iter)]
            raise ValueError(f"{name} is not built by a list comprehension")
    raise KeyError(name)


def numeric_ranges():
    ranges = []
    for code in range(0x110000):
        if chr(code).isnumeric():
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1][1] = code
            else:
                ranges.append([code, code])
    return ranges


def main():
    root = sys.argv[1]
    g2p = os.path.join(root, "models", "tts", "maskgct", "g2p")
    source = open(os.path.join(g2p, "g2p", "mandarin.py"), encoding="utf-8").read()
    sources = os.path.join(g2p, "sources")

    def lines(path):
        with open(path, encoding="utf-8") as handle:
            return handle.readlines()

    lexicon = []
    for line in lines(os.path.join(sources, "chinese_lexicon.txt")):
        word, pinyin = line.strip().split("\t")
        lexicon.append([word, pinyin])
    pinyin_to_bopomofo = [line.strip().split("\t") for line in lines(os.path.join(sources, "pinyin_2_bpmf.txt"))]
    model = os.path.join(sources, "g2p_chinese_model")
    result = {
        "latin_to_bopomofo": table(source, "_latin_to_bopomofo"),
        "bopomofo_to_ipa": table(source, "_bopomofo_to_ipa"),
        "lexicon": lexicon,
        "pinyin_to_bopomofo": pinyin_to_bopomofo,
        "polychar": [line.strip("\n") for line in lines(os.path.join(model, "polychar.txt"))],
        "polydict": json.load(open(os.path.join(model, "polydict.json"), encoding="utf-8")),
        "bert_vocab": [line.rstrip("\n") for line in lines(os.path.join(model, "vocab.txt"))],
        "numeric_ranges": numeric_ranges(),
        "python": sys.version.split()[0],
    }
    data = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources", "maskgct_mandarin.json.gz")
    with open(out, "wb") as handle:
        handle.write(gzip.compress(data, mtime=0))
    print({k: len(v) for k, v in result.items()}, result["python"])


if __name__ == "__main__":
    main()
