"""Export Cutlet 0.5.0 word readings indexed by the existing UniDic-lite token ids.

Requires cutlet, fugashi, unidic-lite==1.0.8 and numpy. Cutlet is MIT;
UniDic-lite is BSD. The native segmenter retains the dictionary's token order.
No Python runtime is needed for native synthesis.
"""
import csv
import gzip
import json
import pathlib
import struct
from types import SimpleNamespace

import cutlet
import unidic_lite
from cutlet.cutlet import has_foreign_lemma
from mecab_unidic_reference import read_dictionary, keys_of


def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    directory = root / 'src/TextToSpeech/FrontEnd/Resources'
    path = pathlib.Path(unidic_lite.DICDIR) / 'sys.dic'
    units, tokens, count, _, _ = read_dictionary(path)
    raw = path.read_bytes()
    header = struct.unpack('<10I', raw[:40])
    features_start = 72 + header[6] + header[7]
    katsu = cutlet.Cutlet()
    entries = [None] * count
    for key, value in keys_of(units).items():
        surface = key.decode('utf-8')
        begin, length = value >> 8, value & 255
        for i in range(begin, begin + length):
            start = features_start + int(tokens[i]['feature'])
            fields = next(csv.reader([raw[start:raw.index(b'\0', start)].decode('utf-8')]))
            feature = SimpleNamespace(pos1=fields[0], pos2=fields[1], lemma=fields[7], pron=fields[9], kana=fields[17])
            word = SimpleNamespace(surface=surface, feature=feature, is_unk=False)
            try:
                reading = katsu.romaji_word(word)
            except KeyError as error:
                # Some dictionary entries are unsupported by upstream Cutlet too;
                # preserve that failure when selected, rather than inventing a reading.
                reading = '\0CutletKeyError:' + str(error)
            entries[i] = [reading, fields[0], fields[1], bool(has_foreign_lemma(word))]
    assert all(x is not None for x in entries)
    payload = dict(version='cutlet-0.5.0/unidic-lite-1.0.8', entries=entries, kana=katsu.table, exceptions=katsu.exceptions)
    target = directory / 'cutlet_unidic.json.gz'
    target.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    corpus = ['あ', 'アイウエオ', 'こんにちは。', '私は日本語を勉強しています。', 'カツカレーを食べた。', '東京へ行った。', 'きゃっと しんよう', "John's本", 'すもももももももものうち']
    fixture = dict(source=payload['version'], cases=[dict(text=s, expected=katsu.romaji(s).lower()) for s in corpus])
    (root / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_cutlet_reference.json').write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
    print(target, target.stat().st_size, 'bytes;', len(entries), 'dictionary tokens')


if __name__ == '__main__':
    main()
