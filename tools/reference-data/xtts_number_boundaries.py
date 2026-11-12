"""Adversarial number/currency/ordinal boundaries from the unmodified Coqui cleaner.

Run with a coqui-ai/TTS checkout and num2words==0.5.12 on PYTHONPATH.
Expected strings are produced solely by the authors' Python front end.
"""
import ast
import importlib
import json
import os
from pathlib import Path
import re
import sys
from num2words import num2words, CONVERTER_CLASSES

source = Path(sys.argv[1]) / 'TTS/tts/layers/xtts/tokenizer.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
namespace = dict(re=re, num2words=num2words, os=os, __file__=str(source))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))],
    type_ignores=[]), str(source), 'exec'), namespace)
suffixes = dict(en='th', fr='ème', es='º', de='.', nl='ste', pt='º', it='º', ru='-й', pl='th', hu='.', tr='.', ar='ر')
values = list(range(221)) + [299, 300, 399, 999, 1000, 1001, 1010, 1100, 2000, 2001, 10000,
    21000, 100000, 100001, 101001, 999999, 1000000, 1000001, 2000000, 123456789, 1000000000, 1000000000001]
cases = []
for language in ['en', 'fr', 'es', 'de', 'nl', 'pt', 'it', 'ru', 'pl', 'cs', 'hu', 'tr', 'ar']:
    texts = [str(value) for value in values]
    if language in suffixes:
        texts += [str(value) + suffixes[language] for value in values if value > 0]
    texts += [f'{value}{symbol}' for symbol in ['€', '$', '£'] for value in ['0', '1', '2', '11', '21', '100', '101', '1000', '1000000', '0.01', '1.50', '21.21']]
    texts += ['-1', '-21', '0.0', '0.01', '2.0', '10.05', '100.08', '1000.001']
    for text in texts:
        if language == 'ar':
            CONVERTER_CLASSES['ar'] = importlib.import_module('num2words.lang_AR').Num2Word_AR()
        try:
            expected = namespace['multilingual_cleaners'](text, language)
        except (NotImplementedError, OverflowError, ValueError):
            continue  # No defined author output for these inputs.
        cases.append(dict(language=language, text=text, expected=expected))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_number_boundaries.json'
destination.write_text(json.dumps(dict(source='Unmodified coqui-ai/TTS multilingual_cleaners; num2words 0.5.12',
    cases=cases), ensure_ascii=True, separators=(',', ':')), encoding='utf-8')
print(f'{len(cases)} author reference cases')
