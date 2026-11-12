"""Export the released TorToiSe vocabulary and Unidecode 1.3.8 mappings; generate real VoiceBpeTokenizer cases.

Argument: neonbjb/tortoise-tts checkout. Requires its inflect/tokenizers dependencies and Unidecode 1.3.8.
"""
import gzip
import json
from pathlib import Path
import sys
import types
import warnings
import unidecode

root = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[2]
resources = repo / 'src/TextToSpeech/FrontEnd/Resources'
with gzip.open(resources / 'tortoise_tokenizer.json.gz', 'wb') as output:
    output.write((root / 'tortoise/data/tokenizer.json').read_bytes())
warnings.simplefilter('ignore')
table = {}
for code in range(128, 0x110000):
    value = unidecode.unidecode(chr(code))
    if value:
        table[code] = value
with gzip.open(resources / 'unidecode.json.gz', 'wt', encoding='utf-8') as output:
    json.dump(table, output, ensure_ascii=True, separators=(',', ':'))
package = types.ModuleType('tortoise')
package.__path__ = [str(root / 'tortoise')]
sys.modules['tortoise'] = package
from tortoise.utils.tokenizer import VoiceBpeTokenizer
tokenizer = VoiceBpeTokenizer()
texts = ['Hello, world!', "Don't stop... (really?)", 'Dr. Smith paid $12.50 in 2005.',
    'He finished 21st in 1905.', 'Mrs. Jones has £1,234 and $0.01.',
    '  A\tB\nC  ', 'Émile, Straße, Łódź, 你好.', '“Hello”—she said.',
    'zero 0 1000 1001 2000 2010 2999 3000', 'a/b @#$%^&*{}', '😀 hello']
data = dict(source='neonbjb/tortoise-tts/tortoise/utils/tokenizer.py VoiceBpeTokenizer; inflect 7.0.0; Unidecode 1.3.8',
    cases=[dict(text=text, cleaned=tokenizer.preprocess_text(text), ids=tokenizer.encode(text)) for text in texts])
out = repo / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_text_reference.json'
out.write_text(json.dumps(data, ensure_ascii=True, separators=(',', ':')), encoding='utf-8')
print(len(table), 'nonempty transliterations;', out)
