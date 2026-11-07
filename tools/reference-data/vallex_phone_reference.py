"""Original community VALL-E-X HF tokenizer; no reimplementation of its token IDs."""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer

source = Path(sys.argv[1]) / 'utils/g2p/bpe_69.json'
tokenizer = Tokenizer.from_file(str(source))
texts = ['həlˈoʊ_wˈɝld.', 'ni→_xɑʊ↓↑.', 'tʃʰæŋ_θɹiː…',
         '[UNK]a[CLS]_[SEP][PAD][MASK]', 'a\tb\nc\rd\u0085e\u00a0f',
         'a😀b𐐀c', 'a b  d', '\u001ca\u001db\u001ec\u001fd', '',
         ''.join(chr(i) for i in range(32, 127))]
vocabulary = sorted(tokenizer.get_vocab().items(), key=lambda pair: pair[1])
data = dict(source='Plachtaa/VALL-E-X 3faaf8ccadb154d63b38070caf518ce9309ea0f4 utils/g2p/bpe_69.json',
            vocabulary=[token for token, _ in vocabulary], cases=[])
for text in texts:
    data['cases'].append(dict(phones=text, ids=tokenizer.encode(text.replace(' ', '_')).ids))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vallex_phone_reference.json'
destination.write_text(json.dumps(data), encoding='utf-8')
print(f'Generated {len(texts)} original HF phone-tokenizer cases and {len(vocabulary)} vocabulary IDs.')
