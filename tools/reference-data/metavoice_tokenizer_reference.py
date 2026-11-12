"""Original trained BPE and author flattened/tilted adapters, including malformed tails."""
import argparse
import base64
import json
from pathlib import Path
import pickle
import random
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from fam.quantiser.text.tokenise import TrainedBPETokeniser
from fam.llm.adapters.flattened_encodec import FlattenedInterleavedEncodec2Codebook
from fam.llm.adapters.tilted_encodec import TiltedEncodec
from fam.llm.utils import normalize_text
release = json.loads(args.release.read_text())
path = release['checkpoint_paths']['first_stage.pt']
checkpoint = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
metadata = checkpoint['meta']['tokenizer']
ranks = metadata['mergeable_ranks']
texts = ['', 'Hello, world!', "She's I'M we'd Can't 12345\n\t ", '𠀀éñ中🙂\u0301\u200d\u00a0\r\n', 'a  \n b\t\t  ']
rng = random.Random(229471)
alphabet = list('abc XYZ01234!?-.,\t\r\néñ中🙂𠀀')
texts += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 120))) for _ in range(80)]
stages = []
for offset in (2049, 1025):
    config = dict(metadata, offset=offset)
    tokenizer = TrainedBPETokeniser(**config)
    stages.append(dict(offset=offset, examples=[dict(text=text, ids=tokenizer.encode(text),
        decoded=tokenizer.decode(tokenizer.encode(text))) for text in texts]))
adapters = []
for flattened, streams in [(True, [[2050, 2561, 2, 1027, 4, 1029, 2048]]),
        (True, [[2561, 2, 1027, 4, 2048, 2050]]), (True, [[2561, 2048]]),
        (False, [[1026, 1537, 2, 3, 1024], [1024, 1024, 4, 5, 1024], [1024, 1024, 6, 7, 1024]]),
        (False, [[1537, 2, 3, 1024], [1024, 4, 1024], [1024, 6, 7, 8]])]:
    adapter = FlattenedInterleavedEncodec2Codebook(1024) if flattened else TiltedEncodec(1024)
    text, audio = adapter.decode(streams)
    adapters.append(dict(flattened=flattened, streams=streams, text=text, audio=audio))
normalization = []
for text in ['', ' \tHello** world\n', 'A\u00a0B', 'A\u00a0\u00a0B', '\x1c\x1dA\x1e\x1fB\x1c',
        ''.join(chr(x) for x in [8175, 8189, 8190, *range(8208, 8215), *range(8216, 8224), 8228, 8229, 8230, 8242, 8243, 8245, 8246, 180, 2122]),
        'élève ñ', '中文', '🙂', '™', 'A\u0301', ''.join(chr(i) for i in range(256))]:
    try:
        normalization.append(dict(text=text, normalized=normalize_text(text)))
    except ValueError:
        normalization.append(dict(text=text, rejected=True))
result = dict(source='https://github.com/metavoiceio/metavoice-src/tree/de3fa211ac4621e03a5f990651aeecc64da418f5',
    tiktoken='0.5.1', ranks=[dict(bytes=base64.b64encode(key).decode(), rank=value) for key, value in ranks.items()],
    stages=stages, adapters=adapters, normalization=normalization,
    metadata_pickle_base64=base64.b64encode(pickle.dumps(dict(meta=dict(tokenizer=metadata),
        model_args=dict(spkemb_dropout=.1, rmsnorm_eps=1e-5)), protocol=2)).decode(),
    metadata_pickle4_base64=base64.b64encode(pickle.dumps(dict(meta=dict(tokenizer=metadata),
        model_args=dict(spkemb_dropout=.1, rmsnorm_eps=1e-5)), protocol=4)).decode())
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_tokenizer_reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(examples=len(texts) * 2, adapters=len(adapters), ranks=len(ranks))))
