"""Execute the original UniAudio search loop with controlled neural logits.

Only the neural model and cache-hook installation are replaced. InferenceImp,
its masks, length gate, complete local search, and chunk_after_end are original.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import torch
checkout = Path(sys.argv[1]).resolve()
path = checkout / 'UniAudio/infer.py'
source = path.read_bytes()
tree = ast.parse(source.decode('utf-8'))
selected = [node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == 'InferenceImp'
    or isinstance(node, ast.FunctionDef) and node.name in ['chunk_after_end', 'select_cache']]
namespace = {'torch': torch, 'logging': logging,
             'install_kv_cache_hook': lambda layer, cache: (cache, [])}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
class ControlledModel:
    na_repeat = 3
    n_ctx = 100
    g_layers = []
    l_layers = []
    def __init__(self): self.calls = 0
    def __call__(self, *args): return torch.zeros(1)
    def global_forward(self, *args, **kwargs): return torch.zeros(1, 3, 1)
    def local_forward(self, *args):
        codebook = self.calls % 3
        self.calls += 1
        logits = torch.zeros(1, 1, 15, dtype=torch.float64)
        logits[0, 0, 8 + 2 * codebook] = 10
        logits[0, 0, 7] = 20
        return logits
vocabulary = ['<pad>', '<start>', '<end>', '<mega_global>', '<mega_local>',
              '<audio_start>', '<text_start>', '<audio_end>'] + [f'<audio_{i}>' for i in range(7)]
cases = []
for minimum_length in [0, 1, 2, 3, 4, 6]:
    model = ControlledModel()
    inference = namespace['InferenceImp'](model, vocabulary, mode='greedy',
        maxlen_ratio=4 / 9, minlen_ratio=minimum_length / 9, topk=1)
    result = inference(torch.tensor([1, 1, 1] * 8 + [5, 5, 5]), torch.zeros(27, dtype=torch.bool), [])
    raw = result[0]['tokens']
    codec = namespace['chunk_after_end'](raw, 7)
    cases.append({'minimum_length': minimum_length, 'prefix_patches': 9,
        'minimum_ratio': minimum_length / 9, 'maximum_frames': 4,
        'raw_tokens': raw.tolist(), 'codec_tokens': codec.tolist(),
        'codec_frames': len(codec) // 3, 'local_calls': model.calls})
result = {'code_revision': subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
    'author_sha256': hashlib.sha256(source).hexdigest(),
    'scope': 'Original greedy search controller, controlled logits; no trained-network replacement claim.',
    'cases': cases}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Wrote', len(cases), 'original UniAudio controller cases')
