"""Original community VALL-E X sampler and complete AR loop with controlled logits."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F

checkout, output = map(Path, sys.argv[1:3])
path = checkout / 'models/vallex.py'
source = path.read_bytes()
tree = ast.parse(source.decode('utf-8'))
loop = next(n for n in ast.walk(tree) if isinstance(n, ast.While) and any(
    isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == 'topk_sampling' for c in ast.walk(n)))
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['top_k_top_p_filtering', 'topk_sampling']]
controllers = []
for prepend, text_length in [(False, 1), (True, 1), (False, 2), (True, 2)]:
    ns = {'torch': torch, 'F': F, 'NUM_AUDIO_TOKENS': 4,
        'x': torch.zeros(1, text_length, 8), 'x_lens': torch.tensor([text_length]),
        'x_len': torch.tensor(text_length), 'x_attn_mask': torch.zeros(text_length, text_length, dtype=torch.bool),
        'y': torch.tensor([[5, 1] if prepend else [1]]), 'prompts': torch.zeros(1, 1, 3, dtype=torch.long),
        'prefix_len': 1, 'kv_cache': None, 'use_kv_caching': True, 'sum_logprobs': torch.zeros(1),
        'temperature': 1., 'top_k': 1, 'length_penalty': 1., 'return_worst': False}
    ns['self'] = SimpleNamespace(ar_audio_prepend_bos=prepend,
        ar_audio_embedding=lambda ids: torch.zeros(1, ids.shape[1], 8),
        ar_audio_prenet=lambda value: value, ar_audio_position=lambda value: value,
        ar_decoder=SimpleNamespace(infer=lambda *args, **kwargs: (torch.zeros(1, 1, 8), [])),
        ar_predict_layer=lambda value: torch.tensor([[10., 0., 0., 0., 0.]]))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=helpers + [loop], type_ignores=[]), str(path), 'exec'), ns)
    # Original post-loop extraction, including the BOS offset.
    inference = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and loop in n.body)
    extraction = next(n for n in inference.body if isinstance(n, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'codes' for target in n.targets))
    exec(compile(ast.Module(body=[extraction], type_ignores=[]), str(path), 'exec'), ns)
    codes = ns['codes'][0][0].tolist()
    controllers.append({'prepend_bos': prepend, 'text_length': text_length,
        'prompt_frames': 1, 'max_codes_per_text_token': 16, 'generated_codes': codes})

ns = {'torch': torch, 'F': F}
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), ns)
sampling = []
definitions = [([0, 1, 2, 3, 4], 1., 1), ([3, 2, 2, 2, 1], .8, 2),
               ([-2, -1, -.3, -4, -.8], 1.7, 0), ([3, 1, 0, -1, -2], .5, 3)]
original_draw = torch.multinomial
for logits, temperature, topk in definitions:
    captured = []
    def capture(probabilities, num_samples):
        captured.append(probabilities[0].tolist())
        return probabilities.argmax(-1, keepdim=True)
    try:
        torch.multinomial = capture
        ns['topk_sampling'](torch.tensor([logits], dtype=torch.float64), top_k=topk, temperature=temperature)
    finally:
        torch.multinomial = original_draw
    sampling.append({'logits': logits, 'temperature': temperature, 'top_k': topk,
        'probabilities': captured[0], 'uniforms': [0., .000001, .1, .5, .8, .999999]})
result = {'code_revision': subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': hashlib.sha256(source).hexdigest(), 'torch_version': torch.__version__,
    'scope': 'Community reproduction, not unreleased Microsoft checkpoint; controlled neural logits and categorical draws.',
    'controller_cases': controllers, 'sampling_cases': sampling}
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Wrote', len(controllers), 'original controller and', len(sampling), 'original sampler cases')
