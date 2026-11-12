"""Original Coqui prefix construction plus original supported HF sampling policy.

Arguments: coqui-TTS checkout, uninstalled Transformers 4.33.0 wheel, output JSON.
4.33.0 is the author's supported lower bound, not an exact dependency pin.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, MethodType
from typing import Optional
import zipfile
import torch
import torch.nn.functional as F

checkout, wheel, output = map(Path, sys.argv[1:4])
assert 'transformers>=4.33.0' in (checkout / 'requirements.txt').read_text(encoding='utf-8')
assert '4.33.0' in wheel.name
author = (checkout / 'TTS/tts/layers/xtts/gpt.py').read_bytes()
author_tree = ast.parse(author.decode('utf-8'))
prefix_method = next(n for n in ast.walk(author_tree) if isinstance(n, ast.FunctionDef) and n.name == 'compute_embeddings')
namespace = {'torch': torch, 'F': F, 'Optional': Optional,
             'add_start_docstrings': lambda *args: lambda obj: obj,
             'LOGITS_PROCESSOR_INPUTS_DOCSTRING': ''}
exec(compile(ast.Module(body=[prefix_method], type_ignores=[]), 'original-gpt.py', 'exec'), namespace)
stub = SimpleNamespace(start_text_token=8, stop_text_token=9, start_audio_token=6,
    text_embedding=lambda ids: torch.zeros(1, ids.shape[1], 4),
    text_pos_embedding=lambda ids: torch.zeros(1, ids.shape[1], 4),
    gpt_inference=SimpleNamespace(store_prefix_emb=lambda emb: None))
prefix = namespace['compute_embeddings'](stub, torch.zeros(1, 2, 4), torch.tensor([[2, 3]]))[0].tolist()
gpt_class = next(n for n in author_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GPT')
replay_names = {'set_inputs_and_targets', 'set_mel_padding', 'forward', 'generate'}
replay_methods = [n for n in gpt_class.body if isinstance(n, ast.FunctionDef) and n.name in replay_names]
exec(compile(ast.Module(body=replay_methods, type_ignores=[]), 'original-gpt-replay.py', 'exec'), namespace)
stub.compute_embeddings = MethodType(namespace['compute_embeddings'], stub)
stub.stop_audio_token = 7
stub.max_gen_mel_tokens = 3
stub.max_conditioning_inputs = 1
stub.code_stride_len = 4
stub.training = False
stub.text_head = stub.mel_head = None
stub.set_inputs_and_targets = MethodType(namespace['set_inputs_and_targets'], stub)
stub.set_mel_padding = MethodType(namespace['set_mel_padding'], stub)
stub.mel_embedding = lambda ids: torch.stack([ids.float(), torch.arange(ids.shape[1]).expand(1, -1).float()], dim=-1)
stub.mel_pos_embedding = lambda ids: torch.zeros(1, ids.shape[1], 2)
stub.get_logits = lambda first, first_head, second, second_head, **kwargs: (first, second)
stub.gpt_inference.generate = lambda ids, **kwargs: torch.cat([ids, torch.tensor([[7]])], dim=1)
raw_eos = namespace['generate'](stub, torch.zeros(1, 2, 4), torch.tensor([[2, 3]]))
replayed = namespace['forward'](stub, torch.tensor([[2, 3]]), torch.tensor([2]), raw_eos,
    torch.tensor([raw_eos.shape[-1] * stub.code_stride_len]), cond_latents=torch.zeros(1, 2, 4), return_latent=True)
eos_trace = {'generated_codes': raw_eos[0].tolist(), 'latent_frames': replayed.shape[1],
             'latent_input_ids': replayed[0, :, 0].long().tolist(), 'latent_positions': replayed[0, :, 1].long().tolist()}
with zipfile.ZipFile(wheel) as package:
    policy = package.read('transformers/generation/logits_process.py')
    pipeline = package.read('transformers/generation/utils.py')
names = {'LogitsProcessor', 'LogitsWarper', 'TemperatureLogitsWarper', 'TopKLogitsWarper',
         'TopPLogitsWarper', 'RepetitionPenaltyLogitsProcessor'}
selected = [n for n in ast.parse(policy.decode('utf-8')).body if isinstance(n, ast.ClassDef) and n.name in names]
exec(compile(ast.Module(body=selected, type_ignores=[]), 'original-hf433-policy.py', 'exec'), namespace)
warpers = {'TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper'}
method = next(n for n in ast.walk(ast.parse(pipeline.decode('utf-8')))
              if isinstance(n, ast.FunctionDef) and n.name == '_get_logits_warper')
order = [n.func.id for n in sorted(ast.walk(method), key=lambda n: getattr(n, 'lineno', 0))
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in warpers]
assert order == ['TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper']
definitions = [
    ('released-defaults', [3, 2, 1, 0, -1, -2, 1.5, -.5], [0, 0, 4], .75, 50, .85, 10),
    ('topk-one-max-ties', [2, 2, 2, 1, 0, -1, -2, -3], [], 1., 1, 1., 1.),
    ('topk-cutoff-ties', [3, 2, 2, 2, 1, 0, -1, -2], [], 1., 2, 1., 1.),
    ('uniform-half-nucleus', [0] * 8, [], 1., 0, .5, 1.),
    ('uniform-eighth-nucleus', [0] * 8, [], 1., 0, .125, 1.),
    ('zero-nucleus', [3, 2, 1, 0, -1, -2, -3, -4], [], 1., 0, 0., 1.),
    ('positive-prefix-penalty', [0, 8, 0, 0, 0, 0, 2, 3], [], 1., 1, 1., 10.),
    ('bos-prefix-penalty', [0, 0, 0, 0, 0, 0, 8, 3], [], 1., 1, 1., 10.),
    ('negative-repeat-and-temperature', [-2, -1, -.4, -.7, -3, -4, -2, -1.5], [2, 2, 4], 1.3, 0, 1., 1.7),
    ('immediate-eos', [0, 0, 0, 0, 0, 0, 0, 5], [], 1., 1, 1., 1.),
    ('speech-then-eos', [8, 0, 0, 0, 0, 0, 0, 3], [], 1., 1, 1., 10.),
    ('raw-bos-at-generation-cap', [0, 0, 0, 0, 0, 0, 8, 3], [], 1., 1, 1., 1.),
]
cases = []
for name, logits, generated, temperature, topk, topp, penalty in definitions:
    history = prefix + generated
    ids = torch.tensor([history])
    scores = torch.tensor([logits], dtype=torch.float64)
    scores = namespace['RepetitionPenaltyLogitsProcessor'](float(penalty))(ids, scores)
    scores = namespace['TemperatureLogitsWarper'](float(temperature))(ids, scores)
    if topk > 0:
        scores = namespace['TopKLogitsWarper'](topk)(ids, scores)
    scores = namespace['TopPLogitsWarper'](float(topp))(ids, scores)
    case = {'name': name, 'logits': logits, 'history': history, 'generated': generated,
        'temperature': temperature, 'top_k': topk, 'top_p': topp, 'repetition_penalty': penalty,
        'probabilities': scores.softmax(-1)[0].tolist(), 'uniforms': [.000001, .1, .3, .5, .8, .999999]}
    if name in {'positive-prefix-penalty', 'bos-prefix-penalty', 'immediate-eos', 'speech-then-eos', 'raw-bos-at-generation-cap'}:
        scripted = []
        for step in range(3):
            current_ids = torch.tensor([prefix + scripted])
            current = namespace['RepetitionPenaltyLogitsProcessor'](float(penalty))(
                current_ids, torch.tensor([logits], dtype=torch.float64))
            current = namespace['TemperatureLogitsWarper'](float(temperature))(current_ids, current)
            current = namespace['TopKLogitsWarper'](topk)(current_ids, current)
            current = namespace['TopPLogitsWarper'](float(topp))(current_ids, current)
            assert torch.isfinite(current).sum() == 1
            scripted.append(current.argmax(-1).item())
            if scripted[-1] == 7:
                break
        stub.gpt_inference.generate = lambda ids, **kwargs: torch.cat([ids, torch.tensor([scripted])], dim=1)
        raw = namespace['generate'](stub, torch.zeros(1, 2, 4), torch.tensor([[2, 3]]))
        trace = namespace['forward'](stub, torch.tensor([[2, 3]]), torch.tensor([2]), raw,
            torch.tensor([raw.shape[-1] * stub.code_stride_len]), cond_latents=torch.zeros(1, 2, 4), return_latent=True)
        case['generation_trace'] = {'generated_codes': raw[0].tolist(), 'latent_frames': trace.shape[1],
            'latent_input_ids': trace[0, :, 0].long().tolist(), 'latent_positions': trace[0, :, 1].long().tolist()}
    cases.append(case)
result = {'code_revision': subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
    'author_sha256': hashlib.sha256(author).hexdigest(), 'transformers_reference_version': '4.33.0',
    'dependency_constraint': '>=4.33.0', 'policy_sha256': hashlib.sha256(policy).hexdigest(),
    'pipeline_sha256': hashlib.sha256(pipeline).hexdigest(), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
    'torch_version': torch.__version__, 'prefix': prefix, 'immediate_eos_replay': eos_trace, 'cases': cases}
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Wrote', len(cases), 'original XTTS/HF policy cases; original prefix:', prefix)
