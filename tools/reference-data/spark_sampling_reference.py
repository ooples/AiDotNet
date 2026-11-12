"""Execute Spark's pinned HF4.46.2 sampling classes from an uninstalled wheel.

Arguments: Spark checkout, transformers-4.46.2 wheel. Only documentation
decorators are inert; original processor class bodies execute unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Optional
import zipfile
import torch

checkout, wheel = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
requirements = (checkout / 'requirements.txt').read_text(encoding='utf-8')
assert 'transformers==4.46.2' in requirements and '4.46.2' in wheel.name
with zipfile.ZipFile(wheel) as package:
    source = package.read('transformers/generation/logits_process.py')
    pipeline_source = package.read('transformers/generation/utils.py')
pipeline_tree = ast.parse(pipeline_source.decode('utf-8'))
warpers = {'TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper'}
pipeline_method = next(node for node in ast.walk(pipeline_tree) if isinstance(node, ast.FunctionDef)
    and warpers.issubset({call.func.id for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}))
actual_order = [call.func.id for call in sorted(ast.walk(pipeline_method), key=lambda node: getattr(node, 'lineno', 0))
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in warpers]
assert actual_order == ['TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper']
tree = ast.parse(source.decode('utf-8'))
names = {'LogitsProcessor', 'TemperatureLogitsWarper', 'TopKLogitsWarper', 'TopPLogitsWarper'}
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
namespace = dict(torch=torch, Optional=Optional, add_start_docstrings=lambda *args: lambda item: item,
                 LOGITS_PROCESSOR_INPUTS_DOCSTRING='')
exec(compile(ast.Module(body=classes, type_ignores=[]), '<original-HF4.46.2-sampling>', 'exec'), namespace)
author = (checkout / 'cli/SparkTTS.py').read_text(encoding='utf-8')
inference = next(node for node in ast.walk(ast.parse(author)) if isinstance(node, ast.FunctionDef) and node.name == 'inference')
call = next(node for node in ast.walk(inference) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'generate')
assert {'top_k', 'top_p', 'temperature', 'do_sample', 'max_new_tokens'}.issubset({k.arg for k in call.keywords})

def probabilities(values, temperature, top_k, top_p):
    scores = torch.tensor([values], dtype=torch.float64)
    history = torch.tensor([[1]])
    if temperature != 1:
        scores = namespace['TemperatureLogitsWarper'](temperature)(history, scores)
    if top_k > 0:
        scores = namespace['TopKLogitsWarper'](top_k)(history, scores)
    if top_p < 1:
        scores = namespace['TopPLogitsWarper'](top_p)(history, scores)
    return scores.softmax(-1)[0]

base = [1.2, -.7, .1, 2.4, .8, -1.5, 1., .2]
specifications = [
    ('released_defaults', base, .8, 50, .95),
    ('top_k_one_tied_maxima', [3., 3., 2., 1., 0., -1., -2., -3.], 1., 1, 1.),
    ('top_k_cutoff_ties', [4., 3., 3., 0., -1., -2., -3., -4.], 1., 2, 1.),
    ('top_p_exact_half', [0., 0.] + [-10000.] * 6, 1., 0, .5),
    ('top_p_zero', base, .8, 0, 0.),
    ('temperature_then_top_k_then_nucleus', base, 1.3, 4, .9),
    ('nucleus_after_tied_top_k', [4., 3., 3., 1., 0., -1., -2., -3.], .7, 3, .8),
    ('unfiltered_full_distribution', base, 1., 0, 1.),
    ('uniform_eighth_nucleus', [0.] * 8, 1., 0, .125),
]
data = dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
    author_sha256=hashlib.sha256(author.encode()).hexdigest(), transformers_version='4.46.2',
    policy_sha256=hashlib.sha256(source).hexdigest(), pipeline_sha256=hashlib.sha256(pipeline_source).hexdigest(),
    pipeline_method=pipeline_method.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), cases=[])
for label, values, temperature, top_k, top_p in specifications:
    probs = probabilities(values, temperature, top_k, top_p)
    draws = []
    for uniform in (0., .1, .4, .6, .9, .999999):
        cumulative = 0.
        for token in torch.argsort(probs, descending=True, stable=True).tolist():
            if probs[token] <= 0:
                continue
            cumulative += probs[token].item()
            if uniform <= cumulative:
                draws.append(dict(uniform=uniform, token=token))
                break
    data['cases'].append(dict(label=label, logits=values, temperature=temperature, top_k=top_k,
        top_p=top_p, probabilities=probs.tolist(), draws=draws))
data['uniform_topk_one'] = probabilities([0.] * 8, 1., 1, 1.).tolist()
output = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spark_sampling_reference.json'
output.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(output)
