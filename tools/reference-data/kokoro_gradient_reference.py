"""Full released Kokoro neural-gradient audit, retaining the author's detached source.

Only the outer inference no_grad decorator is unwrapped. All neural modules and
the generator's internal no_grad source branch execute unchanged. This checks
neural differentiation, not a claim to recover Kokoro's unpublished trainer.
Large gradient tensors belong in the external audit directory, not the repo.
"""
import argparse
import functools
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import torch
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=Path)
p.add_argument('--release', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sys.argv = ['kokoro_pretrained_reference.py', '--source', str(a.source), '--release', str(a.release),
            '--output', str(a.output/'inference-reference.json')]
scope = runpy.run_path(str(Path(__file__).with_name('kokoro_pretrained_reference.py')))
model, tokens, style = scope['model'], scope['tokens'], scope['style']
matched = scope['precision_matched']
captured = {}
handles = []
for name, module in [('bert', model.bert), ('projected', model.bert_encoder),
                     ('duration', model.predictor.duration_proj), ('pitch', model.predictor.F0_proj),
                     ('energy', model.predictor.N_proj), ('text', model.text_encoder)]:
    def capture(_, inputs, output, key=name): captured[key] = output
    handles.append(module.register_forward_hook(capture))
draws = iter(matched['excitation_draws'])
def draw(kind):
    entry = next(draws)
    assert entry['kind'] == kind
    return torch.tensor(entry['values'], dtype=torch.float64).reshape(entry['shape'])
style = style.detach().requires_grad_()
with scope['DoubleBuffers'](), patch('torch.rand', side_effect=lambda *args, **kwargs: draw('uniform')), \
        patch('torch.randn_like', side_effect=lambda tensor, **kwargs: draw('normal')):
    # __wrapped__ removes the public inference decorator, not a neural forward.
    waveform, durations = model.forward_with_tokens.__wrapped__(model, tokens, style, speed=20.)
captured['audio'] = waveform
order = ['bert', 'projected', 'duration', 'pitch', 'energy', 'text', 'audio']
loss = sum((value * torch.linspace(-.7, .9, value.numel()).reshape_as(value)).sum() * ((i+1)/100.)
           for i, name in enumerate(order) for value in [captured[name]])
loss.backward()
def legacy(name):
    return name.replace('.parametrizations.weight.original0', '.weight_g').replace('.parametrizations.weight.original1', '.weight_v')
expected_names = {group+'.'+name.removeprefix('module.') for group, state in scope['checkpoint'].items() for name in state}
tensors, missing = {}, []
for name, parameter in model.named_parameters():
    key = legacy(name)
    if key not in expected_names: continue  # recorded identity ONNX InstanceNorm extensions
    if parameter.grad is None:
        missing.append(key)
    else:
        tensors[key] = parameter.grad.contiguous()
assert sorted(missing) == sorted(['bert.pooler.weight', 'bert.pooler.bias',
                                  'decoder.generator.m_source.l_linear.weight', 'decoder.generator.m_source.l_linear.bias']), missing
assert len(tensors) == 544
tensors['input.style'] = style.grad.contiguous()
save_file(tensors, str(a.output/'gradients.safetensors'))
metadata = dict(source_revision=scope['result']['source_revision'], checkpoint_sha256=scope['result']['checkpoint_sha256'],
                torch=torch.__version__, unchanged_neural_forwards=True, inference_decorator_only_unwrapped=True,
                original_detached_harmonic_source=True, loss=loss.item(), gradient_tensors=544,
                intentionally_no_gradient=missing, objective_outputs=order,
                outputs={name: value.detach().flatten().tolist() for name,value in captured.items()})
(a.output/'gradients.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(loss=loss.item(), gradients=544, intentionally_no_gradient=missing)))
