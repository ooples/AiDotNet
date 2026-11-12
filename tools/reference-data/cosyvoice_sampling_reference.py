"""Run original CosyVoice RAS/nucleus/random sampling at fixed categorical quantiles.

Arguments: original v2.0 checkout, optional current CosyVoice checkout. Original
sampling functions execute by AST. Only Tensor.multinomial's random quantiles are
replayed, allowing exact tests without assuming shared Torch/.NET RNG algorithms.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch

base = [math.log(value) for value in (.9, .04, .03, .02, .01)]
specifications = [
    ('nucleus_without_repeat', base, [1, 2], .8, 10, .1, [.5]),
    ('fallback_can_repeat', base, [0, 3], .8, 10, .1, [.5, .5]),
    ('fallback_can_leave_nucleus', base, [0, 3], .8, 10, .1, [.5, .95]),
    ('old_repeat_outside_window', base, [0] + [4] * 10, .8, 10, .1, [.5]),
    ('below_repeat_threshold', base, [0], .8, 4, .5, [.5]),
    ('at_repeat_threshold', base, [0, 0], .8, 4, .5, [.5, .1]),
    ('large_positive_logit_shift', [v + 1000 for v in base], [], .95, 10, .1, [.99]),
    ('large_negative_logit_shift', [v - 1000 for v in base], [], .95, 10, .1, [.99]),
    ('unnormalized_logits', [2., 1., 0., -1., -2.], [], .8, 10, .1, [.9]),
    ('repeated_largest_nonzero_id', [base[1], base[2], base[3], base[4], base[0]], [4], .8, 10, .1, [.5, .5]),
    ('masked_fallback_at_zero_quantile', [-float('inf'), math.log(.9), math.log(.1)], [1], .8, 10, .1, [.5, 0.]),
]
original = torch.Tensor.multinomial
sources = []
for checkout_arg in sys.argv[1:]:
    checkout = Path(checkout_arg).resolve()
    path = checkout / 'cosyvoice/utils/common.py'
    source = path.read_text(encoding='utf-8')
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('ras_sampling', 'nucleus_sampling', 'random_sampling')]
    assert len(functions) == 3
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    cases = []
    for label, scores, decoded, top_p, window, ratio, uniforms in specifications:
        pending = list(uniforms)
        probabilities = []
        def multinomial(tensor, num_samples=1, replacement=False, **kwargs):
            assert num_samples == 1
            values = tensor / tensor.sum()
            probabilities.append(values.tolist())
            draw, cumulative = pending.pop(0), 0.
            for index, probability in enumerate(values.tolist()):
                if probability <= 0:
                    continue
                cumulative += probability
                if draw <= cumulative:
                    return torch.tensor([index], dtype=torch.long)
            raise AssertionError('No categorical outcome')
        torch.Tensor.multinomial = multinomial
        try:
            tensor = torch.tensor(scores, dtype=torch.float64)
            token = namespace['ras_sampling'](tensor,
                decoded, 25, top_p=top_p, top_k=25, win_size=window, tau_r=ratio)
        finally:
            torch.Tensor.multinomial = original
        assert not pending, (label, pending)
        cases.append(dict(label=label, scores=[None if v == -float('inf') else v for v in scores],
            decoded=decoded, top_p=top_p, top_k=25, window=window, ratio=ratio,
            uniforms=uniforms, token=int(token.item() if isinstance(token, torch.Tensor) else token), probabilities=probabilities,
            post_scores=[None if value == -float('inf') else value for value in tensor.tolist()]))
    llm_source = (checkout / 'cosyvoice/llm/llm.py').read_text(encoding='utf-8')
    controller = next(node for node in ast.walk(ast.parse(llm_source))
        if isinstance(node, ast.FunctionDef) and node.name == 'sampling_ids')
    namespace['List'] = list
    exec(compile(ast.Module(body=[controller], type_ignores=[]), '<original-sampling-ids>', 'exec'), namespace)
    rejection = any(isinstance(node, ast.While) for node in ast.walk(controller))
    generation_cases = []
    generation_specs = [
        ('eos_allowed', [.3, .2, 0., 0., .5], False, [.1], [.1]),
        ('reject_eos_then_accept', [.25, .15, 0., 0., .6], True, [.1, .9], [.1]),
        ('conditional_nucleus_differs', [.4, .2, 0., 0., .4], True, [.9, .1], [.9]),
        ('initial_eos_retry_limit', [.1, 0., 0., 0., .9], True, [.5] * 101, [.5]),
    ]
    for label, masses, ignore_eos, old_uniforms, new_uniforms in generation_specs:
        uniforms = old_uniforms if rejection else new_uniforms
        pending = list(uniforms)
        probabilities = []
        tensor = torch.tensor([math.log(value) if value else -float('inf') for value in masses], dtype=torch.float64)
        owner = SimpleNamespace(speech_token_size=4, sampling=namespace['ras_sampling'])
        torch.Tensor.multinomial = multinomial
        failed = False
        token = None
        try:
            token = namespace['sampling_ids'](owner, tensor, [], 25, ignore_eos=ignore_eos)
        except RuntimeError as error:
            assert 'sampling reaches max_trials' in str(error)
            failed = True
        finally:
            torch.Tensor.multinomial = original
        assert not pending, (label, pending)
        generation_cases.append(dict(label=label, masses=masses, ignore_eos=ignore_eos,
            uniforms=uniforms, failed=failed, token=None if failed else int(token.item() if isinstance(token, torch.Tensor) else token),
            post_scores=[None if value == -float('inf') else value for value in tensor.tolist()]))
    sources.append(dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout,
        text=True).strip(), source_sha256=hashlib.sha256(source.encode()).hexdigest(), cases=cases,
        controller_sha256=hashlib.sha256(llm_source.encode()).hexdigest(), eos_rule='rejection' if rejection else 'mask',
        generation=generation_cases))
output = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_sampling_reference.json'
output.write_text(json.dumps(dict(sources=sources), separators=(',', ':'), allow_nan=False), encoding='utf-8')
print(output)
