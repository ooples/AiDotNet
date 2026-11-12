"""Execute the pinned author's actual T3 post-logit inference statements.

Argument: Chatterbox checkout. Uses its exact pinned Transformers dependency.
No neural forward is replaced: this is a post-processing reference complementing
the separately verified complete trained T3 logits and conditioning graph.
"""
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
import transformers
from transformers import MinPLogitsWarper, TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

checkout = Path(sys.argv[1]).resolve()
source = (checkout / 'src/chatterbox/models/t3/t3.py').read_text(encoding='utf-8')
assert f'transformers=={transformers.__version__}' in (checkout / 'pyproject.toml').read_text(encoding='utf-8')
tree = ast.parse(source)
inference = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'inference')
assigns = lambda node, name: isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
loop = next(node for node in ast.walk(inference) if isinstance(node, ast.For)
            and any(assigns(child, 'logits_step') for child in node.body))
start = next(i for i, node in enumerate(loop.body) if assigns(node, 'logits_step'))
end = next(i for i, node in enumerate(loop.body) if assigns(node, 'logits')
           and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'top_p_warper')
constructors = [node for node in inference.body if any(assigns(node, name)
    for name in ('top_p_warper', 'min_p_warper', 'repetition_penalty_processor'))]
probabilities = next(node for node in loop.body if assigns(node, 'probs'))
program = compile(ast.Module(body=constructors + loop.body[start:end+1] + [probabilities], type_ignores=[]),
                  '<original-T3-logit-processing>', 'exec')
base = [1.2, -.7, .1, 2.4, .8, -1.5, 1., .2]
unconditioned = [-.3, .2, .4, .8, -.1, .2, .2, .3]
tie = [0., 0.] + [-10000.] * 6
minimum = [0., math.log(.5)] + [-10000.] * 6
specifications = [
    ('cfg_penalty_and_nucleus', base, unconditioned, .5, .8, 1.2, .05, .95, [6, 3, 3]),
    ('negative_repeated_logits', base, unconditioned, 1.5, 1.3, 2., 0., 1., [6, 0, 2]),
    ('min_p_without_nucleus', base, unconditioned, .25, 1., 1., .2, 1., [6]),
    ('min_p_exact_threshold', minimum, minimum, .5, 1., 1., .5, 1., [6]),
    ('top_p_exact_half', tie, tie, .5, 1., 1., .1, .5, [6]),
    ('uniform_eighth_nucleus', [0.] * 8, [0.] * 8, .5, 1., 1., 0., .125, [6]),
    ('nucleus_with_penalty', base, base, .5, 1.2, 1.5, 0., .8, [6, 0, 3]),
    ('single_dominant_token', [10.] + [0.] * 7, [0.] * 8, .5, .8, 1.2, .1, .8, [6]),
    ('zero_top_p_is_argmax', base, unconditioned, .5, .8, 1.2, 0., 0., [6]),
]
data = dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(), transformers_version=transformers.__version__,
    policy_sha256=hashlib.sha256(Path(inspect.getsourcefile(TopPLogitsWarper)).read_bytes()).hexdigest(), cases=[])
for label, cond, uncond, cfg, temperature, repetition, min_p, top_p, history in specifications:
    namespace = dict(torch=torch, output=SimpleNamespace(logits=torch.tensor([cond, uncond], dtype=torch.float64).reshape(2, 1, 8)),
        generated_ids=torch.tensor([history]), cfg_weight=cfg, temperature=temperature,
        repetition_penalty=repetition, min_p=min_p, top_p=top_p,
        MinPLogitsWarper=MinPLogitsWarper, TopPLogitsWarper=TopPLogitsWarper,
        RepetitionPenaltyLogitsProcessor=RepetitionPenaltyLogitsProcessor)
    exec(program, namespace)
    probs = namespace['probs'][0]
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
    data['cases'].append(dict(label=label, conditional=cond, unconditional=uncond, cfg=cfg,
        temperature=temperature, repetition=repetition, min_p=min_p, top_p=top_p,
        history=history, probabilities=probs.tolist(), draws=draws))
output = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chatterbox_sampling_reference.json'
output.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(output)
