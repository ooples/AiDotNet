"""Execute author candidate-tail repair and extract its actual silence-cut loop."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import transformers
from transformers import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper

checkout = Path(sys.argv[1]).resolve()
source = (checkout / 'tortoise/api.py').read_text()
tree = ast.parse(source)
tts = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'tts')
api_defaults = dict(zip([arg.arg for arg in tts.args.args[-len(tts.args.defaults):]], tts.args.defaults))
default_maximum = ast.literal_eval(api_defaults['max_mel_tokens'])
repair = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'fix_autoregressive_output')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[repair], type_ignores=[]), '<original-tail-repair>', 'exec'), namespace)
# Extract the actual author loop, preserving its test, increment and [:k] slice.
loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
            and ast.unparse(node.iter) == 'range(codes.shape[-1])'
            and any(isinstance(child, ast.Name) and child.id == 'ctokens' for child in ast.walk(node)))
cut = ast.parse('def cut(codes):\n ctokens=0\n calm_token=83\n latents=torch.arange(codes.shape[-1]).reshape(1,-1)\n return latents.shape[-1]')
cut.body[0].body.insert(-1, loop)
exec(compile(ast.fix_missing_locations(cut), '<original-silence-cut>', 'exec'), namespace)
cases = []
streams = [([12, 13, 8193], 20), ([12, 13, 14], 20),
           ([12, 13, 14, 15, 16, 17], 6), ([8193], 12),
           ([12, 8193, 14, 8193], 16),
           ([12] + [83] * 8 + [14, 15], 11),
           ([12] + [83] * 9 + [14, 15], 12),
           ([12] + [83] * 5 + [14] + [83] * 9 + [15], 17)]
for raw, maximum in streams:
    padded = raw + [8193] * (maximum - len(raw))
    fixed = namespace['fix_autoregressive_output'](torch.tensor(padded), 8193, complain=False)
    cases.append(dict(raw=raw, maximum=maximum, repaired=fixed.tolist(),
                      latent_length=namespace['cut'](fixed.reshape(1, -1))))
output = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_candidate_reference.json'
sampling = []
for previous, minimum, top_p, temperature, scale in [([3, 4], 0, .8, .8, 1.),
        ([], 2, .95, .8, 1.), ([1, 4], 0, 1., .8, 1.), ([], 0, .95, 1e-8, 1e-8)]:
    logits = torch.tensor([[.5, 3., 1.1, -.2, 2.2, .7, 2.7, 1.3]], dtype=torch.float64) * scale
    prefix = torch.tensor([[1, 1, 1, 6] + previous])
    scores = RepetitionPenaltyLogitsProcessor(2.)(prefix, logits.clone())
    if len(previous) < minimum:
        scores[0, 7] = -float('inf')
    scores = TemperatureLogitsWarper(temperature)(prefix, scores)
    scores = TopPLogitsWarper(top_p)(prefix, scores)
    probabilities = scores.softmax(-1)[0]
    # Compare the distribution with fixed uniform quantiles, not Torch's different
    # categorical RNG implementation. The native sampler orders by descending mass.
    order = torch.argsort(probabilities, descending=True, stable=True)
    draws = []
    for draw in (0., .1, .3, .5, .7, .9, .999999):
        cumulative = 0.
        for token in order.tolist():
            cumulative += probabilities[token].item()
            if draw <= cumulative:
                draws.append(dict(uniform=draw, token=token))
                break
    sampling.append(dict(previous=previous, minimum=minimum, top_p=top_p, temperature=temperature,
        logits=logits.flatten().tolist(), probabilities=probabilities.tolist(), draws=draws))
output.write_text(json.dumps(dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
    cwd=checkout, text=True).strip(), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    cases=cases, sampling=sampling, default_maximum=default_maximum,
    transformers_version=transformers.__version__), separators=(',', ':')))
print(output)
