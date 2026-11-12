"""Execute the author's nested editing/TTS samplers, without importing the model.

Argument: pinned VoiceCraft checkout. Only categorical RNG is replaced with fixed
uniform quantiles; original logits filtering, repetition penalties, delay masking,
termination and codebook-flush control flow execute unchanged from the author AST.
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
import torch.nn.functional as F

checkout = Path(sys.argv[1]).resolve()
source_path = checkout / 'models/voicecraft.py'
source = source_path.read_text()
tree = ast.parse(source)
samplers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'sample_helper']
filters = [node for node in tree.body if isinstance(node, ast.FunctionDef)
           and node.name in ('topk_sampling', 'top_k_top_p_filtering')]
flat = lambda tensor: tensor.detach().flatten().tolist()
data = dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout,
    text=True).strip(), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    sampling_note='Canonical descending-probability inverse-CDF quantiles replace only categorical RNG.', cases=[])

specifications = [
    ('positive_silence', False, {1: 5., 2: 2.}, 8, 3, 0, 1., 0.),
    ('negative_silence', False, {1: -.5, 2: -2.}, 9, 3, 0, 1., 0.),
    ('argmax_eog', False, {0: .5, 9: 1.}, 4, 0, 0, 1., .9),
    ('tts_eos_and_minimum', True, {1: 0., 11: 10.}, 14, 0, 0, 1., 0.),
    ('top_k_threshold_ties', False, {1: 3., 2: 3.}, 3, 0, 1, 1., .75),
    ('top_p_exact_boundary', False, {1: 3., 2: 3.}, 3, 0, 0, .5, .75),
]
original_multinomial = torch.multinomial
for label, tts, first_logits, maximum, repetition, top_k, top_p, uniform in specifications:
    environment = dict(torch=torch, F=F,
        self=SimpleNamespace(args=SimpleNamespace(n_codebooks=4, eog=9, eos=11,
            empty_token=8, encodec_sr=50)), eog_inference=11)
    exec(compile(ast.Module(body=filters + [samplers[1 if tts else 0]], type_ignores=[]),
                 str(source_path), 'exec'), environment)
    logits = torch.full((4, 12), -10000., dtype=torch.float64)
    for token, value in first_logits.items():
        logits[0, token] = value
    logits[1:, 3] = 1.
    rows, snapshots = [], []
    flags = [False] * 4
    previous, consecutive = -1, 0
    # A real growing y_input reaches the author's text-proportional budget at
    # exactly the requested maximum. No sampler stopping branch is substituted.
    phones = math.ceil(maximum / 10)
    prefix_length = phones * 10 - maximum + 1
    environment['x_lens'] = torch.tensor([phones])
    recorded_probabilities = []
    def inverse_cdf(probabilities, num_samples=1, **kwargs):
        assert num_samples == 1
        recorded_probabilities.append(flat(probabilities))
        result = []
        for distribution in probabilities:
            cumulative = 0.
            for token in torch.argsort(distribution, descending=True, stable=True).tolist():
                cumulative += distribution[token].item()
                if uniform <= cumulative:
                    result.append([token])
                    break
        return torch.tensor(result, dtype=torch.long)
    torch.multinomial = inverse_cdf
    try:
        for step in range(maximum + 4):
            environment['y_input'] = torch.zeros(1, prefix_length + step, 4)
            sampled, flags, previous, consecutive = environment['sample_helper'](
                sum(flags), logits.clone(), flags, top_k, top_p, 1., previous,
                consecutive, repetition, [1], step)
            rows.append(sampled[:, 0].tolist())
            snapshots.append(dict(previous=int(previous), consecutive=consecutive, ended=list(flags)))
            if all(flags):
                break
    finally:
        torch.multinomial = original_multinomial
    end_token = 11 if tts else 9
    frames = next(i for i, row in enumerate(rows) if row[0] == end_token)
    codes = [[rows[frame + q][q] for frame in range(frames)] for q in range(4)]
    data['cases'].append(dict(label=label, tts=tts, logits=flat(logits), maximum=maximum,
        minimum=10 if tts else 0, repetition=repetition, top_k=top_k, top_p=top_p,
        uniform=uniform, rows=rows, frames=frames, codes=codes, snapshots=snapshots,
        probabilities=recorded_probabilities))
output = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/voicecraft_generation_reference.json'
output.write_text(json.dumps(data, separators=(',', ':')))
print(output)
