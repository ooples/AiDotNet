"""Julius filters, original MBD re_eq, and every audio/reference gradient.

re_eq is compiled directly from its unchanged author AST, avoiding unrelated
codec/solver imports. The original source hash and extraction are recorded.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import julius
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
source = args.source / 'audiocraft/models/multibanddiffusion.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
author_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiBandDiffusion')
method = next(node for node in author_class.body if isinstance(node, ast.FunctionDef) and node.name == 're_eq')
scope = dict(torch=torch, julius=julius)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229468)
cases = []
for count in (8, 32):
    splitter = julius.SplitBands(sample_rate=24000, n_bands=count)
    waveform = torch.randn(2, 2, 37, requires_grad=True)
    reference = torch.randn_like(waveform, requires_grad=True)
    bands = splitter(waveform)
    equalized = scope['re_eq'](SimpleNamespace(codec_model=SimpleNamespace(sample_rate=24000)), waveform, reference,
        n_bands=count, strictness=.8)
    weights = torch.linspace(-.2, .3, equalized.numel()).reshape_as(equalized)
    band_weights = torch.linspace(-.1, .2, bands.numel()).reshape_as(bands)
    loss = (equalized * weights).sum() + .13 * (bands * band_weights).sum()
    loss.backward()
    cases.append(dict(bands=count, batch=2, channels=2, time=37,
        filters=splitter.lowpass.filters.flatten().tolist(), filter_shape=list(splitter.lowpass.filters.shape),
        waveform=waveform.detach().flatten().tolist(), reference=reference.detach().flatten().tolist(),
        split=bands.detach().flatten().tolist(), equalized=equalized.detach().flatten().tolist(),
        weights=weights.flatten().tolist(), band_weights=band_weights.flatten().tolist(), loss=loss.item(),
        waveform_gradient=waveform.grad.flatten().tolist(), reference_gradient=reference.grad.flatten().tolist()))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_band_filters.json'
destination.write_text(json.dumps(dict(source='https://github.com/facebookresearch/audiocraft',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), unchanged_author_method_ast=True,
    julius='0.2.7', matched_float64_filter_construction=True, cases=cases)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), filter_shapes=[x['filter_shape'] for x in cases])))
