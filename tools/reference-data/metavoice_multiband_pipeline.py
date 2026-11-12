"""Entire released AudioCraft MBD decoder, original controllers/NN forwards and every Gaussian draw.

Author classes are compiled from unchanged ASTs to avoid unrelated training/solver
imports. Optional transformer symbols are raising guards because the release does
not use them. Transformers is the exact MetaVoice-pinned 4.39.2 dependency.
"""
import argparse
import ast
from abc import ABC, abstractmethod
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import types
import typing as tp
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--pinned-python', type=Path, required=True)
parser.add_argument('--multiband-release', type=Path, required=True)
parser.add_argument('--encodec-release', type=Path, required=True)
parser.add_argument('--codes-reference', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.pinned_python))
import numpy as np
import torch
import julius
import transformers
from transformers import EncodecConfig, EncodecModel
from safetensors.torch import load_file

assert transformers.__version__ == '4.39.2'
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

def unused(*args, **kwargs):
    raise AssertionError('Released MBD must not call optional transformer code')

optional = types.ModuleType('audiocraft.modules.transformer')
optional.StreamingTransformer = unused
optional.create_sin_embedding = unused
sys.modules[optional.__name__] = optional

def original_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, args.source / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

unet = original_module('audiocraft/models/unet.py', 'original_mbd_unet')
schedule = original_module('audiocraft/modules/diffusion_schedule.py', 'original_mbd_schedule')
scope = dict(torch=torch, julius=julius, tp=tp, math=math, np=np, nn=torch.nn,
    ABC=ABC, abstractmethod=abstractmethod, Path=Path, HFEncodecModel=EncodecModel,
    qt=types.SimpleNamespace(QuantizedResult=object), logger=logging.getLogger(__name__),
    DiffusionUnet=unet.DiffusionUnet, NoiseSchedule=schedule.NoiseSchedule)

def original_classes(relative, names):
    source = args.source / relative
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)

original_classes('audiocraft/models/encodec.py', ['CompressionModel', 'HFEncodecCompressionModel'])
original_classes('audiocraft/models/multibanddiffusion.py', ['DiffusionProcess', 'MultiBandDiffusion'])
codec_release = json.loads(args.encodec_release.read_text())
config = EncodecConfig.from_json_file(codec_release['paths']['config.json'])
codec = EncodecModel(config).eval()
codec.load_state_dict(load_file(codec_release['paths']['model.safetensors']), strict=True)
codec.double()
wrapper = scope['HFEncodecCompressionModel'](codec)
wrapper.set_num_codebooks(8)
release = json.loads(args.multiband_release.read_text())
checkpoint = torch.load(release['checkpoint_path'], mmap=True, weights_only=False, map_location='cpu')
processes = []
for band in range(4):
    saved = checkpoint[band]
    cfg = saved['cfg']
    with torch.device('meta'):
        model = unet.DiffusionUnet(chin=1, num_steps=1000, **dict(cfg['diffusion_unet']))
    model.load_state_dict(saved['model_state'], strict=True, assign=True)
    model.double().eval()
    if cfg['processor']['use']:
        processor = schedule.MultiBandProcessor(sample_rate=24000, n_bands=8,
            power_std=float(cfg['processor']['power_std']))
        processor.load_state_dict(saved['processor_state'], strict=True)
        processor.double()
    else:
        processor = schedule.SampleProcessor()
    controller = schedule.NoiseSchedule(device='cpu', sample_processor=processor, **dict(cfg['schedule']))
    # Matched-float64 schedule; coefficients retain the exact original float32 values.
    controller.betas = controller.betas.double()
    processes.append(scope['DiffusionProcess'](model, controller))
    print('Loaded original band', band, flush=True)
pipeline = scope['MultiBandDiffusion'](processes, wrapper)
codes = torch.tensor([[[0, 17, 1023, 43], [1, 18, 1022, 44], [2, 19, 1021, 45], [3, 20, 1020, 46],
    [4, 21, 1019, 47], [5, 22, 1018, 48], [6, 23, 1017, 49], [7, 24, 1016, 50]]])
if args.codes_reference:
    codes = torch.tensor(json.loads(args.codes_reference.read_text())['final_codes'])[None]
draws = []
original_randn = torch.randn_like
def gaussian(value):
    result = original_randn(value)
    draws.append(result.flatten().tolist())
    return result
torch.manual_seed(229474)
with torch.no_grad(), patch('torch.randn_like', side_effect=gaussian):
    waveform = pipeline.tokens_to_wav(codes)
assert len(draws) == 80, len(draws)  # Four initial draws plus 19 per-step draws per band.
result = dict(source='https://github.com/facebookresearch/audiocraft',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    release=release, codec_release=codec_release, transformers=transformers.__version__,
    unchanged_author_classes_ast=True, matched_float64_schedule_and_filter_construction=True,
    source_sha256={name: hashlib.sha256((args.source / name).read_bytes()).hexdigest() for name in
        ['audiocraft/models/unet.py', 'audiocraft/models/multibanddiffusion.py', 'audiocraft/modules/diffusion_schedule.py', 'audiocraft/models/encodec.py']},
    codes=codes[0].tolist(), gaussian_draws=draws, waveform=waveform.flatten().tolist(), shape=list(waveform.shape))
destination = args.output or args.multiband_release.parent / 'pipeline-reference.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(reference=str(destination), samples=waveform.numel(), draws=len(draws))), flush=True)
