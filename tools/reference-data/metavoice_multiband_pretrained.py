"""Compare unchanged AudioCraft denoisers with each released MBD checkpoint band."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--band', type=int, choices=range(4), required=True)
args = parser.parse_args()

def unused(*args, **kwargs):
    raise AssertionError('Released checkpoint must not enter optional transformer code')

optional = types.ModuleType('audiocraft.modules.transformer')
optional.StreamingTransformer = unused
optional.create_sin_embedding = unused
sys.modules[optional.__name__] = optional
source = args.source / 'audiocraft/models/unet.py'
spec = importlib.util.spec_from_file_location('original_audiocraft_unet', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_num_threads(4)
checkpoint = torch.load(args.checkpoint, mmap=True, weights_only=False, map_location='cpu')
band = checkpoint[args.band]
state = band['model_state']
cfg = dict(band['cfg']['diffusion_unet'])
with torch.device('meta'):
    model = module.DiffusionUnet(chin=1, num_steps=1000, **cfg)
model.load_state_dict(state, strict=True, assign=True)
model.eval()
torch.manual_seed(229466 + args.band)
waveform = torch.randn(1, 1, 1031)
condition = torch.randn(1, 128, 7)
step = 517
with torch.no_grad():
    original = model(waveform, step, condition).sample.detach().flatten().tolist()
args.output.mkdir(parents=True, exist_ok=True)
weights = args.output / ('band-' + str(args.band) + '.safetensors')
save_file(dict(state), str(weights))
model.double()
with torch.no_grad():
    matched = model(waveform.double(), step, condition.double()).sample.detach().flatten().tolist()
result = dict(source='https://github.com/facebookresearch/audiocraft',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    release='facebook/multiband-diffusion@62daf9743d37c54e13366e396241a476908ecad7',
    original_checkpoint=str(args.checkpoint), band=args.band, weights=str(weights),
    configuration=cfg, state_tensor_count=len(state), unchanged_author_forward=True,
    waveform=waveform.flatten().tolist(), condition=condition.flatten().tolist(),
    step=step, original_float32=original, matched_float64=matched)
destination = args.output / ('band-' + str(args.band) + '-reference.json')
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(band=args.band, tensors=len(state), values=sum(x.numel() for x in state.values()),
    reference=str(destination))), flush=True)
