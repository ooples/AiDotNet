"""Independent AudioCraft 1.2.0 MBD denoiser outputs and every gradient.

Only the unused optional transformer import is guarded; the author's UNet and
all its neural forwards execute unchanged. Reduced widths retain the released
four-stage, two-residual-block architecture, including the conditioned skip alias.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
source = args.source / 'audiocraft/models/unet.py'

def unused_transformer(*args, **kwargs):
    raise AssertionError('Released MBD denoisers do not use the transformer branch')

optional = types.ModuleType('audiocraft.modules.transformer')
optional.StreamingTransformer = unused_transformer
optional.create_sin_embedding = unused_transformer
sys.modules[optional.__name__] = optional
spec = importlib.util.spec_from_file_location('original_audiocraft_unet', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(229465)
model = module.DiffusionUnet(chin=1, hidden=4, depth=4, growth=2,
    num_steps=32, emb_all_layers=True, codec_dim=3, kernel=8, stride=4,
    norm_groups=4, res_blocks=2, dropout=0., bilstm=False, transformer=False).double().eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if 'norm' in name and name.endswith('weight'):
            parameter.uniform_(.8, 1.2)
        else:
            parameter.uniform_(-.08, .08)

cases = []
for batch, length, step in [(1, 1031, 17), (2, 1027, [3, 23])]:
    model.zero_grad(set_to_none=True)
    waveform = torch.randn(batch, 1, length, requires_grad=True)
    condition = torch.randn(batch, 3, 7, requires_grad=True)
    argument = torch.tensor(step) if isinstance(step, list) else step
    result = model(waveform, argument, condition).sample
    weights = torch.linspace(-.3, .4, result.numel()).reshape_as(result)
    loss = (result * weights).sum() + .13 * result.square().mean()
    loss.backward()
    assert condition.grad.abs().max().item() > 0
    parameters = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        parameters[name] = dict(shape=list(parameter.shape), values=parameter.detach().flatten().tolist(),
            gradient=parameter.grad.flatten().tolist())
    assert len(parameters) == 158
    cases.append(dict(batch=batch, length=length, step=step,
        waveform=waveform.detach().flatten().tolist(), condition=condition.detach().flatten().tolist(),
        output=result.detach().flatten().tolist(), weights=weights.flatten().tolist(), loss=loss.item(),
        waveform_gradient=waveform.grad.flatten().tolist(), condition_gradient=condition.grad.flatten().tolist(),
        parameters=parameters))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_multiband_unet.json'
destination.write_text(json.dumps(dict(source='https://github.com/facebookresearch/audiocraft',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), torch=torch.__version__,
    unchanged_author_forwards=True, unused_transformer_guard=True, cases=cases)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), parameters=158, destination=str(destination))))
