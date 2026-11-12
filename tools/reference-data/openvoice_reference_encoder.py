"""Run the unchanged official OpenVoice ReferenceEncoder; export every weight and gradient.

Usage: python openvoice_reference_encoder.py --source <myshell-ai/OpenVoice checkout>
The architecture retains all six convolution widths and the original 128-wide GRU.
Odd time/frequency sizes and three recurrent steps detect flattening/reset/bias mistakes.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from openvoice.models import ReferenceEncoder

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(229460)
root = Path(__file__).resolve().parents[2]
destination = root / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
tensors = {}
cases = []
for normalized in (True, False):
    label = 'normalized' if normalized else 'plain'
    model = ReferenceEncoder(17, 7, layernorm=normalized).double().eval()
    # Distinct, nonzero recurrent biases are important: reset-after must multiply bias_hh_n.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'layernorm.weight' in name:
                parameter.uniform_(.8, 1.2)
            elif 'weight_g' in name:
                parameter.uniform_(.5, 1.1)
            else:
                parameter.uniform_(-.08, .08)
    x = torch.randn(2, 129, 17, requires_grad=True)
    y = model(x)
    objective_weights = torch.linspace(-.7, .9, y.numel()).reshape_as(y)
    loss = (y * objective_weights).sum()
    loss.backward()
    tensors[label + '.input'] = x.detach().contiguous()
    tensors[label + '.output'] = y.detach().contiguous()
    tensors[label + '.input_gradient'] = x.grad.contiguous()
    tensors[label + '.objective_weights'] = objective_weights.contiguous()
    parameters = []
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        tensors[label + '.weight.' + name] = parameter.detach().contiguous()
        tensors[label + '.gradient.' + name] = parameter.grad.contiguous()
        parameters.append(name)
    cases.append(dict(label=label, layer_normalization=normalized, parameters=parameters,
                      loss=loss.item()))
save_file(tensors, str(destination / 'openvoice_reference_encoder.safetensors'))
source_file = Path(inspect.getfile(ReferenceEncoder))
metadata = dict(source='https://github.com/myshell-ai/OpenVoice',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_file='openvoice/models.py', source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    torch=torch.__version__, frequencies=17, speaker_channels=7, batch=2, time=129,
    unchanged_author_forward=True, cases=cases)
(destination / 'openvoice_reference_encoder.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
