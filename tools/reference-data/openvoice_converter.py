"""Complete unchanged official OpenVoice voice_conversion; explicit replayed posterior noise.

All 16 posterior WaveNet layers and all four four-layer coupling blocks are retained.
Only widths and HiFi-GAN upsampling depth are reduced; both zero_g variants and masked
unequal-length batches execute the original author's forward methods without replacement.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from openvoice.models import SynthesizerTrn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(229461)
tensors, cases = {}, []
for zero in (False, True):
    label = 'v2' if zero else 'v1'
    model = SynthesizerTrn(10, 9, 4, 4, 8, 1, 1, 3, 0., '1', [3], [[1, 3, 5]],
                          [2], 8, [4], n_speakers=0, gin_channels=3, zero_g=zero).double().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith('ref_enc.'):
                continue
            parameter.uniform_(.6, 1.) if 'weight_g' in name else parameter.uniform_(-.12, .12)
    spectrogram = torch.randn(2, 9, 7, requires_grad=True)
    source = torch.randn(2, 3, 1, requires_grad=True)
    target = torch.randn(2, 3, 1, requires_grad=True)
    noise = torch.randn(2, 4, 7, requires_grad=True)
    lengths = torch.tensor([7, 5], dtype=torch.long)
    with patch('torch.randn_like', side_effect=lambda x: noise):
        waveform, mask, (posterior, prior, converted) = model.voice_conversion(
            spectrogram, lengths, source, target, tau=.37)
    outputs = dict(waveform=waveform, posterior=posterior, prior=prior, converted=converted)
    loss = sum(((value * torch.linspace(-.8, .7, value.numel()).reshape_as(value)).sum()
                * (index + 1) / 10) for index, value in enumerate(outputs.values()))
    loss.backward()
    tensors[label + '.mask'] = mask.contiguous()
    for name, tensor in dict(spectrogram=spectrogram, source=source, target=target, noise=noise).items():
        assert tensor.grad is not None, name
        tensors[label + '.' + name] = tensor.detach().contiguous()
        tensors[label + '.' + name + '_gradient'] = tensor.grad.contiguous()
    for name, value in outputs.items():
        tensors[label + '.' + name] = value.detach().contiguous()
    parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith('ref_enc.'):
            continue
        assert parameter.grad is not None, name
        tensors[label + '.weight.' + name] = parameter.detach().contiguous()
        tensors[label + '.gradient.' + name] = parameter.grad.contiguous()
        parameters.append(name)
    cases.append(dict(label=label, zero_condition=zero, parameters=parameters, loss=loss.item()))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
save_file(tensors, str(destination / 'openvoice_converter.safetensors'))
source_file = Path(inspect.getfile(SynthesizerTrn))
metadata = dict(source='https://github.com/myshell-ai/OpenVoice',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(), torch=torch.__version__,
    unchanged_author_forward=True, noise_replayed=True, tau=.37, cases=cases)
(destination / 'openvoice_converter.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(dict(parameter_counts=[len(c['parameters']) for c in cases], cases=len(cases))))
