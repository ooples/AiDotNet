"""OpenVoice paper section 2.3 direct posterior decoder and separate flow KL.

The released author's posterior, flow and decoder execute unchanged, including
all sixteen posterior WaveNet layers and four four-layer mean-only flow blocks.
This component reference does not stand in for the missing complete GAN trainer.
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
sys.path.insert(0, str(args.source))
from openvoice.models import SynthesizerTrn
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229480)
tensors, cases = {}, []
for zero in (False, True):
    label = 'v2' if zero else 'v1'
    model = SynthesizerTrn(10, 9, 4, 4, 8, 1, 1, 3, 0., '1', [3], [[1, 3, 5]],
                          [2], 8, [4], n_speakers=0, gin_channels=3, zero_g=zero).double().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith('ref_enc.'):
                parameter.uniform_(.6, 1.) if 'weight_g' in name else parameter.uniform_(-.12, .12)
    inputs = dict(spectrogram=torch.randn(2, 9, 7, requires_grad=True),
                  speaker=torch.randn(2, 3, 1, requires_grad=True),
                  noise=torch.randn(2, 4, 7, requires_grad=True),
                  prior_mean=torch.randn(2, 4, 7, requires_grad=True),
                  prior_log_scale=(torch.randn(2, 4, 7) * .1).requires_grad_())
    condition = torch.zeros_like(inputs['speaker']) if zero else inputs['speaker']
    with patch('torch.randn_like', side_effect=lambda x: inputs['noise']):
        posterior, mean, logs, mask = model.enc_q(inputs['spectrogram'], torch.tensor([7, 5]), g=condition, tau=1.)
    prior = model.flow(posterior, mask, g=inputs['speaker'])
    # Reconstruction bypasses both flow directions. Flow contributes to the KL branch.
    waveform = model.dec(posterior[:, :, 1:4], g=condition)
    difference = prior - inputs['prior_mean']
    kl = ((inputs['prior_log_scale'] - logs - .5 +
           .5 * difference.square() * torch.exp(-2 * inputs['prior_log_scale'])) * mask).sum() / mask.sum()
    probe = torch.linspace(-.31, .27, waveform.numel()).reshape_as(waveform)
    loss = .2 * (waveform * probe).sum() + .7 * kl
    loss.backward()
    outputs = dict(waveform=waveform, posterior=posterior, prior=prior, mean=mean, log_scale=logs, kl=kl.reshape(1))
    tensors[label + '.mask'] = mask.detach().contiguous()
    tensors[label + '.wave_probe'] = probe.contiguous()
    for name, value in inputs.items():
        tensors[label + '.' + name] = value.detach().contiguous()
        assert value.grad is not None, name
        tensors[label + '.' + name + '_gradient'] = value.grad.contiguous()
    for name, value in outputs.items(): tensors[label + '.' + name] = value.detach().contiguous()
    parameters = []
    for name, value in model.named_parameters():
        if name.startswith('ref_enc.'): continue
        assert value.grad is not None, name
        tensors[label + '.weight.' + name] = value.detach().contiguous()
        tensors[label + '.gradient.' + name] = value.grad.contiguous()
        parameters.append(name)
    cases.append(dict(label=label, zero_condition=zero, parameters=parameters, loss=loss.item()))
directory = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
save_file(tensors, str(directory / 'openvoice_training.safetensors'))
source = Path(inspect.getfile(SynthesizerTrn))
(directory / 'openvoice_training.json').write_text(json.dumps(dict(
    source='https://github.com/myshell-ai/OpenVoice',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), torch=torch.__version__,
    unchanged_author_neural_forwards=True, gaussian_kl_equation_reference=True,
    segment_start=1, segment_frames=3, cases=cases), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), parameters=[len(case['parameters']) for case in cases])))
