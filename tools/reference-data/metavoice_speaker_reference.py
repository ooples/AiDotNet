"""Original MetaVoice speaker LSTM: every parameter gradient and partial-window boundaries."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
module = importlib.import_module('fam.quantiser.audio.speaker_encoder.model')
module.mel_n_channels = 5
module.model_hidden_size = 8
module.model_embedding_size = 6
module.model_num_layers = 3
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229472)
lstm = torch.nn.LSTM(5, 8, 3, batch_first=True)
linear = torch.nn.Linear(8, 6)
with torch.no_grad():
    linear.bias.fill_(.2)  # Keep every projected coordinate differentiable and nonzero.
state = {'lstm.' + name: value for name, value in lstm.state_dict().items()}
state.update({'linear.' + name: value for name, value in linear.state_dict().items()})
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / 'speaker.pt'
    torch.save(dict(model_state=state), checkpoint)
    model = module.SpeakerEncoder(str(checkpoint), device='cpu', verbose=False, eval=True)
cases = []
for batch, frames in [(1, 7), (2, 9)]:
    model.zero_grad(set_to_none=True)
    mels = torch.randn(batch, frames, 5, requires_grad=True)
    output = model(mels)
    weights = torch.linspace(-.3, .4, output.numel()).reshape_as(output)
    loss = (output * weights).sum()
    loss.backward()
    parameters = {name: dict(shape=list(value.shape), values=value.detach().flatten().tolist(),
        gradient=value.grad.flatten().tolist()) for name, value in model.named_parameters()}
    assert len(parameters) == 14
    cases.append(dict(batch=batch, frames=frames, mels=mels.detach().flatten().tolist(),
        output=output.detach().flatten().tolist(), weights=weights.flatten().tolist(),
        loss=loss.item(), gradient=mels.grad.flatten().tolist(), parameters=parameters))
boundaries = []
for samples in [1, 159, 160, 25599, 25600, 25601, 37919, 37920, 45000, 60000, 100000]:
    waveform, mel = model.compute_partial_slices(samples, 1.3, .75)
    boundaries.append(dict(samples=samples, starts=[int(x.start) for x in mel], stops=[int(x.stop) for x in waveform]))
frontends = []
for samples in (641, 37920):
    waveform = (.1 * np.sin(np.arange(samples) * 2 * np.pi * 220 / 16000)).astype(np.float32)
    wav_slices, mel_slices = model.compute_partial_slices(samples, 1.3, .75)
    padded = np.pad(waveform, (0, max(0, wav_slices[-1].stop - samples)))
    mel = module.audio.wav_to_mel_spectrogram(padded)
    partials = np.array([mel[item] for item in mel_slices])
    frontends.append(dict(audio=waveform.tolist(), shape=list(partials.shape), partials=partials.flatten().tolist()))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_speaker_reference.json'
destination.write_text(json.dumps(dict(source='https://github.com/metavoiceio/metavoice-src',
    revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    unchanged_author_forwards=True, reduced_width=True, cases=cases, boundaries=boundaries, frontends=frontends)) + '\n', encoding='utf-8')
print(json.dumps(dict(cases=len(cases), parameters=14, boundaries=len(boundaries))))
