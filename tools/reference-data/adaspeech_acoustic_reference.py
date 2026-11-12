"""AdaSpeech complete reduced-width paper graph and joint-training gradients.

Equation reference; no original Microsoft checkpoint is claimed. Reuses the
independent PyTorch FastSpeech 2 backbone, with paper Eq. 1 conditional norms,
reference acoustic encoders, and detached phoneme labels. Native projection from
the four-dimensional phoneme vector to the hidden width is explicitly exposed.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fastspeech2_acoustic_reference import Model, FFT, positions
from natspeech_cwt import get_cont_lf0, get_lf0_cwt

torch.manual_seed(229404)

class ConditionalNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Linear(8, 8, bias=False)
        self.shift = nn.Linear(8, 8, bias=False)

    def forward(self, value, speaker):
        return F.layer_norm(value, (8,), eps=1e-5) * self.scale(speaker)[:, None] + self.shift(speaker)[:, None]

class ConditionalFFT(FFT):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = ConditionalNorm(), ConditionalNorm()

    def forward(self, value, speaker):
        value = self.norm1(value + self.attention(value, value, value, need_weights=False)[0], speaker)
        return self.norm2(value + self.conv2(self.conv1(value.transpose(1, 2)).relu()).transpose(1, 2), speaker)

class AcousticEncoder(nn.Module):
    def __init__(self, inputs, kernel, stride, output):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, 8, kernel, stride=stride, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(8)
        self.conv2 = nn.Conv1d(8, 8, kernel, stride=stride, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(8)
        self.projection = nn.Linear(8, output) if output else None

    def forward(self, value):
        value = self.norm1(self.conv1(value.transpose(1, 2)).relu().transpose(1, 2))
        value = self.norm2(self.conv2(value.transpose(1, 2)).relu().transpose(1, 2))
        return self.projection(value) if self.projection is not None else value.mean(1)

class AdaSpeech(Model):
    def __init__(self):
        super().__init__()
        self.decoder = ConditionalFFT()
        self.final_norm = ConditionalNorm()
        self.speaker = nn.Embedding(3, 8)
        self.utterance = AcousticEncoder(5, 5, 3, 0)
        self.phoneme = AcousticEncoder(5, 3, 1, 4)
        self.phoneme_predictor = AcousticEncoder(8, 3, 1, 4)
        self.phoneme_projection = nn.Linear(4, 8)

    def forward(self, tokens, durations, f0, energy, mel):
        hidden = self.encoder(positions(self.embedding(tokens)))
        speaker = self.speaker(torch.tensor([1]))
        utterance = self.utterance(mel)
        split = mel.split(durations.tolist(), dim=1)
        phoneme_mel = torch.stack([value.mean(1) if value.shape[1] else value.sum(1) for value in split], dim=1)
        phoneme = self.phoneme(phoneme_mel)
        predicted_phoneme = self.phoneme_predictor(hidden)
        conditioned = hidden + speaker[:, None] + utterance[:, None] + self.phoneme_projection(phoneme)
        duration, _ = self.duration(conditioned)
        frames = torch.repeat_interleave(conditioned, durations, dim=1)
        pitch, pitch_hidden = self.pitch(frames)
        statistics = self.statistics(pitch_hidden.mean(1))
        pitch_bins = torch.exp(torch.linspace(np.log(71.), np.log(800.), 255))
        frames = frames + self.pitch_embedding(torch.bucketize(f0, pitch_bins))
        energy_prediction, _ = self.energy(frames)
        energy_bins = torch.linspace(0., 1024 * np.sqrt(7 / 16), 255)
        frames = frames + self.energy_embedding(torch.bucketize(energy, energy_bins))
        predicted = self.mel(self.final_norm(self.decoder(positions(frames), speaker), speaker))
        return (predicted, duration, pitch, statistics, energy_prediction, phoneme, utterance, conditioned, predicted_phoneme)

root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
result = json.loads((root / 'fastspeech2_acoustic_reference.json').read_text(encoding='utf-8'))
tokens = torch.tensor([result['tokens']], dtype=torch.long)
durations = torch.tensor(result['durations'], dtype=torch.long)
target = torch.tensor(result['target']).reshape(1, 16, 5)
energy = torch.tensor([result['energy']])
_, lf0 = get_cont_lf0(np.array(result['f0']))
mean, std = float(lf0.mean()), float(lf0.std())
cwt, _ = get_lf0_cwt((lf0 - mean) / std)
model = AdaSpeech().eval()
outputs = model(tokens, durations, torch.tensor([result['continuous_f0']]), energy, target)
losses = [F.l1_loss(outputs[0], target), F.mse_loss(outputs[1].flatten(), torch.log1p(durations.double())),
          F.mse_loss(outputs[2][0], torch.tensor(cwt)), F.mse_loss(outputs[3][0], torch.tensor([mean, std])),
          F.mse_loss(outputs[4].flatten(), energy.flatten()), F.mse_loss(outputs[8], outputs[5].detach())]
loss = sum(losses)
loss.backward()
flat = lambda value: value.detach().flatten().tolist()
result.update(source=__doc__, outputs=[flat(x) for x in outputs], losses=[x.item() for x in losses], loss=loss.item(),
    weights={k: flat(p) for k, p in model.named_parameters()}, gradients={k: flat(p.grad) for k, p in model.named_parameters()})
(root / 'adaspeech_acoustic_reference.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete AdaSpeech joint graph: {len(result["weights"])} tensors, loss={loss.item():.17g}')
