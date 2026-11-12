"""Complete reduced-width FastSpeech 2 paper graph using independent PyTorch ops.

This is an equation reference, not an author checkpoint. It retains all encoder,
variance, length-regulation and decoder branches. CWT targets use the unmodified
author NATSpeech/pycwt routine. Appendix C.2's pooled convolution features and
single statistics projection are intentional; later NATSpeech models differ.
"""
import json
from pathlib import Path
import numpy as np
import torch
from natspeech_cwt import get_cont_lf0, get_lf0_cwt

torch.set_num_threads(1)
torch.manual_seed(229402)
torch.set_default_dtype(torch.float64)

class FFT(torch.nn.Module):
    def __init__(self, kernels=(9, 1)):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(8, 2, bias=False, batch_first=True)
        self.attention.out_proj = torch.nn.Linear(8, 8)
        self.norm1 = torch.nn.LayerNorm(8)
        self.conv1 = torch.nn.Conv1d(8, 16, kernels[0], padding=kernels[0] // 2)
        self.conv2 = torch.nn.Conv1d(16, 8, kernels[1], padding=kernels[1] // 2)
        self.norm2 = torch.nn.LayerNorm(8)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x, need_weights=False)[0])
        return self.norm2(x + self.conv2(self.conv1(x.transpose(1, 2)).relu()).transpose(1, 2))

class Predictor(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(8, 8, 3, padding=1)
        self.norm1 = torch.nn.LayerNorm(8)
        self.conv2 = torch.nn.Conv1d(8, 8, 3, padding=1)
        self.norm2 = torch.nn.LayerNorm(8)
        self.projection = torch.nn.Linear(8, output)

    def forward(self, x):
        x = self.norm1(self.conv1(x.transpose(1, 2)).relu().transpose(1, 2))
        x = self.norm2(self.conv2(x.transpose(1, 2)).relu().transpose(1, 2))
        return self.projection(x), x

def positions(x):
    position = torch.arange(x.shape[1])[:, None]
    angle = position / (10000 ** (torch.arange(0, 8, 2) / 8))
    pe = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(1)
    return x + pe

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 8)
        self.encoder = FFT()
        self.duration = Predictor(1)
        self.pitch = Predictor(10)
        self.statistics = torch.nn.Linear(8, 2)
        self.pitch_embedding = torch.nn.Embedding(256, 8)
        self.energy = Predictor(1)
        self.energy_embedding = torch.nn.Embedding(256, 8)
        self.decoder = FFT()
        self.mel = torch.nn.Linear(8, 5)

    def forward(self, tokens, durations, f0, energy):
        h = self.encoder(positions(self.embedding(tokens)))
        duration, _ = self.duration(h)
        frames = torch.repeat_interleave(h, durations, dim=1)
        pitch, pitch_hidden = self.pitch(frames)
        statistics = self.statistics(pitch_hidden.mean(1))
        pitch_bins = torch.exp(torch.linspace(np.log(71.), np.log(800.), 255))
        frames = frames + self.pitch_embedding(torch.bucketize(f0, pitch_bins))
        predicted_energy, _ = self.energy(frames)
        energy_bins = torch.linspace(0., 1024 * np.sqrt(7 / 16), 255)
        frames = frames + self.energy_embedding(torch.bucketize(energy, energy_bins))
        mel = self.mel(self.decoder(positions(frames)))
        return mel, duration, pitch, statistics, predicted_energy

model = Model().eval()
tokens = torch.tensor([[2, 5, 7, 9]])
durations = torch.tensor([3, 0, 5, 8])
f0 = np.array([120., 125., 0., 0., 142., 160., 170., 165., 155., 149., 145., 152., 166., 180., 190., 200.])
_, lf0 = get_cont_lf0(f0)
mean, std = float(lf0.mean()), float(lf0.std())
cwt, _ = get_lf0_cwt((lf0 - mean) / std)
energy = torch.linspace(10., 400., 16)[None, :]
target = torch.randn(1, 16, 5)
outputs = model(tokens, durations, torch.tensor(np.exp(lf0))[None, :], energy)
losses = [torch.nn.functional.l1_loss(outputs[0], target),
          torch.nn.functional.mse_loss(outputs[1].flatten(), torch.log1p(durations.double())),
          torch.nn.functional.mse_loss(outputs[2][0], torch.tensor(cwt)),
          torch.nn.functional.mse_loss(outputs[3][0], torch.tensor([mean, std])),
          torch.nn.functional.mse_loss(outputs[4].flatten(), energy.flatten())]
loss = sum(losses)
loss.backward()
flat = lambda x: x.detach().flatten().tolist()
result = dict(source=__doc__, tokens=flat(tokens), durations=flat(durations), f0=f0.tolist(), continuous_f0=np.exp(lf0).tolist(), energy=flat(energy),
              target=flat(target), outputs=[flat(x) for x in outputs], losses=[x.item() for x in losses],
              loss=loss.item(), weights={k: flat(p) for k, p in model.named_parameters()},
              gradients={k: flat(p.grad) for k, p in model.named_parameters()})
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/fastspeech2_acoustic_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete FastSpeech 2 graph: {len(result["weights"])} learned tensors, loss={loss.item():.17g}')

class FastSpeech(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 8)
        self.encoder = FFT((3, 3))
        self.duration = Predictor(1)
        self.decoder = FFT((3, 3))
        self.mel = torch.nn.Linear(8, 5)

    def forward(self, tokens, durations):
        h = self.encoder(positions(self.embedding(tokens)))
        duration, _ = self.duration(h)
        expanded = torch.repeat_interleave(h, durations, dim=1)
        return self.mel(self.decoder(positions(expanded))), duration

# The original FastSpeech acoustic student consumes externally distilled mel and
# teacher-attention durations. This comparison does not claim to train the teacher.
torch.manual_seed(229401)
model = FastSpeech().eval()
outputs = model(tokens, durations)
losses = [torch.nn.functional.mse_loss(outputs[0], target),
          torch.nn.functional.mse_loss(outputs[1].flatten(), torch.log1p(durations.double()))]
loss = sum(losses)
loss.backward()
result.update(source='FastSpeech sections 3.1-3.3: independent PyTorch acoustic student, supplied teacher targets',
              outputs=[flat(x) for x in outputs], losses=[x.item() for x in losses], loss=loss.item(),
              weights={k: flat(p) for k, p in model.named_parameters()},
              gradients={k: flat(p.grad) for k, p in model.named_parameters()})
destination.with_name('fastspeech_acoustic_reference.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete FastSpeech student: {len(result["weights"])} learned tensors, loss={loss.item():.17g}')
