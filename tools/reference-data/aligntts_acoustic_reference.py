"""Independent complete AlignTTS paper acoustic graph and four phase gradients.

Uses explicit enumeration of every positive-duration monotonic alignment, NOT the
native dynamic program or Coqui's modified AlignTTS objective. Widths/block counts
are reduced; all paper branches remain. This fixture selects three density hidden
layers to exercise configurable stacking; the paper does not disclose their count.
"""
import itertools
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from fastspeech2_acoustic_reference import FFT, positions

torch.manual_seed(229405)

class AlignTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.encoder = FFT((3, 3))
        self.decoder = FFT((3, 3))
        self.mel = nn.Linear(8, 5)
        self.duration_embedding = nn.Embedding(12, 8)
        self.duration_encoder = FFT((3, 3))
        self.duration_encoder.conv1 = nn.Conv1d(8, 8, 3, padding=1)
        self.duration_encoder.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.duration_projection = nn.Linear(8, 1)
        self.density = nn.Sequential(*[layer for _ in range(3) for layer in
            (nn.Linear(8, 8), nn.LayerNorm(8), nn.ReLU())], nn.Linear(8, 10))

    def forward(self, tokens, target):
        hidden = self.encoder(positions(self.embedding(tokens)))
        mean, log_variance = self.density(hidden).chunk(2, dim=-1)
        log_prob = -.5 * (math.log(2 * math.pi) + log_variance[:, :, None] +
            (target[:, None] - mean[:, :, None]).square() * torch.exp(-log_variance[:, :, None])).sum(-1)[0]
        paths = []
        durations = []
        for cuts in itertools.combinations(range(1, 6), 3):
            sizes = [right - left for left, right in zip((0, *cuts), (*cuts, 6))]
            alignment = torch.repeat_interleave(torch.arange(4), torch.tensor(sizes))
            paths.append(log_prob[alignment, torch.arange(6)].sum())
            durations.append(sizes)
        path_scores = torch.stack(paths)
        selected = torch.tensor(durations[int(path_scores.detach().argmax())])
        expanded = torch.repeat_interleave(hidden, selected, dim=1)
        mel = self.mel(self.decoder(positions(expanded)))
        log_duration = self.duration_projection(self.duration_encoder(positions(self.duration_embedding(tokens))))
        nll = -torch.logsumexp(path_scores, dim=0)
        mel_loss = F.mse_loss(mel, target)
        duration_loss = F.mse_loss(log_duration.flatten(), torch.log1p(selected.double()))
        return hidden, log_prob, selected, mel, log_duration, (nll, mel_loss, nll + mel_loss, duration_loss)

model = AlignTTS().eval()
tokens = torch.tensor([[2, 5, 7, 9]])
target = torch.randn(1, 6, 5)
flat = lambda value: value.detach().flatten().tolist()
result = dict(source=__doc__, tokens=flat(tokens), target=flat(target),
              weights={k: flat(p) for k, p in model.named_parameters()}, phases=[])
for phase in range(4):
    model.zero_grad(set_to_none=True)
    hidden, likelihood, durations, mel, log_duration, losses = model(tokens, target)
    loss = losses[phase]
    loss.backward()
    result['phases'].append(dict(loss=loss.item(), gradients={k: flat(p.grad) for k, p in model.named_parameters() if p.grad is not None}))
result.update(hidden=flat(hidden), likelihood=flat(likelihood), durations=flat(durations), mel=flat(mel), log_duration=flat(log_duration))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/aligntts_acoustic_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete AlignTTS: {len(result["weights"])} Torch tensors; four enumerated-alignment objectives and gradients.')
