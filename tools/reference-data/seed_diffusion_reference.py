"""Independent native Seed token diffusion: VP corruption, x0 loss and DDIM.

The Seed report discloses a diffusion transformer but no schedule or parameter
architecture. This explicitly tests native standard choices, not private ByteDance
weights: linear beta VP, clean prediction, deterministic DDIM, sinusoidal position
and time features, and post-norm bidirectional attention with exact GELU FFN.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(229407)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(4, 2, bias=False, batch_first=True)
        self.attention.out_proj = nn.Linear(4, 4)
        self.norm1, self.norm2 = nn.LayerNorm(4), nn.LayerNorm(4)
        self.ff1, self.ff2 = nn.Linear(4, 16), nn.Linear(16, 4)

    def forward(self, value):
        value = self.norm1(value + self.attention(value, value, value, need_weights=False)[0])
        return self.norm2(value + self.ff2(F.gelu(self.ff1(value))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(8, 4)
        self.noisy = nn.Linear(4, 4)
        self.timbre = nn.Linear(3, 4)
        self.blocks = nn.ModuleList([Block()])
        self.output = nn.Linear(4, 4)

    def forward(self, noisy, tokens, timbre, time):
        frequency = 10000 ** (-torch.arange(0, 4, 2) / 4)
        temporal = torch.stack(((time * frequency).sin(), (time * frequency).cos()), dim=-1).flatten()
        angles = torch.arange(noisy.shape[1])[:, None] * frequency
        position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        hidden = self.noisy(noisy) + self.tokens(tokens) + self.timbre(timbre)[:, None] + temporal + position
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)

model = Model().eval()
target = torch.randn(1, 3, 4, requires_grad=True)
noise = torch.randn(1, 3, 4, requires_grad=True)
timbre = torch.randn(1, 3, requires_grad=True)
tokens = torch.tensor([[2, 4, 6]])
alpha = torch.cat((torch.ones(1), (1 - torch.linspace(.01, .1, 8)).cumprod(0)))
flat = lambda value: value.detach().flatten().tolist()
result = dict(source=__doc__, target=flat(target), noise=flat(noise), timbre=flat(timbre), alpha=flat(alpha),
              weights={k: flat(p) for k, p in model.named_parameters()}, cases=[])
for timestep in [1, 3, 8]:
    model.zero_grad(set_to_none=True)
    for value in (target, noise, timbre): value.grad = None
    noisy = alpha[timestep].sqrt() * target + (1 - alpha[timestep]).sqrt() * noise
    predicted = model(noisy, tokens, timbre, timestep / 8)
    loss = F.mse_loss(predicted, target)
    loss.backward()
    result['cases'].append(dict(timestep=timestep, noisy=flat(noisy), predicted=flat(predicted), loss=loss.item(),
        gradients={k: flat(p.grad) for k, p in model.named_parameters()}, target_gradient=flat(target.grad),
        noise_gradient=flat(noise.grad), timbre_gradient=flat(timbre.grad)))
state = noise.detach()
trajectory = []
with torch.no_grad():
    for timestep, next_timestep in [(8, 6), (6, 4), (4, 2), (2, 0)]:
        clean = model(state, tokens, timbre, timestep / 8)
        epsilon = (state - alpha[timestep].sqrt() * clean) / (1 - alpha[timestep]).sqrt()
        state = alpha[next_timestep].sqrt() * clean + (1 - alpha[next_timestep]).sqrt() * epsilon
        trajectory.append(flat(state))
result.update(trajectory=trajectory, sampled=flat(state))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/seed_diffusion_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print('Three noise levels: complete native diffusion predictions, objectives, all gradients and four DDIM steps.')
