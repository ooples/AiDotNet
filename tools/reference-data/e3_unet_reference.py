"""Independent PyTorch E3 waveform U-Net equation reference (not private weights).

All ten resolutions, published strides/kernels/repetitions/experts and attention
positions are retained with width 4. Figure 2 connects adaptive kernels to BOTH
convolutions. Undisclosed implementation choices are explicit: shared conditioning
mixture coefficients, nearest upsampling, affine GN, additive skips, pre-LN
attention, and sinusoidal timestep embedding. Text memory is supplied separately
from the independently implemented/frozen BERT; no private checkpoint is claimed.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(229403)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
STRIDES = [2, 2, 2, 2, 4, 4, 2, 2, 2, 2]
KERNELS = [5] * 5 + [3] * 5
EXPERTS = [8, 8, 4, 4, 2] + [0] * 5
REPEATS = [2] * 5 + [1] * 5

class Attention(nn.Module):
    def __init__(self, width, memory):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(memory, width)
        self.v = nn.Linear(memory, width)
        self.out = nn.Linear(width, width)

    def forward(self, x, memory=None):
        query = self.norm(x)
        memory = query if memory is None else memory
        return x + self.out(F.scaled_dot_product_attention(self.q(query)[:, None],
            self.k(memory)[:, None], self.v(memory)[:, None])[:, 0])

class Block(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride, experts, attention, up=False):
        super().__init__()
        self.stride, self.up = stride, up
        self.first = nn.ModuleList([nn.Conv1d(inputs, outputs, kernel, stride=1 if up else stride,
            padding=kernel // 2) for _ in range(max(1, experts))])
        self.norm1 = nn.GroupNorm(2 if outputs % 2 == 0 else 1, outputs)
        self.second = nn.ModuleList([nn.Conv1d(outputs, outputs, kernel, padding=kernel // 2)
            for _ in range(experts)]) if experts else nn.Conv1d(outputs, outputs, kernel, padding=kernel // 2)
        self.norm2 = nn.GroupNorm(2 if outputs % 2 == 0 else 1, outputs)
        self.shortcut = nn.Conv1d(inputs, outputs, 1, stride=1 if up else stride) if inputs != outputs or stride != 1 else None
        self.adaptive_logits = nn.Linear(4, experts) if experts else None
        self.film = None if experts else nn.Linear(4, 4 * outputs)
        if attention:
            self.add_module('self', Attention(outputs, outputs))
            self.cross = Attention(outputs, 4)

    def forward(self, x, condition, memory):
        if self.up and self.stride != 1:
            x = F.interpolate(x, scale_factor=self.stride, mode='nearest')
        mixture = self.adaptive_logits(condition).softmax(-1) if self.adaptive_logits is not None else None
        def conv(bank, value):
            if mixture is None:
                return bank[0](value) if isinstance(bank, nn.ModuleList) else bank(value)
            return sum(layer(value) * mixture[:, i, None, None] for i, layer in enumerate(bank))
        film = self.film(condition).chunk(4, dim=-1) if self.film is not None else None
        def normalize(value, norm, pair):
            value = norm(value)
            if film is not None:
                value = value * (1 + film[2 * pair + 1][:, :, None]) + film[2 * pair][:, :, None]
            return F.silu(value)
        h = normalize(conv(self.first, x), self.norm1, 0)
        h = normalize(conv(self.second, h), self.norm2, 1)
        h = h + (x if self.shortcut is None else self.shortcut(x))
        if hasattr(self, 'cross'):
            rows = self._modules['self'](h.transpose(1, 2))
            h = self.cross(rows, memory).transpose(1, 2)
        return h

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(8, 4), nn.SiLU(), nn.Linear(4, 4))
        self.speaker_embedding = nn.Embedding(3, 4)
        self.variance = nn.Linear(4, 1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        channels = 1
        for i in range(10):
            block = Block(channels, 4, KERNELS[i], STRIDES[i], EXPERTS[i], i >= 5)
            block.residual = nn.ModuleList([Block(4, 4, KERNELS[i], 1, EXPERTS[i], i >= 5)
                for _ in range(REPEATS[i] - 1)])
            self.down.append(block)
            channels = 4
        for i in reversed(range(10)):
            output = 1 if i == 0 else 4
            block = Block(4, output, KERNELS[i], STRIDES[i], EXPERTS[i], i >= 5, up=True)
            block.residual = nn.ModuleList([Block(4, 4, KERNELS[i], 1, EXPERTS[i], i >= 5)
                for _ in range(REPEATS[i] - 1)])
            self.up.append(block)
        self.output = nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, waveform, memory, level, speaker):
        frequency = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(4) / 3)
        time = self.time(torch.cat(((level * frequency).sin(), (level * frequency).cos()))[None])
        condition = time + self.speaker_embedding(torch.tensor([speaker]))
        length = waveform.numel()
        x = F.pad(waveform, (0, (-length) % 4096))[None, None]
        skips = []
        for block in self.down:
            skips.append(x)
            x = block(x, condition, memory)
            for residual in block.residual:
                x = residual(x, condition, memory)
        for block, skip in zip(self.up, reversed(skips)):
            for residual in block.residual:
                x = residual(x, condition, memory)
            x = block(x, condition, memory) + skip
        return self.output(x).flatten()[:length], self.variance(time)

model = Unet().eval()
waveform = torch.randn(4097, requires_grad=True)
text = torch.randn(1, 3, 4, requires_grad=True)
noise = torch.randn_like(waveform)
weights = torch.ones_like(waveform)
weights[3001:] = .1
epsilon, log_variance = model(waveform, text, .63, 1)
loss = torch.exp(-log_variance) * ((epsilon - noise).square() * weights).sum() / weights.sum() + log_variance
loss.sum().backward()
flat = lambda value: value.detach().flatten().tolist()
result = dict(source=__doc__, waveform=flat(waveform), text=flat(text), noise=flat(noise), weights=flat(weights),
    epsilon=flat(epsilon), log_variance=flat(log_variance), loss=loss.item(), waveform_gradient=flat(waveform.grad),
    text_gradient=flat(text.grad), parameters={k: dict(shape=list(p.shape), values=flat(p), gradient=flat(p.grad))
        for k, p in model.named_parameters()})
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/e3_unet_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'All ten E3 resolutions: {len(result["parameters"])} tensors, loss={loss.item():.17g}')
