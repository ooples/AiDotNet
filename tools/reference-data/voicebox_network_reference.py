"""Independent Voicebox audio/duration networks and full masked-loss gradients.

Paper sections 3.3, 5.1, A.6 and D: symmetric ALiBi (zero flow-token bias),
convolutional position and concatenated symmetric skips. Reduced widths/layers
preserve the complete network structure. This is an equation reference, not an
unreleased Meta checkpoint. Conv weight-normalization dim=0 and sinusoidal time
details are explicitly native choices, not certified private implementation.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(229406)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def slopes(n):
    # The ALiBi author's published symmetric option, linked by Voicebox A.6.
    def powers(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start * start ** i for i in range(count)]
    if math.log2(n).is_integer():
        return powers(n)
    closest = 2 ** math.floor(math.log2(n))
    return powers(closest) + slopes(2 * closest)[0::2][:n - closest]

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.ff_norm = nn.LayerNorm(8), nn.LayerNorm(8)
        self.query, self.key, self.value, self.attention_out = [nn.Linear(8, 8) for _ in range(4)]
        self.ff_in, self.ff_out = nn.Linear(8, 16), nn.Linear(16, 8)

    def forward(self, value, flow_token):
        normalized = self.attention_norm(value)
        q, k, v = [layer(normalized).reshape(1, -1, 2, 4).transpose(1, 2) for layer in (self.query, self.key, self.value)]
        positions = torch.arange(value.shape[1])
        distance = (positions[:, None] - positions[None, :]).abs()
        bias = -torch.tensor(slopes(2))[:, None, None] * distance
        if flow_token:
            bias[:, -1, :] = 0
            bias[:, :, -1] = 0
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(1, -1, 8)
        value = value + self.attention_out(attended)
        return value + self.ff_out(F.gelu(self.ff_in(self.ff_norm(value))))

class Transformer(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.position = nn.ModuleList([torch.nn.utils.weight_norm(nn.Conv1d(8, 8, 3, groups=2, padding=1), dim=0)
            for _ in range(2)])
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.skips = nn.ModuleList([nn.Linear(16, 8) for _ in range(depth // 2)])
        self.final_norm = nn.LayerNorm(8)

    def forward(self, value, flow_token):
        position = value.transpose(1, 2)
        for layer in self.position:
            position = F.gelu(layer(position))
        value = value + position.transpose(1, 2)
        skips = []
        for i, layer in enumerate(self.blocks):
            if i >= len(self.blocks) // 2:
                value = self.skips[i - len(self.blocks) // 2](torch.cat((value, skips.pop()), dim=-1))
            value = layer(value, flow_token)
            if i < len(self.blocks) // 2:
                skips.append(value)
        return self.final_norm(value)

class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.phones = nn.Embedding(8, 4)
        self.input = nn.Linear(10, 8)
        self.transformer = Transformer(4)
        self.output = nn.Linear(8, 3)

    def forward(self, state, context, ids, conditioned):
        phones = self.phones(ids) if conditioned else torch.zeros(1, 6, 4)
        context = context if conditioned else torch.zeros_like(context)
        value = self.input(torch.cat((state, context, phones), dim=-1))
        angles = .37 * 1000 * torch.exp(-math.log(10000.) * torch.arange(0, 8, 2) / 8)
        time = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten()[None, None]
        return self.output(self.transformer(torch.cat((value, time), dim=1), True)[:, :6])

class Duration(nn.Module):
    def __init__(self):
        super().__init__()
        self.phones = nn.Embedding(8, 4)
        self.input = nn.Linear(5, 8)
        self.transformer = Transformer(2)
        self.output = nn.Linear(8, 1)

    def forward(self, context, ids, conditioned):
        phones = self.phones(ids) if conditioned else torch.zeros(1, 4, 4)
        context = context if conditioned else torch.zeros_like(context)
        return self.output(self.transformer(self.input(torch.cat((context, phones), dim=-1)), False))

models = nn.ModuleDict(dict(audio=Audio().eval(), duration=Duration().eval()))
flat = lambda value: value.detach().flatten().tolist()
state = torch.randn(1, 6, 3, requires_grad=True)
audio_context = torch.randn(1, 6, 3, requires_grad=True)
duration_context = torch.tensor([.3, 0., 1.1, 0.])[None, :, None].requires_grad_()
audio_target = torch.randn_like(state)
duration_target = torch.tensor([.7, -.2, 1.3, 1.5])[None, :, None]
audio_ids, duration_ids = torch.tensor([[2, 3, 4, 1, 3, 5]]), torch.tensor([[2, 3, 4, 1]])
audio_mask, duration_mask = torch.tensor([True, False, True, True, False, True]), torch.tensor([False, True, True, True])
result = dict(source=__doc__, state=flat(state), audio_context=flat(audio_context), duration_context=flat(duration_context),
    audio_target=flat(audio_target), duration_target=flat(duration_target), weights={k: flat(p) for k, p in models.named_parameters()}, cases=[])
for conditioned in [True, False]:
    models.zero_grad(set_to_none=True)
    for value in (state, audio_context, duration_context): value.grad = None
    audio = models['audio'](state, audio_context, audio_ids, conditioned)
    duration = models['duration'](duration_context, duration_ids, conditioned)
    loss = F.mse_loss(audio[:, audio_mask], audio_target[:, audio_mask]) + F.l1_loss(duration[:, duration_mask], duration_target[:, duration_mask])
    loss.backward()
    result['cases'].append(dict(conditioned=conditioned, audio=flat(audio), duration=flat(duration), loss=loss.item(),
        gradients={k: flat(p.grad) if p.grad is not None else flat(torch.zeros_like(p)) for k, p in models.named_parameters()},
        state_gradient=flat(state.grad), audio_context_gradient=flat(audio_context.grad) if audio_context.grad is not None else flat(torch.zeros_like(audio_context)),
        duration_context_gradient=flat(duration_context.grad) if duration_context.grad is not None else flat(torch.zeros_like(duration_context))))
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/voicebox_network_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete Voicebox conditional/unconditional networks: {len(result["weights"])} tensors and all loss gradients.')
