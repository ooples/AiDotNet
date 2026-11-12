"""Complete reduced DiTTo diffusion Transformer and all masked-loss gradients.

Independent PyTorch paper equations, not unreleased KRAFTON weights. The original
DiT TimestepEmbedder executes unchanged from the supplied pinned author file.
Both original float32 timestep features and a separately labeled double-operator
reference are retained. RoPE acts inside each head, as in the cited RoFormer.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

source = Path(sys.argv[1])
node = next(item for item in ast.parse(source.read_text(encoding='utf-8')).body
            if isinstance(item, ast.ClassDef) and item.name == 'TimestepEmbedder')
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), globals())
torch.manual_seed(229408)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get('dtype') == torch.float32:
            kwargs = dict(kwargs, dtype=torch.float64)
        elif func is torch.ops.aten.to.dtype and args[1] == torch.float32:
            args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.out = [nn.Linear(8, 8) for _ in range(4)]

    def forward(self, value, memory, rotary):
        q, k, v = [projection(x).reshape(1, -1, 2, 4).transpose(1, 2)
                   for projection, x in ((self.q, value), (self.k, memory), (self.v, memory))]
        if rotary:
            def rotate(x):
                angles = torch.arange(x.shape[2])[:, None] / (10000 ** (torch.arange(0, 4, 2) / 4))
                phase = torch.polar(torch.ones_like(angles), angles)
                complex_x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
                return torch.view_as_real(complex_x * phase).flatten(-2)
            q, k = rotate(q), rotate(k)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, -1, 8))

class Geglu(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('in', nn.Linear(8, 32))
        self.out = nn.Linear(16, 8)

    def forward(self, value):
        value, gate = self._modules['in'](value).chunk(2, dim=-1)
        return self.out(value * F.gelu(gate))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_offset = nn.Parameter(.1 * torch.randn(72))
        self.add_module('self', Attention())
        self.cross = Attention()
        self.geglu = Geglu()

    def forward(self, value, memory, shared):
        modulation = (shared + self.adaln_offset).chunk(9, dim=-1)
        def normalized(x, index):
            return F.layer_norm(x, (8,), eps=1e-6) * (1 + modulation[index + 1][:, None]) + modulation[index][:, None]
        hidden = normalized(value, 0)
        value = value + modulation[2][:, None] * self._modules['self'](hidden, hidden, True)
        value = value + modulation[5][:, None] * self.cross(normalized(value, 3), memory, False)
        return value + modulation[8][:, None] * self.geglu(normalized(value, 6))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input, self.text_keys = nn.Linear(5, 8), nn.Linear(6, 8)
        self.time = TimestepEmbedder(8)
        self.text_condition = nn.Sequential(nn.Linear(6, 8), nn.SiLU(), nn.Linear(8, 8))
        self.shared_adaln = nn.Linear(8, 72)
        self.null_text = nn.Parameter(torch.randn(6) * .1)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.long_skip = nn.Sequential(nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 8))
        self.output = nn.Linear(8, 4)

    def forward(self, noisy, mask, text, drop, time=.37):
        value = self.input(torch.cat((noisy, mask), dim=-1))
        first = value
        null = self.null_text[None, None]
        memory = self.text_keys(null if drop else torch.cat((null, text), dim=1))
        pooled = torch.zeros(1, 6) if drop else text.mean(1)
        condition = self.time(torch.tensor([time * 1000.])) + self.text_condition(pooled)
        shared = self.shared_adaln(F.silu(condition))
        for block in self.blocks:
            value = block(value, memory, shared)
        return self.output(F.layer_norm(self.long_skip(torch.cat((first, value), dim=-1)), (8,), eps=1e-6))

model = Model().eval()
noisy = torch.randn(1, 4, 4, requires_grad=True)
mask = torch.tensor([0., 1., 1., 0.]).reshape(1, 4, 1)
text = torch.randn(1, 3, 6, requires_grad=True)
target = torch.randn_like(noisy)
flat = lambda value: value.detach().flatten().tolist()
key = lambda value: value.replace('time.mlp.', 'time.')
original_time = TimestepEmbedder.timestep_embedding(torch.tensor([370.]), 256)
with DoubleOperators():
    double_time = TimestepEmbedder.timestep_embedding(torch.tensor([370.]), 256)
result = dict(source=__doc__, timestep_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    original_float32_timestep=flat(original_time), double_operator_timestep=flat(double_time), noisy=flat(noisy),
    mask=flat(mask), text=flat(text), target=flat(target), weights={key(k): flat(p) for k, p in model.named_parameters()}, cases=[])
for drop in [False, True]:
    model.zero_grad(set_to_none=True); noisy.grad = None; text.grad = None
    with DoubleOperators():
        prediction = model(noisy, mask, text, drop)
    loss = ((prediction - target).square() * mask).sum() / (mask.sum() * 4)
    loss.backward()
    result['cases'].append(dict(drop=drop, prediction=flat(prediction), loss=loss.item(),
        gradients={key(k): flat(p.grad) if p.grad is not None else flat(torch.zeros_like(p)) for k, p in model.named_parameters()},
        noisy_gradient=flat(noisy.grad), text_gradient=flat(text.grad) if text.grad is not None else flat(torch.zeros_like(text))))
result['sampling_cases'] = []
def noise_level(time, scale):
    power = min(1 - 1e-9, max(1e-9, math.cos(time * math.pi / 2) ** 2))
    log_snr = math.log(power / (1 - power)) + 2 * math.log(scale)
    signal_squared = 1 / (1 + math.exp(-log_snr))
    return math.sqrt(signal_squared), math.sqrt(1 - signal_squared)

# Full neural forward, including both CFG branches, executes at every step.
# The finite endpoint clamp is a disclosed native numerical choice, not private
# KRAFTON sampler source. eta=0 DDIM is evaluated using the canonical v rotation.
for prompt_length, steps, guidance, scale in [(0, 1, 0., .3), (1, 2, 1., .3),
        (1, 25, 5., .3), (0, 25, 5., .3), (4, 25, 5., .3), (1, 25, 5., .5)]:
    initial = noisy.detach().clone()
    prompt = target.detach()[:, :prompt_length].clone()
    sampling_mask = torch.ones(1, 4, 1)
    sampling_mask[:, :prompt_length] = 0
    state = initial.clone()
    state[:, :prompt_length] = prompt
    trajectory = []
    with torch.no_grad(), DoubleOperators():
        for step in range(steps, 0, -1):
            time, next_time = step / steps, (step - 1) / steps
            conditional = model(state, sampling_mask, text.detach(), False, time)
            unconditional = model(state, sampling_mask, text.detach(), True, time)
            velocity = torch.lerp(unconditional, conditional, guidance)
            alpha, sigma = noise_level(time, scale)
            next_alpha, next_sigma = noise_level(next_time, scale)
            clean = alpha * state - sigma * velocity
            epsilon = sigma * state + alpha * velocity
            state = next_alpha * clean + next_sigma * epsilon
            state[:, :prompt_length] = prompt
            trajectory.append(flat(state))
    result['sampling_cases'].append(dict(prompt_length=prompt_length, steps=steps,
        guidance=guidance, scale=scale, initial=flat(initial), prompt=flat(prompt),
        trajectory=trajectory, final=flat(state)))

destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/ditto_diffusion_reference.json'
destination.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(f'Complete DiTTo conditioned/unconditioned field: {len(result["weights"])} tensors and all gradients.')
