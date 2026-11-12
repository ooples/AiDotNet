"""Independent float64 PyTorch oracle for WaveNet 1609.03499, Fig.4, Eqs.1–3.

This implements the paper's causal convolutions, tanh/sigmoid gate, local
conditioning and categorical next-sample likelihood with PyTorch primitives.
Residual/skip scaling and weight normalization are the explicitly documented
r9y9 recipe choices; this is an equation oracle, not a released checkpoint.
Run from the repository root with Python, torch and numpy installed.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
layers = []
parameters = []


def layer(inputs, outputs, kernel, normalized=True, bias=True, transposed=False):
    index = len(layers)
    shape = (inputs, outputs, kernel) if transposed else (outputs, inputs, kernel)
    n = math.prod(shape)
    direction = (.15 * torch.sin(.23 * (torch.arange(n) + 1) + .41 * index)).reshape(shape).requires_grad_()
    gain = (torch.full((shape[0],), .7 + .01 * index).requires_grad_() if normalized else None)
    b = (.03 * torch.cos(.37 * (torch.arange(outputs) + 1) + .19 * index)).requires_grad_() if bias else torch.zeros(outputs)
    parameters.append(direction)
    if gain is not None:
        parameters.append(gain)
    if bias:
        parameters.append(b)
    item = dict(direction=direction, gain=gain, bias=b, use_bias=bias)
    layers.append(item)
    return item


def weight(item):
    v, g = item["direction"], item["gain"]
    return v if g is None else v * (g / torch.linalg.vector_norm(v.flatten(1), dim=1))[:, None, None]


def conv(x, item, dilation=1, causal=False):
    if causal:
        x = F.pad(x, (dilation * (item["direction"].shape[2] - 1), 0))
    return F.conv1d(x, weight(item), item["bias"] if item["use_bias"] else None, dilation=dilation)


scales = [2, 1, 3]
upsamplers = [layer(2, 2, 2 * s - s % 2, normalized=False, transposed=True) for s in scales]
initial = layer(16, 3, 2)
blocks = [(layer(3, 4, 2), layer(2, 4, 1, bias=False), layer(2, 3, 1), layer(2, 3, 1)) for _ in range(3)]
post1, post2 = layer(3, 3, 1), layer(3, 16, 1)
mel = (.2 * torch.sin(torch.arange(6) * .31)).reshape(1, 2, 3).requires_grad_()
audio = .7 * torch.sin(torch.arange(18) * .41)
mu = 15
companded = torch.sign(audio) * torch.log1p(mu * torch.abs(audio)) / math.log1p(mu)
labels = torch.floor((companded + 1) * mu / 2 + .5).long()
previous = torch.cat((torch.tensor([8]), labels[:-1]))
one_hot = F.one_hot(previous, 16).T.unsqueeze(0).to(torch.float64)
condition = mel
for s, item in zip(scales, upsamplers):
    condition = F.conv_transpose1d(condition, weight(item), item["bias"], stride=s, padding=s // 2)
x = conv(one_hot, initial, causal=True)
skips = []
for index, (dilated, local, residual, skip) in enumerate(blocks):
    a = conv(x, dilated, dilation=2 ** (index % 2), causal=True) + conv(condition, local)
    z = torch.tanh(a[:, :2]) * torch.sigmoid(a[:, 2:])
    skips.append(conv(z, skip))
    x = (x + conv(z, residual)) * math.sqrt(.5)
logits = conv(F.relu(conv(F.relu(sum(skips) / math.sqrt(3)), post1)), post2)
log_probabilities = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probabilities, labels.unsqueeze(0))
loss.backward()


def flat(tensor):
    return tensor.detach().reshape(-1).tolist()


data = dict(paper="https://arxiv.org/abs/1609.03499", description=__doc__,
            torch_version=torch.__version__, scales=scales,
            layers=[dict(direction=flat(l["direction"]), gain=None if l["gain"] is None else flat(l["gain"]),
                         bias=flat(l["bias"])) for l in layers],
            learned_scalars=sum(p.numel() for p in parameters), mel=flat(mel), audio=flat(audio), labels=flat(labels),
            logits=flat(logits), log_probabilities=flat(log_probabilities), loss=loss.item(),
            parameter_gradients=[None if p.grad is None else flat(p.grad) for p in parameters], mel_gradient=flat(mel.grad))
schedule_parameter = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = torch.optim.Adam([schedule_parameter], lr=1e-3, betas=(.9, .999), eps=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.5)
data["learning_rates_after_updates"] = []
for _ in range(6):
    optimizer.zero_grad()
    schedule_parameter.square().sum().backward()
    optimizer.step()
    scheduler.step()
    data["learning_rates_after_updates"].append(optimizer.param_groups[0]["lr"])
path = Path("tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wavenet_equations_reference.json")
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"{path}: {len(parameters)} tensors, {data['learned_scalars']} learned scalars, loss {loss.item():.15g}")
