"""Independent PyTorch equations 1, 3–6 and 9 from CLaM-TTS (arXiv:2404.02781).

The authors have not released this codec. This is an equation reference, not an official model fixture.
Checks the detached mean-field posterior, depth-isolated codebook gradients, learned variance,
monotone codebook scales, and straight-through encoder gradients.
"""
import json
import math
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(502)
books = [torch.randn(4, 3, requires_grad=True) for _ in range(2)]
maximum = torch.tensor([0.13], requires_grad=True)
scale_logits = torch.tensor([0.2, -0.4], requires_grad=True)
log_variance = torch.tensor([-0.3], requires_grad=True)
z = torch.tensor([[0.5, -0.2, 0.8], [-0.4, 0.9, 0.1]], requires_grad=True)
prob = scale_logits.softmax(0)
scales = maximum.exp() * prob.flip(0).cumsum(0).flip(0)
tables = [torch.nn.functional.normalize(book, dim=1) * scales[d] for d, book in enumerate(books)]
selected = []
residual = z.detach()
for table in tables:
    indices = torch.cdist(residual, table.detach()).argmin(1)
    chosen = table[indices]
    selected.append(chosen)
    residual = residual - chosen.detach()
quantized = sum(selected)
variational = 0
for d, table in enumerate(tables):
    target = (z - (quantized - selected[d])).detach()
    squared = (target[:, None, :] - table[None, :, :]).square().sum(-1)
    nll = 0.5 * squared / log_variance.exp() + 0.5 * 3 * (math.log(2 * math.pi) + log_variance)
    posterior = (-nll).softmax(-1).detach()
    variational = variational + (posterior * nll).sum(-1).mean()
commitment = (z - quantized.detach()).square().mean()
straight = z + (quantized - z).detach()
loss = variational + 0.25 * commitment + 0.1 * straight.sum()
loss.backward()
parameters = books + [maximum, scale_logits, log_variance]
data = dict(source='CLaM-TTS equations 1, 3–6, 9; independent PyTorch equation implementation',
    parameters=[p.detach().flatten().tolist() for p in parameters],
    gradients=[p.grad.flatten().tolist() for p in parameters],
    latents=z.detach().flatten().tolist(), latent_gradient=z.grad.flatten().tolist(),
    quantized=straight.detach().flatten().tolist(), variational=variational.item(),
    commitment=commitment.item(), loss=loss.item())
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/clam_rvq_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
