"""CLaM Eq. 6 L1/LSGAN/feature-matching reference in independent PyTorch autograd.

The critic topology is an explicit native implementation choice: CLaM's private
critic details were not released. This verifies the chosen mel critics and loss
gradients; it does not certify equivalence to their private checkpoint.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

torch.manual_seed(406)
torch.set_num_threads(1)
critics = []
windows = [4, 8]
for window, reduction, resolution in [(4, 1, False), (8, 1, False), (0, 1, True), (0, 2, True)]:
    layers = torch.nn.ModuleList()
    for index in range(5 if resolution else 3):
        kernel = (3, 9 if resolution and index < 4 else 3)
        stride = (1 if resolution else 2, 1 if resolution and index in [0, 4] else 2)
        layers.append(torch.nn.Conv2d(1 if index == 0 else 2, 2, kernel, stride,
            padding=(1, kernel[1] // 2)).double())
    layers.append(torch.nn.Conv2d(2, 1, 3, padding=1).double())
    critics.append((window, reduction, layers))
flat = lambda tensor: tensor.detach().flatten().tolist()
parameters = [p for _, _, layers in critics for p in layers.parameters()]
real = torch.randn(8, 4, dtype=torch.float64)
fake = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
starts = [1, 0]

def forward(mel):
    outputs = []
    for index, (window, reduction, layers) in enumerate(critics):
        value = mel[starts[index]:starts[index]+window] if window else mel
        value = value.t()[None, None]
        if reduction > 1: value = F.avg_pool2d(value, reduction)
        features = []
        for conv in layers[:-1]:
            value = F.leaky_relu(conv(value), .2)
            features.append(value)
        value = layers[-1](value)
        features.append(value)
        outputs.append((value, features))
    return outputs

positive = forward(real)
negative = forward(fake)
loss_g = .7 + .25 * .4 + F.l1_loss(fake, real)
for (_, reference), (score, features) in zip(positive, negative):
    loss_g = loss_g + ((score - 1).square().mean() + sum(F.l1_loss(a, b.detach())
        for a, b in zip(features, reference))) / len(critics)
loss_g.backward()
generator_gradient = flat(fake.grad)
for parameter in parameters: parameter.grad = None
loss_d = sum((p[0] - 1).square().mean() + n[0].square().mean()
    for p, n in zip(forward(real), forward(fake.detach()))) / len(critics)
loss_d.backward()
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root / 'ditto_codec_loss_reference.json').write_text(json.dumps(dict(
    source='CLaM-TTS Eq. 6, explicit native critic configuration, PyTorch Conv2d/autograd',
    weights=[flat(parameter) for parameter in parameters], real=flat(real), fake=flat(fake), starts=starts,
    generator_loss=loss_g.item(), discriminator_loss=loss_d.item(), fake_gradient=generator_gradient,
    critic_gradients=[flat(parameter.grad) for parameter in parameters]), separators=(',', ':')), encoding='utf-8')
print('Codec G/D losses', loss_g.item(), loss_d.item())
