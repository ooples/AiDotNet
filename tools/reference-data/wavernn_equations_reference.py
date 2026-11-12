"""Independent float64 PyTorch evaluation of WaveRNN's primary-paper Equation 2.

The paper (https://arxiv.org/pdf/1802.08435, Section 2) specifies a reset-after
GRU, a structurally masked current-coarse input and two 256-way softmaxes.
This oracle uses separate dense matrices and autograd, rather than the native
convolution implementation. The paper does not specify its conditioning network;
the extra ConvTranspose1d tests AiDotNet's documented mel-conditioning choice.
No native outputs or fitted tolerances are used to construct these expectations.
"""
import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def parameters(layer, shape, output):
    n = math.prod(shape)
    weight = torch.tensor([.15 * math.sin(.23 * (i + 1) + .41 * layer) for i in range(n)]).reshape(shape)
    bias = torch.tensor([.03 * math.cos(.37 * (i + 1) + .19 * layer) for i in range(output)])
    return weight.requires_grad_(), bias.requires_grad_()


specifications = [((2, 2, 4), 2), ((12, 4), 12), ((12, 2), 12),
                  ((6, 1), 6), ((12, 2), 12), ((2, 2), 2),
                  ((256, 2), 256), ((2, 2), 2), ((256, 2), 256)]
p = [parameters(i, shape, output) for i, (shape, output) in enumerate(specifications)]
mel = torch.tensor([[[.2, -.3, .5], [-.4, .1, .6]]], requires_grad=True)
audio = [-.7, .1, .56, -.2, .8, -.4]
encoded = [round((x + 1) * 32767.5) for x in audio]
coarse, fine = [x >> 8 for x in encoded], [x & 255 for x in encoded]
upsampled = F.conv_transpose1d(mel, *p[0], stride=2, padding=1)


def scaled(value):
    return value / 127.5 - 1


h = torch.zeros(4)
states, coarse_logits, fine_logits = [], [], []
for t in range(len(audio)):
    previous = torch.tensor([scaled(coarse[t - 1] if t else 128), scaled(fine[t - 1] if t else 0)])
    # Only the fine half of EACH gate sees the current coarse byte.
    current = F.linear(torch.tensor([scaled(coarse[t])]), *p[3]).reshape(3, 2)
    masked = torch.cat((torch.zeros(3, 2), current), dim=1).flatten()
    inputs = F.linear(previous, *p[2]) + masked + F.linear(upsampled[0, :, t], *p[4])
    recurrent = F.linear(h, *p[1])
    u = torch.sigmoid(recurrent[:4] + inputs[:4])
    r = torch.sigmoid(recurrent[4:8] + inputs[4:8])
    e = torch.tanh(r * recurrent[8:] + inputs[8:])
    h = u * h + (1 - u) * e
    states.append(h)
    coarse_logits.append(F.linear(F.relu(F.linear(h[:2], *p[5])), *p[6]))
    fine_logits.append(F.linear(F.relu(F.linear(h[2:], *p[7])), *p[8]))

c_logits = torch.stack(coarse_logits, dim=1).unsqueeze(0)
f_logits = torch.stack(fine_logits, dim=1).unsqueeze(0)
loss = F.cross_entropy(c_logits, torch.tensor([coarse])) + F.cross_entropy(f_logits, torch.tensor([fine]))
loss.backward()


def values(tensor):
    return tensor.detach().flatten().tolist()


result = {
    "source": "Kalchbrenner et al. 2018, Equation 2 and Figure 1; independent PyTorch dense-matrix autograd",
    "torchVersion": torch.__version__, "dtype": "float64",
    "mel": values(mel), "audio": audio, "coarse": coarse, "fine": fine,
    "states": values(torch.stack(states)), "coarseLogits": values(c_logits), "fineLogits": values(f_logits),
    "loss": loss.item(), "melGradient": values(mel.grad),
    "parameterGradients": [values(t.grad) for pair in p for t in pair],
}
destination = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    pathlib.Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wavernn_equations_reference.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"Saved {destination}; loss={loss.item():.15g}, learned scalars={sum(t.numel() for pair in p for t in pair)}")
