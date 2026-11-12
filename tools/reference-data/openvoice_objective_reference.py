"""Independent complete OpenVoice section 2.3 objective reference.

The public release omits its auxiliary trainer. This exercises unchanged author
posterior, flow, reference encoder, decoder, text encoder and discriminator
forwards, with explicit reduced widths and VITS/HiFi-GAN loss equations. It does
not assert that unpublished optimizer settings or IPA inventories were recovered.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import torch
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--source', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--vits-models', required=True, type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.source))
from openvoice.models import SynthesizerTrn, TextEncoder
from openvoice import mel_processing, modules
import librosa

# The converter release omits discriminators. Execute the original VITS
# discriminator classes unchanged, rather than inventing an upstream trainer.
tree = ast.parse(a.vits_models.read_text(encoding='utf-8'))
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
           node.name in ('DiscriminatorP', 'DiscriminatorS', 'MultiPeriodDiscriminator')]
assert len(classes) == 3
scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, modules=modules,
             Conv1d=torch.nn.Conv1d, Conv2d=torch.nn.Conv2d,
             weight_norm=torch.nn.utils.weight_norm, spectral_norm=torch.nn.utils.spectral_norm,
             get_padding=lambda kernel, dilation=1: (kernel * dilation - dilation) // 2,
             LRELU_SLOPE=.1)
exec(compile(ast.Module(body=classes, type_ignores=[]), str(a.vits_models), 'exec'), scope)
MultiPeriodDiscriminator = scope['MultiPeriodDiscriminator']

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(229481)
# Librosa changed these arguments to keyword-only. Use float64 coefficients for
# the double-precision mathematical reference, recorded explicitly in metadata.
mel_processing.librosa_mel_fn = lambda sr, fft, bands, low, high: librosa.filters.mel(
    sr=sr, n_fft=fft, n_mels=bands, fmin=low, fmax=high, dtype=np.float64)

def narrowed_discriminator():
    # Allocate the original large architecture on meta, then substitute only
    # convolution widths/groups. All discriminator forwards remain unchanged.
    with torch.device('meta'):
        result = MultiPeriodDiscriminator()
    for discriminator in result.discriminators:
        for name, old in list(discriminator.named_children()):
            modules = list(old) if isinstance(old, torch.nn.ModuleList) else [old]
            new = []
            for conv in modules:
                inc = max(1, conv.in_channels // 64)
                out = max(1, conv.out_channels // 64)
                groups = conv.groups
                while inc % groups: groups -= 1
                layer = type(conv)(inc, out, conv.kernel_size, conv.stride,
                                   conv.padding, conv.dilation, groups, bias=True)
                new.append(torch.nn.utils.weight_norm(layer))
            setattr(discriminator, name, torch.nn.ModuleList(new) if isinstance(old, torch.nn.ModuleList) else new[0])
    return result.double()

def path(scores):
    # Independent hard monotonic alignment, each token receives >=1 frame.
    n, t = scores.shape
    dp = torch.full((n, t), -float('inf'))
    moves = torch.zeros((n, t), dtype=torch.bool)
    dp[0, 0] = scores[0, 0]
    for frame in range(1, t):
        for token in range(min(n, frame + 1)):
            stay = dp[token, frame - 1]
            advance = dp[token - 1, frame - 1] if token else -float('inf')
            moves[token, frame] = advance > stay
            dp[token, frame] = max(stay, advance) + scores[token, frame]
    durations = torch.zeros(n, dtype=torch.long)
    token = n - 1
    for frame in reversed(range(t)):
        durations[token] += 1
        if moves[token, frame]: token -= 1
    assert token == 0
    return durations

tensors, cases = {}, []
for zero in (False, True):
    label = 'v2' if zero else 'v1'
    model = SynthesizerTrn(17, 9, 4, 4, 8, 1, 2, 3, 0., '1', [3], [[1, 3, 5]],
                          [2], 8, [4], n_speakers=0, gin_channels=3, zero_g=zero).double().eval()
    prior = TextEncoder(17, 4, 4, 8, 1, 2, 3, 0.).double().eval()
    disc = narrowed_discriminator().eval()
    for module in (model, prior, disc):
        with torch.no_grad():
            for name, value in module.named_parameters():
                if 'weight_g' in name or 'gamma' in name or name.endswith('layernorm.weight'):
                    value.uniform_(.7, 1.1)
                else: value.uniform_(-.08, .08)
    audio = .23 * torch.sin(torch.arange(64) * .31) + .07 * torch.cos(torch.arange(64) * .71)
    ids = torch.tensor([[1, 4, 1, 7]])
    spec = mel_processing.spectrogram_torch(audio[None], 16, 22050, 2, 16)
    noise = torch.randn(1, 4, spec.shape[-1])
    speaker = model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
    condition = torch.zeros_like(speaker) if zero else speaker
    with patch('torch.randn_like', side_effect=lambda x: noise):
        posterior, mean, logs, mask = model.enc_q(spec, torch.tensor([spec.shape[-1]]), g=condition, tau=1.)
    transformed = model.flow(posterior, mask, g=speaker)
    generated = model.dec(posterior[:, :, 2:10], g=condition)
    _, pm, pl, _ = prior(ids, torch.tensor([ids.numel()]))
    scores = (-.5 * np.log(2 * np.pi) - pl[:, :, :, None] - .5 *
              (transformed[:, :, None, :] - pm[:, :, :, None]).square() * torch.exp(-2 * pl[:, :, :, None])).sum(1)[0]
    durations = path(scores.detach())
    am = torch.repeat_interleave(pm, durations, dim=2)
    al = torch.repeat_interleave(pl, durations, dim=2)
    kl = ((al - logs - .5 + .5 * (transformed - am).square() * torch.exp(-2 * al)) * mask).sum() / mask.sum()
    real = audio[4:20].reshape(1, 1, -1)
    mel = lambda wave: mel_processing.mel_spectrogram_torch(wave.flatten()[None], 16, 4, 22050, 2, 16, 0, 8000)
    mel_loss = (mel(generated) - mel(real)).abs().mean()
    real_scores, fake_scores, real_features, fake_features = disc(real, generated)
    gan = sum((1 - value).square().mean() for value in fake_scores)
    fm = sum((r.detach() - f).abs().mean() for rs, fs in zip(real_features, fake_features) for r, f in zip(rs, fs))
    loss = 45 * mel_loss + kl + gan + 2 * fm
    groups = {'native': model, 'prior': prior, 'discriminator': disc}
    generator = [(group + '.' + name, value) for group, module in groups.items() if group != 'discriminator'
                 for name, value in module.named_parameters()]
    gradients = torch.autograd.grad(loss, [v for _, v in generator], allow_unused=True)
    used = []
    for (name, value), gradient in zip(generator, gradients):
        if gradient is None: continue
        used.append(name)
        tensors[label + '.gradient.' + name] = gradient.contiguous()
    dloss = sum((1 - r).square().mean() + f.square().mean() for r, f in zip(*disc(real, generated.detach())[:2]))
    dgrads = torch.autograd.grad(dloss, list(disc.parameters()))
    for (name, _), gradient in zip(disc.named_parameters(), dgrads):
        tensors[label + '.discriminator_gradient.' + name] = gradient.contiguous()
    for group, module in groups.items():
        for name, value in module.named_parameters():
            if group == 'native' and name.startswith('enc_p.'): continue
            tensors[label + '.weight.' + group + '.' + name] = value.detach().contiguous()
    for name, value in dict(audio=audio, tokens=ids.double().flatten(), noise=noise,
                            waveform=generated.flatten(), spectrum=spec.transpose(1, 2).squeeze(0),
                            durations=durations.double()).items():
        tensors[label + '.' + name] = value.detach().contiguous()
    cases.append(dict(label=label, zero_condition=zero, parameters=used, generator_loss=loss.item(),
                      discriminator_loss=dloss.item(), mel=mel_loss.item(), kl=kl.item(), gan=gan.item(), fm=fm.item()))
a.output.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(a.output / 'openvoice_objective.safetensors'))
(a.output / 'openvoice_objective.json').write_text(json.dumps(dict(
    source='https://github.com/myshell-ai/OpenVoice',
    revision=subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip(),
    torch=torch.__version__, unchanged_neural_forwards=True, discriminator_width_divisor=64,
    discriminator_source='https://github.com/jaywalnut310/vits/blob/main/models.py',
    discriminator_source_sha256=hashlib.sha256(a.vits_models.read_bytes()).hexdigest(),
    mel_filter_dtype='float64', auxiliary_training_hyperparameters='explicit native baseline, unpublished upstream',
    cases=cases), indent=2) + '\n', encoding='utf-8')
print(json.dumps(cases, indent=2))
