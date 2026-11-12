"""Released UniAudio six-critic checkpoint: unchanged author networks and gradients.

Arguments: original checkout, released universal_model directory, output JSON.
The double variant regenerates only nonlearned Hann/mel buffers using original
constructors at double precision and preserves double through explicit .float().
The original float32 execution is retained as a separate baseline.
"""
import argparse, collections, json, sys, typing
from datetime import timedelta
from pathlib import Path
import numpy as np
import scipy.signal
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from torch.utils._python_dispatch import TorchDispatchMode

checkout, release, destination = map(Path, sys.argv[1:4])
sys.path.insert(0, str(checkout.resolve() / 'codec'))
scipy.signal.kaiser = scipy.signal.windows.kaiser
original_stft = torch.stft
def legacy_stft(*args, **kwargs):
    kwargs.setdefault('return_complex', False)
    return original_stft(*args, **kwargs)
torch.stft = legacy_stft
from modules.discriminators.frequency_discriminator import MultiFrequencyDiscriminator

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dtype = kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else \
            args[1] if func is torch.ops.aten.to.dtype else None
        if args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.float64 and dtype == torch.float32:
            return args[0].clone()
        return func(*args, **kwargs)

torch.set_num_threads(1)
with torch.serialization.safe_globals([argparse.Namespace, DictConfig, ListConfig, ContainerMetadata, Metadata,
        AnyNode, collections.defaultdict, typing.Any, int, dict, list,
        (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype,
        np.dtypes.Float64DType, np.dtypes.Float32DType, timedelta]):
    package = torch.load(release / 'model.pth', weights_only=True, mmap=True, map_location='cpu')
state = package['mfd']
config = OmegaConf.load(release / 'config.yaml').mfd.config
flat = lambda x: x.detach().flatten().tolist()
results = dict(source='Original UniAudio MultiFrequencyDiscriminator; all released mfd tensors',
               learned_tensors=len([x for x in state if x.startswith('discriminators.')]))

def execute(model, dtype):
    t = torch.arange(16384, dtype=dtype)
    real = (.17 * torch.sin(t * .071) + .06 * torch.cos(t * .113))[None, None]
    fake = (real * .93 + .012 * torch.sin(t * .137)[None, None]).detach().requires_grad_()
    real_scores, fake_scores, real_maps, fake_maps = model(real, fake)
    adversarial = sum((x - 1).square().mean() for x in fake_scores) / len(fake_scores)
    matching = sum((a.detach() - b).abs().mean() for aa, bb in zip(real_maps, fake_maps)
                   for a, b in zip(aa, bb)) / sum(map(len, fake_maps))
    generator = adversarial + 20 * matching
    discriminator = sum((a - 1).square().mean() + b.square().mean()
                        for a, b in zip(real_scores, fake_scores)) / len(fake_scores)
    gradient, = torch.autograd.grad(generator, fake)
    return dict(real=flat(real), fake=flat(fake), real_scores=[flat(x) for x in real_scores],
                fake_scores=[flat(x) for x in fake_scores], shapes=[list(x.shape) for x in fake_scores],
                generator=generator.item(), discriminator=discriminator.item(), gradient=flat(gradient))

original = MultiFrequencyDiscriminator(config).float().eval()
original.load_state_dict(state, strict=True)
results['original_float32'] = execute(original, torch.float32)
del original
torch.set_default_dtype(torch.float64)
matched = MultiFrequencyDiscriminator(config).double().eval()
buffers = {name: value.detach().clone() for name, value in matched.state_dict().items()
           if name.startswith('stfts.')}
matched.load_state_dict(state, strict=True)
matched.load_state_dict(buffers, strict=False)
with DoubleOperators():
    results['float64_operators_derived_buffers'] = execute(matched, torch.float64)
destination.write_text(json.dumps(results), encoding='utf-8')
print(f'Generated all six trained critics, {results["learned_tensors"]} learned tensors, G/D losses and waveform gradient.', flush=True)
