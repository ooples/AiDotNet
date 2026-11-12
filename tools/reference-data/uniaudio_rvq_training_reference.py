"""Run the original UniAudio LSX RVQ training forward, autograd and EMA updates.

Argument: original author checkout. Dead-code case uses one input vector so
replacement sampling is deterministic without replacing the author's RNG.
"""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(sys.argv[1]).resolve() / 'codec'))
from quantization.core_vq_lsx_version import ResidualVectorQuantization

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(2294007)
flat = lambda x: x.detach().flatten().tolist()
fixtures = []
for dead in (False, True):
    model = ResidualVectorQuantization(num_quantizers=3, dim=4, codebook_size=8,
        kmeans_init=False, decay=.99, threshold_ema_dead_code=2).double().train()
    before = []
    for i, layer in enumerate(model.layers):
        cb = layer._codebook
        with torch.no_grad():
            cb.embed.copy_(torch.sin(torch.arange(32).reshape(8, 4) * .17 + i) * .3)
            cb.cluster_size.copy_(torch.arange(8) + 3 if not dead else torch.zeros(8))
            cb.embed_avg.copy_(cb.embed * cb.cluster_size[:, None])
            cb.inited.fill_(1)
        before.append({name: flat(getattr(cb, name)) for name in ('embed', 'embed_avg', 'cluster_size')})
    n = 1 if dead else 5
    x = (torch.cos(torch.arange(4*n).reshape(1, 4, n) * .21) * .2).requires_grad_()
    target = torch.sin(torch.arange(4*n).reshape(1, 4, n) * .13) * .1
    output, codes, commitment = model(x)
    loss = (output - target).square().mean() + commitment.mean()
    loss.backward()
    after = [{name: flat(getattr(layer._codebook, name))
              for name in ('embed', 'embed_avg', 'cluster_size')} for layer in model.layers]
    fixtures.append(dict(dead=dead, shape=list(x.shape), input=flat(x), target=flat(target),
        before=before, after=after, output=flat(output), codes=flat(codes),
        commitment=commitment.mean().item(), loss=loss.item(), gradient=flat(x.grad)))
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root / 'uniaudio_rvq_training_reference.json').write_text(json.dumps(dict(
    source='Original UniAudio codec/quantization/core_vq_lsx_version.py', cases=fixtures)), encoding='utf-8')
print('Generated original UniAudio RVQ warm-codebook and dead-code training references.')
