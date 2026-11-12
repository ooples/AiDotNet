"""Execute the released Spark Perceiver and ResidualFSQ speaker-token path.

Argument: Spark-TTS checkout. Tests all pairs of four-level quantizer indices,
not just encoder-generated values, to exercise digit order and channel-major flattening.
"""
import json
from pathlib import Path
import sys
import types
import torch

root = Path(sys.argv[1]).resolve()
package = types.ModuleType('sparktts'); package.__path__ = [str(root / 'sparktts')]
sys.modules['sparktts'] = package
from sparktts.modules.speaker.perceiver_encoder import PerceiverResampler
from sparktts.modules.fsq.residual_fsq import ResidualFSQ

torch.manual_seed(371)
torch.set_num_threads(1)
flat = lambda t: t.detach().flatten().tolist()
perceiver = PerceiverResampler(dim=8, dim_context=24, num_latents=2, heads=2, depth=1).double().eval()
quantizer = ResidualFSQ(levels=[4, 4], num_quantizers=1, dim=8,
    is_channel_first=True, quantize_dropout=False).double().eval()
project = torch.nn.Linear(16, 6).double().eval()
context = torch.randn(1, 5, 24, dtype=torch.float64)
latent = perceiver(context)
quantized, indices = quantizer(latent.transpose(1, 2))
dvector = project(quantized.reshape(1, -1))
state = {}
for prefix, model in [('perceiver_sampler.', perceiver), ('quantizer.', quantizer), ('project.', project)]:
    state.update({prefix + k: v for k, v in model.state_dict().items()})
cases = []
for i in range(16):
    ids = torch.tensor([[[i], [15 - i]]])
    z = quantizer.get_output_from_indices(ids).transpose(1, 2)
    cases.append(dict(tokens=flat(ids), dvector=flat(project(z.reshape(1, -1)))))
data = dict(source='SparkAudio/Spark-TTS PerceiverResampler + ResidualFSQ + speaker project',
    weights={k: flat(v) for k, v in state.items()}, weight_shapes={k: list(v.shape) for k, v in state.items()},
    context=flat(context), latents=flat(latent), tokens=flat(indices), dvector=flat(dvector), decode_cases=cases)
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(out / 'spark_global_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
