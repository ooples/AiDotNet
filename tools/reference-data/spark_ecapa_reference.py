"""Execute SparkAudio's ECAPA frame path and ASTP pooling. Argument: Spark-TTS checkout.

The public ECAPA constructor keeps its real 1,536-wide aggregation even with eight residual channels.
Only the frame-path weights are exported; pooling is additionally tested with a small independent ASTP instance.
"""
import json
from pathlib import Path
import sys
import types
import torch

root = Path(sys.argv[1]).resolve()
package = types.ModuleType('sparktts')
package.__path__ = [str(root / 'sparktts')]
sys.modules['sparktts'] = package
from sparktts.modules.speaker.ecapa_tdnn import ECAPA_TDNN
from sparktts.modules.speaker.pooling_layers import ASTP

flat = lambda t: t.detach().flatten().tolist()
def state(model, include=lambda name: True):
    return dict(weights={k: flat(v) for k, v in model.state_dict().items() if include(k)},
        weight_shapes={k: list(v.shape) for k, v in model.state_dict().items() if include(k)})
torch.manual_seed(731)
model = ECAPA_TDNN(channels=8, feat_dim=4, embed_dim=6, global_context_att=True).double().eval()
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean.uniform_(-.2, .2)
            module.running_var.uniform_(.5, 1.5)
            module.weight.uniform_(.7, 1.3)
            module.bias.uniform_(-.1, .1)
inputs = torch.randn(1, 5, 4, dtype=torch.float64)
_, frames = model(inputs, True)
data = dict(source='SparkAudio/Spark-TTS/sparktts/modules/speaker/ecapa_tdnn.py ECAPA_TDNN',
    **state(model, lambda k: k.startswith(('layer1.', 'layer2.', 'layer3.', 'layer4.', 'conv.'))),
    input=flat(inputs.transpose(1, 2)), output=flat(frames))
loss = frames.square().mean()
loss.backward()
data['loss'] = loss.item()
data['learning_rate'] = .01
with torch.no_grad():
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.sub_(.01 * parameter.grad)
    data['after_sgd'] = flat(model(inputs, True)[1])
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(out / 'spark_ecapa_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

torch.manual_seed(732)
pool = ASTP(in_dim=6, bottleneck_dim=4, global_context_att=True).double().eval()
inputs = torch.randn(1, 6, 7, dtype=torch.float64)
output = pool(inputs)
data = dict(source='SparkAudio/Spark-TTS/sparktts/modules/speaker/pooling_layers.py ASTP',
    **state(pool), input=flat(inputs), output=flat(output))
loss = output.square().mean()
loss.backward()
data['loss'] = loss.item()
data['learning_rate'] = .01
with torch.no_grad():
    for parameter in pool.parameters():
        parameter.sub_(.01 * parameter.grad)
    data['after_sgd'] = flat(pool(inputs))
(out / 'spark_astp_reference.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print('Wrote Spark ECAPA frame and ASTP references')
