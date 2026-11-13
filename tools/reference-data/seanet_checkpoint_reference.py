"""Execute the original Meta SEANet/LSTM/RVQ codec and save its named state file.

Argument: facebookresearch/encodec checkout. The small deterministic codec exercises
the same original loaders used by VoiceCraft and UniAudio, including EMA buffers.
"""
import json
from pathlib import Path
import sys
import types
import torch
from safetensors.torch import save_file

checkout = Path(sys.argv[1]).resolve()
package = types.ModuleType('encodec'); package.__path__ = [str(checkout / 'encodec')]
sys.modules['encodec'] = package
from encodec.modules.seanet import SEANetEncoder, SEANetDecoder
from encodec.quantization.vq import ResidualVectorQuantizer

torch.manual_seed(2210)
torch.set_num_threads(1)
settings = dict(channels=1, dimension=8, n_filters=2, n_residual_layers=1, ratios=[2, 2],
    causal=True, lstm=1, true_skip=False, compress=2)
encoder = SEANetEncoder(**settings).double().eval()
decoder = SEANetDecoder(**settings).double().eval()
quantizer = ResidualVectorQuantizer(dimension=8, n_q=3, bins=8, kmeans_init=True).double().eval()
with torch.no_grad():
    for layer in quantizer.vq.layers:
        table = layer._codebook
        table.embed.normal_()
        table.cluster_size.fill_(5)
        table.embed_avg.copy_(5 * table.embed)
        table.inited.fill_(1)
state = {}
for prefix, module in [('encoder.', encoder), ('decoder.', decoder), ('quantizer.', quantizer)]:
    state.update({prefix + key: value for key, value in module.state_dict().items()})
flat = lambda tensor: tensor.detach().flatten().tolist()
cases = []
for length in [1, 3, 63, 64, 67]:
    audio = torch.randn(1, 1, length, dtype=torch.float64) * .2
    with torch.no_grad():
        latents = encoder(audio)
        codes = quantizer.encode(latents, frame_rate=4000)
        output = decoder(quantizer.decode(codes))
    cases.append(dict(audio=flat(audio), frames=codes.shape[-1],
        codes=codes[:, 0].flatten().tolist(), waveform=flat(output)))
root = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
(root / 'seanet_checkpoint_reference.json').write_text(json.dumps(dict(
    source='facebookresearch/encodec SEANetEncoder/SEANetDecoder/ResidualVectorQuantizer', cases=cases),
    separators=(',', ':')), encoding='utf-8')
save_file({key: value.float().contiguous().clone() for key, value in state.items()},
    str(root / 'seanet_checkpoint_reference.safetensors'))
print(len(state), 'original named parameters/buffers;', len(cases), 'complete codec cases')
