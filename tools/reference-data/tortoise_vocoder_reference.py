"""Execute the authors' actual UnivNetGenerator. Argument: TorToiSe checkout."""
import json
from pathlib import Path
import sys
import types
import torch

root = Path(sys.argv[1]).resolve()
package = types.ModuleType('tortoise')
package.__path__ = [str(root / 'tortoise')]
sys.modules['tortoise'] = package
from tortoise.models.vocoder import UnivNetGenerator

torch.manual_seed(379)
model = UnivNetGenerator(noise_dim=4, channel_size=2, dilations=[1, 3], strides=[2, 2],
    hop_length=4, n_mel_channels=4).double()
model.eval()
mel = torch.randn(1, 4, 5, dtype=torch.float64)
noise = torch.randn(1, 4, 5, dtype=torch.float64)
flat = lambda t: t.detach().flatten().tolist()
with torch.no_grad():
    waveform = model(mel, noise)
data = dict(source='neonbjb/tortoise-tts/tortoise/models/vocoder.py UnivNetGenerator',
    weights={k: flat(v) for k, v in model.state_dict().items()},
    weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
    mel=flat(mel), noise=flat(noise), waveform=flat(waveform))
loss = model(mel, noise).square().mean()
loss.backward()
data['loss'] = loss.item()
data['learning_rate'] = .01
with torch.no_grad():
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.sub_(data['learning_rate'] * parameter.grad)
    data['after_sgd'] = flat(model(mel, noise))
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_vocoder_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
